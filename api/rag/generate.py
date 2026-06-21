"""LLM generation with model tiering and streaming.

Pipeline:
  1. If context_chunks is empty → yield a canned decline message immediately
     (no API call wasted).
  2. Otherwise → determine the model tier (Flash-Lite vs Flash via should_escalate),
     build the prompt, call _call_gemini (async generator), yield tokens.

Model tiering:
  - GEMINI_LITE_MODEL (gemini-2.5-flash-lite): simple single-part questions.
  - GEMINI_FLASH_MODEL (gemini-3-flash): complex, comparative, or multi-part queries.

Timeouts:
  - GEMINI_TIMEOUT_SECONDS controls both the HTTP client timeout (passed to
    genai.Client) and the per-stream iteration timeout via asyncio.timeout.
    Default is 30 seconds. Set to 0 to disable (not recommended in Lambda).

Public API:
    generate_answer(question, context_chunks, history, api_key, stream) → AsyncGenerator[str]
    should_escalate(question) → bool
    detect_language(text) → str  (ISO 639-1 code)
    _call_gemini(...)  → AsyncGenerator[str]  (internal, injectable in tests)
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator

from langdetect import detect

from api.rag.prompt import SYSTEM_PROMPT, build_user_message, format_context, format_history
from api.rag.types import RetrievedChunk, Turn

__all__ = [
    "generate_answer",
    "should_escalate",
    "detect_language",
    "resolve_model",
    "NO_CONTEXT_REPLY",
]

logger = logging.getLogger(__name__)

# ── Model configuration ───────────────────────────────────────────────────────

GEMINI_LITE_MODEL: str = os.getenv("GEMINI_LITE_MODEL", "gemini-2.5-flash-lite")
GEMINI_FLASH_MODEL: str = os.getenv("GEMINI_FLASH_MODEL", "gemini-3-flash")

# Timeout for Gemini API calls (HTTP + stream iteration). 30s default.
# In Lambda the function timeout is the ultimate backstop, but this prevents
# a single stalled stream from blocking the concurrent execution slot.
_GEMINI_TIMEOUT: float = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "30")) or 30.0

NO_CONTEXT_REPLY = (
    "I don't have information about that in my current knowledge. "
    "For detailed help, please visit https://www.appther.com/contact-us "
    "or book a free consultation."
)

# ── Model resolution helper ──────────────────────────────────────────────


def resolve_model(question: str, context_chunks: list[RetrievedChunk] | None = None) -> str:
    """Return the model name that *would* be used for *question*.

    This is the single function used by both generate_answer and the query
    pipeline, so the model name in RAGResult always matches what was called.
    """
    if not context_chunks:
        return GEMINI_LITE_MODEL
    return GEMINI_FLASH_MODEL if should_escalate(question) else GEMINI_LITE_MODEL


# ── Escalation heuristic ──────────────────────────────────────────────────────

_ESCALATION_KEYWORDS = frozenset(
    {
        "compare",
        "versus",
        "vs",
        "pros and cons",
        "difference between",
        "tradeoff",
        "trade-off",
        "pros",
        "cons",
    }
)

_DETAIL_KEYWORDS = frozenset(
    {
        "explain",
        "walk me through",
        "step by step",
        "in detail",
        "comprehensive",
        "elaborate",
        "breakdown",
    }
)


def should_escalate(question: str) -> bool:
    """Return True when the question warrants escalation to a stronger model.

    Escalation triggers:
    - More than one question mark (multi-part question)
    - Comparative whole-word keywords ("compare", "vs", "versus", …)
    - Detail/depth keywords ("explain", "walk me through", …)

    Matching is whole-word (word-boundary) to avoid false positives from
    substrings like "cons" inside "consultation".
    """
    if question.count("?") > 1:
        return True
    q_lower = question.lower()
    words = q_lower.split()
    # Multi-word phrases checked first; then individual whole-word keywords.
    if any(
        phrase in q_lower
        for phrase in (
            "pros and cons",
            "difference between",
            "walk me through",
            "step by step",
            "in detail",
        )
    ):
        return True
    single_words = frozenset(words)
    return bool(
        single_words & _ESCALATION_KEYWORDS - {"pros and cons", "difference between"}
        or single_words & _DETAIL_KEYWORDS - {"walk me through", "step by step", "in detail"}
    )


# ── Language detection ────────────────────────────────────────────────────────


def detect_language(text: str) -> str:
    """Detect the ISO 639-1 language code of *text*, default 'en' on failure."""
    try:
        return detect(text) or "en"
    except Exception:  # noqa: BLE001
        return "en"


# ── Internal Gemini caller ────────────────────────────────────────────────────


async def _call_gemini(
    question: str,
    context_str: str,
    history: list[Turn],
    model_name: str,
    api_key: str,
) -> AsyncGenerator[str, None]:
    """Stream tokens from Gemini for the given question + context."""
    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=api_key, http_options={"timeout": _GEMINI_TIMEOUT * 1000})

    history_contents = format_history(history)
    user_msg = build_user_message(question, context_str)
    contents = [*history_contents, {"role": "user", "parts": [{"text": user_msg}]}]

    response = client.models.generate_content_stream(
        model=model_name,
        contents=contents,
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=800,
        ),
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text


# ── Public entry-point ────────────────────────────────────────────────────────


async def generate_answer(
    question: str,
    context_chunks: list[RetrievedChunk],
    history: list[Turn],
    api_key: str | None = None,
) -> AsyncGenerator[str, None]:
    """Yield answer tokens for *question* given *context_chunks* and *history*.

    When context is absent, yields the canned decline message immediately
    without making an API call.
    """
    if not context_chunks:
        yield NO_CONTEXT_REPLY
        return

    model_name = resolve_model(question, context_chunks)
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    context_str = format_context(context_chunks)

    async for token in _call_gemini(question, context_str, history, model_name, key):
        yield token
