"""LLM generation with model tiering and streaming via AWS Bedrock Converse.

Pipeline:
  1. If context_chunks is empty → yield a canned decline message immediately
     (no API call wasted).
  2. Otherwise → determine the model tier (Primary vs Escalation via should_escalate),
     build the prompt, call _call_bedrock (async generator), yield tokens.

Model tiering:
  - PRIMARY_MODEL (us.amazon.nova-lite-v1:0 by default): simple single-part questions.
  - ESCALATION_MODEL (us.nvidia.nemotron-3-super-120b-v1:0 by default): complex,
    comparative, or multi-part queries.

Timeouts:
  - BEDROCK_TIMEOUT_SECONDS controls the botocore client's read_timeout (passed to
    Config at client creation). Default is 30 seconds. Set to 0 to disable (not
    recommended in Lambda).

Public API:
    generate_answer(question, context_chunks, history) → AsyncGenerator[str]
    should_escalate(question) → bool
    detect_language(text) → str  (ISO 639-1 code)
    _call_bedrock(...)  → AsyncGenerator[str]  (internal, injectable in tests)
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

from langdetect import detect

from api.rag.prompt import SYSTEM_PROMPT, build_user_message, format_context, format_history
from api.rag.types import RetrievedChunk, Turn

__all__ = [
    "generate_answer",
    "should_escalate",
    "detect_language",
    "resolve_model",
    "NO_CONTEXT_REPLY",
    "PRIMARY_MODEL",
    "ESCALATION_MODEL",
]

logger = logging.getLogger(__name__)

# ── Model configuration ───────────────────────────────────────────────────────

PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "us.amazon.nova-lite-v1:0")
ESCALATION_MODEL: str = os.getenv("ESCALATION_MODEL", "us.nvidia.nemotron-3-super-120b-v1:0")

# Read timeout for the Bedrock client. 0 coerces to 30 — Lambda function
# timeout is the ultimate backstop, but this prevents a stalled stream from
# blocking the execution slot indefinitely.
_BEDROCK_TIMEOUT: float = float(os.getenv("BEDROCK_TIMEOUT_SECONDS", "30")) or 30.0

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
        return PRIMARY_MODEL
    return ESCALATION_MODEL if should_escalate(question) else PRIMARY_MODEL


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


# ── Bedrock client (lazy singleton) ──────────────────────────────────────────

_BEDROCK_CLIENT: Any = None


def _get_bedrock_client() -> Any:
    global _BEDROCK_CLIENT
    if _BEDROCK_CLIENT is None:
        import boto3
        import botocore.config

        _BEDROCK_CLIENT = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            config=botocore.config.Config(
                read_timeout=int(_BEDROCK_TIMEOUT) + 5,
                connect_timeout=5,
                retries={"max_attempts": 2},
            ),
        )
    return _BEDROCK_CLIENT


# ── Internal Bedrock caller ───────────────────────────────────────────────────


async def _call_bedrock(
    question: str,
    context_str: str,
    history: list[Turn],
    model_name: str,
) -> AsyncGenerator[str, None]:
    """Stream tokens from AWS Bedrock Converse for the given question + context.

    Reasoning tokens (<think>…</think>) are stripped inline — Bedrock routes most
    reasoning to reasoningContent delta blocks (filtered by the "text" key check),
    but the stateful stripper handles any that leak into the text stream
    (relevant for the Nemotron escalation path).
    """
    client = _get_bedrock_client()
    messages = [
        *format_history(history),
        {"role": "user", "content": [{"text": build_user_message(question, context_str)}]},
    ]
    response = client.converse_stream(
        modelId=model_name,
        messages=messages,
        system=[{"text": SYSTEM_PROMPT}],
        inferenceConfig={"maxTokens": 800, "temperature": 0.1},
    )

    _OPEN = "<think>"
    _CLOSE = "</think>"
    buf = ""
    in_think = False
    for event in response["stream"]:
        if "contentBlockDelta" not in event:
            continue
        delta = event["contentBlockDelta"]["delta"]
        if "text" not in delta:
            continue  # skip reasoningContent and other non-text delta types
        buf += delta["text"]
        while True:
            if in_think:
                end = buf.find(_CLOSE)
                if end >= 0:
                    buf = buf[end + len(_CLOSE) :]
                    in_think = False
                else:
                    # Keep only a tail long enough to hold a partial </think> tag.
                    if len(buf) > len(_CLOSE) - 1:
                        buf = buf[-(len(_CLOSE) - 1) :]
                    break
            else:
                start = buf.find(_OPEN)
                if start >= 0:
                    if start > 0:
                        yield buf[:start]
                    buf = buf[start + len(_OPEN) :]
                    in_think = True
                else:
                    # Keep a tail long enough to hold a partial <think> tag.
                    if len(buf) > len(_OPEN) - 1:
                        yield buf[: -(len(_OPEN) - 1)]
                        buf = buf[-(len(_OPEN) - 1) :]
                    break
    if buf and not in_think:
        yield buf


# ── Public entry-point ────────────────────────────────────────────────────────


async def generate_answer(
    question: str,
    context_chunks: list[RetrievedChunk],
    history: list[Turn],
) -> AsyncGenerator[str, None]:
    """Yield answer tokens for *question* given *context_chunks* and *history*.

    When context is absent, yields the canned decline message immediately
    without making an API call.
    """
    if not context_chunks:
        yield NO_CONTEXT_REPLY
        return

    model_name = resolve_model(question, context_chunks)
    context_str = format_context(context_chunks)

    async for token in _call_bedrock(question, context_str, history, model_name):
        yield token
