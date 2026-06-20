"""Query rewriting for multi-turn conversations.

When a follow-up question contains pronouns or references that depend on prior
context ("it", "that", "this"), it is rewritten into a fully self-contained
standalone question so the embedding and retrieval steps see a clean query.

Design decisions:
- Only rewrite when there is prior history AND the question contains context-
  dependent pronouns. This avoids an API call for the common case where the
  question is already standalone.
- Gemini Flash-Lite is used for rewrites (cheap, fast, no reasoning needed).
- If the API call fails for any reason, the original question is returned
  unchanged so the pipeline continues gracefully.

Public API:
    rewrite_query(question, history, api_key) → str
    _gemini_rewrite(question, history, api_key) → str   (internal, mockable in tests)
"""

from __future__ import annotations

import logging
import os

from api.rag.prompt import MAX_QUESTION_CHARS
from api.rag.types import Turn

logger = logging.getLogger(__name__)

# Pronouns / demonstratives that suggest a follow-up referencing prior context.
_CONTEXT_WORDS: frozenset[str] = frozenset(
    {
        "it",
        "its",
        "this",
        "that",
        "they",
        "them",
        "their",
        "these",
        "those",
        "which",
        "such",
        "he",
        "she",
        "we",
        "our",
    }
)

_MAX_HISTORY_TURNS = 3


def _needs_rewrite(question: str) -> bool:
    words = {w.strip("?.!,").lower() for w in question.split()}
    return bool(words & _CONTEXT_WORDS)


def rewrite_query(
    question: str,
    history: list[Turn],
    api_key: str | None = None,
) -> str:
    """Return a standalone version of *question* given *history*.

    Returns *question* unchanged if there is no prior history or if the
    question is already self-contained (no context-dependent words found).
    """
    if not history or not _needs_rewrite(question):
        return question[:MAX_QUESTION_CHARS]

    try:
        return _gemini_rewrite(question, history, api_key=api_key)[:MAX_QUESTION_CHARS]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Query rewrite failed — using original question: %s", exc)
        return question[:MAX_QUESTION_CHARS]


def _gemini_rewrite(
    question: str,
    history: list[Turn],
    api_key: str | None = None,
) -> str:
    """Call Gemini Flash-Lite to rewrite the question as a standalone query."""
    from google import genai
    from google.genai import types as genai_types

    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    client = genai.Client(api_key=key)

    recent = history[-_MAX_HISTORY_TURNS:]
    conv_lines = []
    for turn in recent:
        role = "User" if turn.role == "user" else "Assistant"
        conv_lines.append(f"{role}: {turn.content}")
    conversation = "\n".join(conv_lines)

    prompt = (
        "Rewrite the follow-up question below as a fully self-contained question "
        "that does not rely on the prior conversation. Output ONLY the rewritten "
        "question, nothing else.\n\n"
        f"Conversation:\n{conversation}\n\n"
        f"Follow-up: {question}\n\n"
        "Standalone question:"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=128,
        ),
    )
    rewritten = response.text.strip()
    return rewritten if rewritten else question
