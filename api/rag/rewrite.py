"""Query rewriting for multi-turn conversations.

When a follow-up question contains pronouns or references that depend on prior
context ("it", "that", "this"), it is rewritten into a fully self-contained
standalone question so the embedding and retrieval steps see a clean query.

Design decisions:
- Only rewrite when there is prior history AND the question contains context-
  dependent pronouns. This avoids an API call for the common case where the
  question is already standalone.
- Bedrock Nova Lite is used for rewrites (IAM auth, no API key, cheap and fast).
- If the API call fails for any reason, the original question is returned
  unchanged so the pipeline continues gracefully.

Public API:
    rewrite_query(question, history) → str
    _bedrock_rewrite(question, history) → str   (internal, mockable in tests)
"""

from __future__ import annotations

import logging
import os

from api.rag.prompt import MAX_QUESTION_CHARS
from api.rag.types import Turn

logger = logging.getLogger(__name__)

# Pronouns / demonstratives that suggest a follow-up referencing prior context.
# Heavy words (it/they/them/these/those) always need rewriting when history
# exists. Light words (we/our/this/that/such) may or may not — they are
# common in standalone questions like "What is our refund policy?".
_HEAVY_CONTEXT_WORDS: frozenset[str] = frozenset(
    {
        "it",
        "its",
        "they",
        "them",
        "their",
        "these",
        "those",
    }
)
_LIGHT_CONTEXT_WORDS: frozenset[str] = frozenset(
    {
        "this",
        "that",
        "which",
        "such",
        "he",
        "she",
        "we",
        "our",
    }
)
_CONTEXT_WORDS: frozenset[str] = _HEAVY_CONTEXT_WORDS | _LIGHT_CONTEXT_WORDS

_MAX_HISTORY_TURNS = 3


def _needs_rewrite(question: str) -> bool:
    """Return True if *question* genuinely needs rewriting.

    A question needs rewriting when it contains context-dependent words AND
    is NOT already self-contained.

    Heavy words (it/they/them/their/these/those) almost always refer to
    prior context and trigger rewriting.
    Light words (we/our/this/that/such) are common in standalone questions
    ("What is our refund policy?") and are skipped when the question starts
    with a wh-word or query verb and has enough words to be standalone.
    """
    words = {w.strip("?.!,").lower() for w in question.split()}
    matching_words = words & _CONTEXT_WORDS
    if not matching_words:
        return False

    # Heavy words always trigger rewrite
    if matching_words & _HEAVY_CONTEXT_WORDS:
        return True

    # Light words: skip if the question is already self-contained
    q_stripped = question.strip()
    first_word = q_stripped.split()[0].strip("?.!,").lower() if q_stripped.split() else ""
    standalone_starters = {
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "which",
        "can",
        "does",
        "do",
        "is",
        "are",
        "will",
        "would",
        "could",
        "should",
        "did",
        "has",
        "have",
        "tell",
        "explain",
        "describe",
    }
    word_count = len(q_stripped.split())
    return not (first_word in standalone_starters and word_count >= 4)


def rewrite_query(
    question: str,
    history: list[Turn],
) -> str:
    """Return a standalone version of *question* given *history*.

    Returns *question* unchanged if there is no prior history or if the
    question is already self-contained (no context-dependent words found).
    """
    if not history or not _needs_rewrite(question):
        return question[:MAX_QUESTION_CHARS]

    try:
        return _bedrock_rewrite(question, history)[:MAX_QUESTION_CHARS]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Query rewrite failed — using original question: %s", exc)
        return question[:MAX_QUESTION_CHARS]


def _bedrock_rewrite(
    question: str,
    history: list[Turn],
) -> str:
    """Call Bedrock Converse (Nova Lite) to rewrite the question as a standalone query."""
    import boto3
    import botocore.config

    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    model = os.getenv("PRIMARY_MODEL", "us.amazon.nova-lite-v1:0")
    timeout = float(os.getenv("BEDROCK_TIMEOUT_SECONDS", "30")) or 30.0
    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=botocore.config.Config(
            read_timeout=int(timeout) + 5,
            connect_timeout=5,
        ),
    )

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

    response = client.converse(
        modelId=model,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 128, "temperature": 0.0},
    )
    rewritten = response["output"]["message"]["content"][0]["text"].strip()
    return rewritten if rewritten else question
