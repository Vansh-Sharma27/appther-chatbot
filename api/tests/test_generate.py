"""Tests for the generation and pipeline modules.

Covers:
- should_escalate: routing heuristic
- detect_language: language detection with graceful failure
- rewrite_query: standalone question rewriting
- generate_answer: streaming, no-context decline, source citation
- prompt assembly: context + history in final message
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from api.rag.generate import detect_language, generate_answer, should_escalate
from api.rag.prompt import build_user_message, format_context, format_history
from api.rag.rewrite import rewrite_query
from api.rag.types import Turn

from .conftest import make_chunk, make_history

# ── should_escalate ───────────────────────────────────────────────────────────


def test_escalate_simple_faq():
    assert should_escalate("What services does Appther offer?") is False


def test_escalate_multi_question_marks():
    assert should_escalate("What is ERP? How does it work? What are the costs?") is True


def test_escalate_comparative_keyword():
    assert should_escalate("Compare Odoo vs SAP for a mid-size company") is True


def test_escalate_versus_keyword():
    assert should_escalate("Odoo versus Microsoft Dynamics — pros and cons") is True


def test_escalate_detailed_explanation():
    assert should_escalate("Explain in detail how the implementation process works") is True


def test_escalate_walk_through():
    assert should_escalate("Walk me through the ERP migration step by step") is True


def test_escalate_simple_pricing():
    # Single question, no complexity signals → no escalation
    assert should_escalate("What is the price for Odoo implementation?") is False


def test_escalate_consultation_does_not_match_cons_substring():
    """'cons' inside 'consultation' or 'constraints' must NOT trigger escalation."""
    assert should_escalate("What does a consultation cost?") is False
    assert should_escalate("What are the project constraints?") is False


def test_escalate_pros_keyword_only_matches_whole_word():
    assert should_escalate("What are the pros of Odoo?") is True
    assert should_escalate("Tell me about the prospects") is False


# ── detect_language ───────────────────────────────────────────────────────────


def test_detect_language_english():
    lang = detect_language("What services does Appther offer for ERP?")
    assert lang == "en"


def test_detect_language_fallback_on_error():
    with patch("api.rag.generate.detect", side_effect=Exception("langdetect fail")):
        lang = detect_language("short")
    assert lang == "en"


def test_detect_language_short_text_returns_en():
    # Very short text may fail detection; must still return "en" or a valid code
    lang = detect_language("Hi")
    assert isinstance(lang, str)
    assert len(lang) == 2


# ── rewrite_query ─────────────────────────────────────────────────────────────


def test_rewrite_no_history_returns_unchanged():
    q = "What is Appther's main service?"
    assert rewrite_query(q, history=[]) == q


def test_rewrite_self_contained_with_history_returns_unchanged():
    """A clearly standalone question should not trigger a rewrite call."""
    history = make_history(
        ("What does Appther do?", "Appther builds ERP solutions."),
    )
    q = "What industries does Appther serve?"
    result = rewrite_query(q, history=history)
    assert result == q


def test_rewrite_pronoun_reference_triggers_rewrite(mocker):
    """'it' referencing the prior context should be rewritten."""
    history = make_history(
        ("Tell me about Odoo.", "Odoo is an ERP platform that Appther implements."),
    )
    q = "How much does it cost?"

    mock_generate = mocker.patch(
        "api.rag.rewrite._gemini_rewrite",
        return_value="How much does Odoo implementation cost?",
    )
    result = rewrite_query(q, history=history)

    mock_generate.assert_called_once()
    assert result == "How much does Odoo implementation cost?"


def test_rewrite_returns_original_on_gemini_failure(mocker):
    """If Gemini fails during rewrite, return the original question."""
    history = make_history(("Tell me about that feature.", "It is very useful."))
    q = "How does it work?"
    mocker.patch("api.rag.rewrite._gemini_rewrite", side_effect=Exception("API timeout"))
    result = rewrite_query(q, history=history)
    assert result == q


# ── format_context ────────────────────────────────────────────────────────────


def test_format_context_empty():
    result = format_context([])
    assert "No relevant context" in result


def test_format_context_includes_url_and_text():
    chunk = make_chunk("c1", url="https://www.appther.com/faq", text="Appther offers ERP.")
    result = format_context([chunk])
    assert "https://www.appther.com/faq" in result
    assert "Appther offers ERP." in result


def test_format_context_numbered():
    chunks = [make_chunk(f"c{i}", url=f"https://example.com/{i}") for i in range(3)]
    result = format_context(chunks)
    assert "[1]" in result
    assert "[2]" in result
    assert "[3]" in result


def test_format_history_converts_assistant_to_model():
    history = [Turn("user", "Hello"), Turn("assistant", "Hi there")]
    result = format_history(history)
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "model"


def test_format_history_caps_to_max_turns():
    from api.rag.prompt import MAX_HISTORY_TURNS

    history = [Turn("user", f"msg {i}") for i in range(MAX_HISTORY_TURNS + 5)]
    result = format_history(history)
    assert len(result) <= MAX_HISTORY_TURNS


def test_question_length_guard_truncates():
    """Oversized questions should be truncated by the length guard."""
    from api.rag.prompt import MAX_QUESTION_CHARS

    long_q = "a" * (MAX_QUESTION_CHARS + 100)
    # rewrite_query should truncate
    from api.rag.rewrite import rewrite_query

    result = rewrite_query(long_q, history=[], api_key="fake")
    assert len(result) <= MAX_QUESTION_CHARS


def test_question_length_guard_preserves_short():
    short_q = "What is ERP?"
    from api.rag.rewrite import rewrite_query

    result = rewrite_query(short_q, history=[], api_key="fake")
    assert result == short_q


def test_build_user_message_contains_context_and_question():
    msg = build_user_message("What is ERP?", "Context text here.")
    assert "What is ERP?" in msg
    assert "Context text here." in msg


def test_format_context_includes_delimiters():
    """Context must be wrapped in prompt-injection delimiters."""
    chunk = make_chunk("c1", text="Some text")
    result = format_context([chunk])
    assert "BEGIN RETRIEVED CONTEXT" in result
    assert "END RETRIEVED CONTEXT" in result


def test_build_user_message_has_injection_delimiters():
    """User message must have clear context/question boundaries."""
    msg = build_user_message("Test?", "Some context.")
    assert "RETRIEVED CONTEXT" in msg
    assert "END OF CONTEXT" in msg
    assert "USER QUESTION" in msg


def test_system_prompt_has_injection_guard():
    """System prompt must include the security boundary guard."""
    from api.rag.prompt import SYSTEM_PROMPT

    assert "untrusted content" in SYSTEM_PROMPT.lower()
    assert "SECURITY BOUNDARY" in SYSTEM_PROMPT


# ── generate_answer ───────────────────────────────────────────────────────────


async def _collect(gen) -> str:
    parts = []
    async for chunk in gen:
        parts.append(chunk)
    return "".join(parts)


@pytest.mark.asyncio
async def test_generate_declines_when_no_context():
    result = await _collect(
        generate_answer(
            question="What is your pricing?",
            context_chunks=[],
            history=[],
            api_key="fake",
        )
    )
    assert "contact" in result.lower() or "don't have" in result.lower()


@pytest.mark.asyncio
async def test_generate_does_not_call_gemini_when_no_context():
    """No context → decline immediately without an API call."""
    with patch("api.rag.generate._call_gemini") as mock_call:
        result = await _collect(
            generate_answer(
                question="What is pricing?",
                context_chunks=[],
                history=[],
                api_key="fake",
            )
        )
    mock_call.assert_not_called()
    assert result  # non-empty decline message


@pytest.mark.asyncio
async def test_generate_calls_gemini_with_context(mocker):
    """With context chunks, Gemini must be called."""
    chunks = [make_chunk("c1", text="Appther pricing starts at $5000.")]

    async def fake_gemini(*args, **kwargs):
        yield "Appther pricing starts at $5000 for basic ERP."

    mocker.patch("api.rag.generate._call_gemini", side_effect=fake_gemini)
    result = await _collect(
        generate_answer(
            question="What is Appther pricing?",
            context_chunks=chunks,
            history=[],
            api_key="fake",
        )
    )
    assert "5000" in result


@pytest.mark.asyncio
async def test_generate_uses_flash_lite_for_simple_query(mocker):
    """Simple question → Flash-Lite model selected."""
    chunks = [make_chunk()]
    captured = {}

    async def fake_gemini(question, context_str, history, model_name, api_key):
        captured["model"] = model_name
        yield "answer"

    mocker.patch("api.rag.generate._call_gemini", side_effect=fake_gemini)
    await _collect(
        generate_answer(
            question="What does Appther do?",
            context_chunks=chunks,
            history=[],
            api_key="fake",
        )
    )
    from api.rag.generate import GEMINI_LITE_MODEL

    assert captured["model"] == GEMINI_LITE_MODEL


@pytest.mark.asyncio
async def test_generate_escalates_for_complex_query(mocker):
    """Multi-part question → escalation model selected."""
    chunks = [make_chunk()]
    captured = {}

    async def fake_gemini(question, context_str, history, model_name, api_key):
        captured["model"] = model_name
        yield "answer"

    mocker.patch("api.rag.generate._call_gemini", side_effect=fake_gemini)
    await _collect(
        generate_answer(
            question="Compare Odoo vs SAP and explain each one?",
            context_chunks=chunks,
            history=[],
            api_key="fake",
        )
    )
    from api.rag.generate import GEMINI_FLASH_MODEL

    assert captured["model"] == GEMINI_FLASH_MODEL


@pytest.mark.asyncio
async def test_generate_streams_incremental_chunks(mocker):
    """generate_answer must yield multiple tokens, not one big string."""
    chunks = [make_chunk()]
    tokens = ["Appther ", "offers ", "ERP ", "solutions."]

    async def fake_gemini(*args, **kwargs):
        for t in tokens:
            yield t

    mocker.patch("api.rag.generate._call_gemini", side_effect=fake_gemini)
    collected = []
    async for token in generate_answer("What does Appther do?", chunks, [], "fake"):
        collected.append(token)

    assert len(collected) == len(tokens)
    assert "".join(collected) == "Appther offers ERP solutions."
