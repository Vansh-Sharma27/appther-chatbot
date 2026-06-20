"""Tests for the DynamoDB-backed state layer.

Covers:
- AnswerCache: get/set cached answers with TTL
- FeedbackStore: store 👍/👎 feedback tied to retrieved chunk IDs + rerank scores
- LeadStore: capture leads from contact-us routing
- ContentGapLog: log unanswered questions
"""

from __future__ import annotations

import pytest

from api.state import AnswerCache, ContentGapLog, FeedbackStore, LeadStore


@pytest.fixture
def cache_table(dynamo_client):
    """Return an AnswerCache backed by the test DynamoDB table."""
    return AnswerCache(table_name="appther-chatbot-main")


@pytest.fixture
def feedback_store(dynamo_client):
    return FeedbackStore(table_name="appther-chatbot-main")


@pytest.fixture
def lead_store(dynamo_client):
    return LeadStore(table_name="appther-chatbot-main")


@pytest.fixture
def gap_log(dynamo_client):
    return ContentGapLog(table_name="appther-chatbot-main")


# ── AnswerCache ────────────────────────────────────────────────────────────


class TestAnswerCache:
    def test_cache_miss_returns_none(self, cache_table):
        result = cache_table.get("What is ERP?")
        assert result is None

    def test_cache_set_and_get(self, cache_table):
        expected = {
            "answer": "Appther offers ERP implementation services.",
            "sources": ["https://www.appther.com/services/erp"],
            "language": "en",
            "model": "gemini-2.5-flash-lite",
            "rewritten_query": "What is ERP?",
            "chunks_used": 4,
        }
        cache_table.set("What is ERP?", expected)
        result = cache_table.get("What is ERP?")
        assert result == expected

    def test_cache_different_questions_independent(self, cache_table):
        cache_table.set("Question A", {"answer": "Answer A"})
        cache_table.set("Question B", {"answer": "Answer B"})
        assert cache_table.get("Question A")["answer"] == "Answer A"
        assert cache_table.get("Question B")["answer"] == "Answer B"

    def test_cache_overwrite(self, cache_table):
        cache_table.set("Q", {"answer": "v1"})
        cache_table.set("Q", {"answer": "v2"})
        assert cache_table.get("Q")["answer"] == "v2"

    def test_cache_empty_answer_is_not_cached(self, cache_table):
        """Empty/invalid answers should not be cached."""
        with pytest.raises(ValueError, match="Cannot cache empty answer"):
            cache_table.set("Q", {"answer": ""})
        with pytest.raises(ValueError, match="Cannot cache empty answer"):
            cache_table.set("Q", {"answer": None})

    def test_cache_sets_ttl(self, cache_table):
        cache_table.set("Q", {"answer": "test"}, ttl_seconds=3600)
        pk = cache_table._pk("Q")
        client = cache_table._client
        resp = client.get_item(
            TableName=cache_table._table_name,
            Key={"pk": {"S": pk}, "sk": {"S": pk}},
        )
        item = resp.get("Item")
        assert item is not None
        assert "expires_at" in item


# ── FeedbackStore ──────────────────────────────────────────────────────────


class TestFeedbackStore:
    def test_store_feedback(self, feedback_store):
        feedback_store.store(
            question="What is ERP?",
            answer="Appther offers ERP.",
            thumbs_up=True,
            chunks=[
                {"chunk_id": "c1", "url": "https://www.appther.com/faq", "score": 0.95},
                {"chunk_id": "c2", "url": "https://www.appther.com/services", "score": 0.85},
            ],
        )

    def test_store_feedback_with_reason(self, feedback_store):
        feedback_store.store(
            question="Pricing?",
            answer="$5000",
            thumbs_up=False,
            reason="The answer was too vague",
            chunks=[],
        )

    def test_store_feedback_without_chunks(self, feedback_store):
        feedback_store.store(
            question="Test?",
            answer="Test answer",
            thumbs_up=True,
            chunks=[],
        )


# ── LeadStore ──────────────────────────────────────────────────────────────


class TestLeadStore:
    def test_store_lead(self, lead_store):
        lead_store.store(
            name="John Doe",
            email="john@example.com",
            question="I need ERP implementation for my business",
        )

    def test_store_lead_with_phone(self, lead_store):
        lead_store.store(
            name="Jane Doe",
            email="jane@example.com",
            phone="+1234567890",
            question="Interested in Odoo",
        )

    def test_store_lead_with_message(self, lead_store):
        lead_store.store(
            name="Alice",
            email="alice@example.com",
            question="Need CRM",
            message="Please contact me about CRM implementation for a mid-size company",
        )


# ── ContentGapLog ──────────────────────────────────────────────────────────


class TestContentGapLog:
    def test_log_gap(self, gap_log):
        gap_log.log(question="What is the price of Odoo?")

    def test_log_gap_with_history(self, gap_log):
        gap_log.log(
            question="How much does it cost?",
            rewritten_query="How much does Odoo implementation cost?",
        )

    def test_recent_gaps_returns_empty_when_none(self, gap_log):
        assert gap_log.recent(limit=10) == []

    def test_recent_gaps_returns_logged_entries(self, gap_log):
        gap_log.log(question="Q1")
        gap_log.log(question="Q2")
        entries = gap_log.recent(limit=10)
        assert len(entries) == 2
        questions = [e["question"] for e in entries]
        assert "Q1" in questions
        assert "Q2" in questions

    def test_recent_gaps_respects_limit(self, gap_log):
        for i in range(5):
            gap_log.log(question=f"Q{i}")
        entries = gap_log.recent(limit=3)
        assert len(entries) == 3
