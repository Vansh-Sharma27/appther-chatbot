"""Tests for the FastAPI streaming endpoint (api/main.py).

Covers:
- POST /chat: streaming SSE response, cache hit, cache miss, no-answer routing
- POST /feedback: store feedback
- POST /lead: capture leads
- GET /health: health check endpoint
- Validation: oversized questions rejected, missing fields
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import (
    create_app,
    get_cache_store,
    get_feedback_store,
    get_gap_log,
    get_lead_store,
    get_rag_query_fn,
)
from api.rag.types import RAGResult

pytestmark = pytest.mark.asyncio


# ── SSE parsing helper ────────────────────────────────────────────────────────


def _parse_sse(body: str) -> list[dict]:
    """Parse raw SSE text into a list of {event, data} dicts."""
    events: list[dict] = []
    current_event = ""
    current_data = ""
    for line in body.split("\n"):
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            current_data = line[6:]
        elif line == "" and current_event:
            try:
                parsed = json.loads(current_data)
            except json.JSONDecodeError:
                parsed = current_data
            events.append({"event": current_event, "data": parsed})
            current_event = ""
            current_data = ""
    return events


def _reconstruct_answer(events: list[dict]) -> str:
    """Reconstruct the full answer from answer token events."""
    tokens: list[str] = []
    for evt in events:
        if evt["event"] == "answer" and isinstance(evt["data"], dict):
            tokens.append(evt["data"].get("token", ""))
    return "".join(tokens)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_mock_query(answer_text: str = "Appther offers ERP solutions."):
    """Return a callable that simulates a successful RAG query."""

    async def mock_query(question, history=None, **kwargs):
        return RAGResult(
            answer=answer_text,
            sources=["https://www.appther.com/services/erp"],
            language="en",
            model="gemini-2.5-flash-lite",
            rewritten_query=question,
            chunks_used=4,
        )

    return mock_query


def _make_mock_no_answer_query():
    """Return a callable that simulates a no-answer-found RAG result."""

    async def mock_query(question, history=None, **kwargs):
        return RAGResult(
            answer=(
                "I don't have information about that in my current knowledge. "
                "For detailed help, please visit https://www.appther.com/contact-us "
                "or book a free consultation."
            ),
            sources=[],
            language="en",
            model="gemini-2.5-flash-lite",
            rewritten_query=question,
            chunks_used=0,
            is_decline=True,
        )

    return mock_query


@pytest.fixture
def test_app():
    """Create a FastAPI app with all dependencies mocked.

    Tests can override specific dependencies by setting
    ``test_app.dependency_overrides[dep_fn] = lambda: mock``.
    """
    _app = create_app()

    # Wire default mocks for all external dependencies
    mocks = {
        "cache": MagicMock(),
        "feedback": MagicMock(),
        "leads": MagicMock(),
        "gaps": MagicMock(),
    }
    mocks["cache"].get.return_value = None

    _app.dependency_overrides[get_cache_store] = lambda: mocks["cache"]
    _app.dependency_overrides[get_feedback_store] = lambda: mocks["feedback"]
    _app.dependency_overrides[get_lead_store] = lambda: mocks["leads"]
    _app.dependency_overrides[get_gap_log] = lambda: mocks["gaps"]
    # Default RAG query override (wrapped in lambda so FastAPI doesn't inspect inner params)
    _app.dependency_overrides[get_rag_query_fn] = lambda: _make_mock_query()

    return _app, mocks


@pytest.fixture
def client(test_app):
    _app, _mocks = test_app
    transport = ASGITransport(app=_app)
    return AsyncClient(transport=transport, base_url="http://test")


# ── GET /health ──────────────────────────────────────────────────────────────


class TestHealth:
    async def test_health_returns_ok(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    async def test_health_has_service_name(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert "service" in data


# ── POST /chat ────────────────────────────────────────────────────────────────


class TestChat:
    async def test_chat_requires_question(self, client):
        resp = await client.post("/chat", json={})
        assert resp.status_code == 422

    async def test_chat_returns_streaming_response(self, client):
        resp = await client.post(
            "/chat",
            json={"question": "What does Appther do?"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

    async def test_chat_stream_contains_answer_event(self, client):
        resp = await client.post(
            "/chat",
            json={"question": "What does Appther do?"},
        )
        text = await resp.aread()
        events = _parse_sse(text.decode())
        assert any(e["event"] == "answer" for e in events)
        answer = _reconstruct_answer(events)
        assert "Appther offers ERP" in answer

    async def test_chat_stream_sources_event(self, client):
        resp = await client.post(
            "/chat",
            json={"question": "What does Appther do?"},
        )
        text = await resp.aread()
        events = _parse_sse(text.decode())
        sources_events = [e for e in events if e["event"] == "sources"]
        assert len(sources_events) >= 1
        sources = sources_events[-1]["data"].get("sources", [])
        assert any("appther.com" in s for s in sources)

    async def test_chat_with_history(self, client):
        history = [
            {"role": "user", "content": "Tell me about Appther"},
            {"role": "assistant", "content": "Appther is an ERP company"},
        ]
        resp = await client.post(
            "/chat",
            json={"question": "What services do they offer?", "history": history},
        )
        assert resp.status_code == 200

    async def test_chat_oversized_question_rejected(self, client):
        long_q = "x" * 5000
        resp = await client.post(
            "/chat",
            json={"question": long_q},
        )
        assert resp.status_code == 422

    async def test_chat_cache_hit(self, test_app):
        """A cached answer should be returned without calling the RAG pipeline."""
        _app, mocks = test_app
        mocks["cache"].get.return_value = {
            "answer": "Cached answer about ERP.",
            "sources": ["https://www.appther.com/faq"],
            "language": "en",
            "model": "gemini-2.5-flash-lite",
            "rewritten_query": "What is ERP?",
            "chunks_used": 3,
        }
        transport = ASGITransport(app=_app)
        async with AsyncClient(transport=transport, base_url="http://test") as cl:
            resp = await cl.post(
                "/chat",
                json={"question": "What is ERP?"},
            )
            assert resp.status_code == 200
            text = await resp.aread()
            events = _parse_sse(text.decode())
            answer = _reconstruct_answer(events)
            assert "Cached answer about ERP" in answer
            mocks["cache"].get.assert_called_once()

    async def test_chat_no_answer_triggers_gap_log(self, test_app):
        """When no answer is found, a content gap should be logged."""
        _app, mocks = test_app
        _app.dependency_overrides[get_rag_query_fn] = lambda: _make_mock_no_answer_query()
        transport = ASGITransport(app=_app)
        async with AsyncClient(transport=transport, base_url="http://test") as cl:
            resp = await cl.post(
                "/chat",
                json={"question": "Some obscure question"},
            )
            assert resp.status_code == 200
            mocks["gaps"].log.assert_called_once()

    async def test_chat_no_answer_has_lead_suggestion(self, test_app):
        """When no answer is found, a lead_suggestion event should be in the stream."""
        _app, mocks = test_app
        _app.dependency_overrides[get_rag_query_fn] = lambda: _make_mock_no_answer_query()
        transport = ASGITransport(app=_app)
        async with AsyncClient(transport=transport, base_url="http://test") as cl:
            resp = await cl.post(
                "/chat",
                json={"question": "Obscure question"},
            )
            assert resp.status_code == 200
            text = await resp.aread()
            events = _parse_sse(text.decode())
            assert any(e["event"] == "lead_suggestion" for e in events)

    async def test_chat_done_event(self, client):
        """The stream should end with a 'done' event containing metadata."""
        resp = await client.post(
            "/chat",
            json={"question": "What does Appther do?"},
        )
        text = await resp.aread()
        events = _parse_sse(text.decode())
        assert events[-1]["event"] == "done"
        assert "chunks_used" in events[-1]["data"]

    async def test_chat_streams_multiple_answer_events(self, client):
        """Answer should be split into multiple tokens for streaming UX."""
        resp = await client.post(
            "/chat",
            json={"question": "What does Appther do?"},
        )
        text = await resp.aread()
        events = _parse_sse(text.decode())
        answer_events = [e for e in events if e["event"] == "answer"]
        assert len(answer_events) >= 2, (
            f"Expected multiple answer tokens, got {len(answer_events)}: "
            f"{[e['data'].get('token', '') for e in answer_events]}"
        )

    async def test_chat_no_duplicate_sources(self, test_app):
        """The no-answer path should not emit duplicate 'sources' events."""
        _app, mocks = test_app
        _app.dependency_overrides[get_rag_query_fn] = lambda: _make_mock_no_answer_query()
        transport = ASGITransport(app=_app)
        async with AsyncClient(transport=transport, base_url="http://test") as cl:
            resp = await cl.post(
                "/chat",
                json={"question": "Obscure question"},
            )
            assert resp.status_code == 200
            text = await resp.aread()
            events = _parse_sse(text.decode())
            sources_count = sum(1 for e in events if e["event"] == "sources")
            assert sources_count == 1, f"Expected exactly 1 sources event, got {sources_count}"


# ── POST /feedback ────────────────────────────────────────────────────────────


class TestFeedback:
    async def test_feedback_requires_question_and_answer(self, client):
        resp = await client.post("/feedback", json={})
        assert resp.status_code == 422

    async def test_feedback_minimal(self, client):
        resp = await client.post(
            "/feedback",
            json={
                "question": "What is ERP?",
                "answer": "ERP stands for Enterprise Resource Planning.",
                "thumbs_up": True,
                "chunks": [
                    {"chunk_id": "c1", "url": "https://www.appther.com/faq", "score": 0.95},
                ],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    async def test_feedback_with_reason(self, client):
        resp = await client.post(
            "/feedback",
            json={
                "question": "Pricing?",
                "answer": "$5000",
                "thumbs_up": False,
                "reason": "Too vague",
                "chunks": [],
            },
        )
        assert resp.status_code == 200

    async def test_feedback_negative(self, client):
        resp = await client.post(
            "/feedback",
            json={
                "question": "Bad answer?",
                "answer": "Wrong info",
                "thumbs_up": False,
                "chunks": [],
            },
        )
        assert resp.status_code == 200


# ── POST /lead ────────────────────────────────────────────────────────────────


class TestLead:
    async def test_lead_requires_name_email_question(self, client):
        resp = await client.post("/lead", json={})
        assert resp.status_code == 422

    async def test_lead_minimal(self, client):
        resp = await client.post(
            "/lead",
            json={
                "name": "John Doe",
                "email": "john@example.com",
                "question": "I need ERP implementation",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    async def test_lead_with_phone(self, client):
        resp = await client.post(
            "/lead",
            json={
                "name": "Jane",
                "email": "jane@example.com",
                "question": "Need CRM",
                "phone": "+1234567890",
                "message": "Please contact me",
            },
        )
        assert resp.status_code == 200

    async def test_lead_invalid_email(self, client):
        resp = await client.post(
            "/lead",
            json={
                "name": "Bad",
                "email": "not-an-email",
                "question": "Test",
            },
        )
        assert resp.status_code == 422
