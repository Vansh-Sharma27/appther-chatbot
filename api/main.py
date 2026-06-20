"""FastAPI streaming endpoint for the Appther RAG chatbot.

Exposes a scale-to-zero Lambda Function URL (no API Gateway) with:
  POST /chat       — Streaming SSE endpoint (cached or fresh RAG answer)
  POST /feedback   — 👍/👎 feedback tied to retrieved chunks
  POST /lead       — Capture leads from the no-answer fallback flow
  GET  /health     — Health check

Key design choices:
  - The RAG query function is injected as a dependency so tests can override it
    with a mock, avoiding the need for a real LanceDB index or API keys.
  - The DynamoDB state layer is injected as a dependency (mockable in tests).
  - On cold start, the LanceDB index is downloaded from S3 to /tmp and cached
    there for the lifetime of the Lambda execution environment.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from mangum import Mangum
from pydantic import BaseModel, EmailStr, Field

from api.rag import query as rag_query
from api.rag.types import Turn
from api.state import AnswerCache, ContentGapLog, FeedbackStore, LeadStore

logger = logging.getLogger(__name__)

# ── Pydantic schemas ──────────────────────────────────────────────────────────

_CHAT_QUESTION_MAX = 2000


class TurnSchema(BaseModel):
    role: str = Field(pattern=r"^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=_CHAT_QUESTION_MAX)
    history: list[TurnSchema] = []


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    thumbs_up: bool
    chunks: list[dict[str, Any]]
    reason: str | None = None


class LeadRequest(BaseModel):
    name: str = Field(..., min_length=1)
    email: EmailStr
    question: str = Field(..., min_length=1)
    phone: str | None = None
    message: str | None = None


# ── Dependencies (injectable, mockable) ─────────────────────────────────────


def get_cache_store() -> AnswerCache:
    table_name = os.environ.get("DYNAMODB_TABLE", "appther-chatbot-main")
    return AnswerCache(table_name=table_name)


def get_feedback_store() -> FeedbackStore:
    table_name = os.environ.get("DYNAMODB_TABLE", "appther-chatbot-main")
    return FeedbackStore(table_name=table_name)


def get_lead_store() -> LeadStore:
    table_name = os.environ.get("DYNAMODB_TABLE", "appther-chatbot-main")
    return LeadStore(table_name=table_name)


def get_gap_log() -> ContentGapLog:
    table_name = os.environ.get("DYNAMODB_TABLE", "appther-chatbot-main")
    return ContentGapLog(table_name=table_name)


async def get_rag_query_fn():
    """Return the RAG query callable.

    Override this in tests to avoid requiring an index or API keys.
    """
    return rag_query


# ── SSE helpers ──────────────────────────────────────────────────────────────


def _sse(event: str, data: object) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


_CACHE_MISS_REPLY = (
    "I don't have information about that in my current knowledge. "
    "For detailed help, please visit https://www.appther.com/contact-us "
    "or book a free consultation."
)


# ── App factory ──────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(title="Appther Chatbot API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ──────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "appther-chatbot"}

    # ── Chat ────────────────────────────────────────────────────────────────

    @app.post("/chat")
    async def chat(
        body: ChatRequest,
        cache: Annotated[AnswerCache, Depends(get_cache_store)],
        gap_log: Annotated[ContentGapLog, Depends(get_gap_log)],
        query_fn: Annotated[Any, Depends(get_rag_query_fn)],
    ):
        question = body.question
        history = [Turn(t.role, t.content) for t in body.history]  # type: ignore[arg-type]

        async def _stream() -> AsyncGenerator[str, None]:
            # 1. Check cache
            cached = cache.get(question)
            if cached is not None:
                answer = cached.get("answer", "")
                sources = cached.get("sources", [])
                model = cached.get("model", "")
                chunks_used = cached.get("chunks_used", 0)
                yield _sse("answer", {"token": answer})
                yield _sse("sources", {"sources": sources})
                yield _sse("done", {"model": model, "chunks_used": chunks_used})
                return

            # 2. Run the RAG pipeline
            try:
                result = await query_fn(
                    question=question,
                    history=history,
                )
            except Exception:
                logger.exception("RAG query failed")
                yield _sse("error", {"detail": "Failed to generate answer"})
                return

            answer = result.answer
            sources = result.sources
            model = result.model
            chunks_used = result.chunks_used

            # 3. No-answer routing: log content gap and suggest contact-us
            if _is_no_answer(answer):
                gap_log.log(question=question, rewritten_query=result.rewritten_query)
                yield _sse("answer", {"token": answer})
                yield _sse("sources", {"sources": sources})
                yield _sse(
                    "lead_suggestion",
                    {"message": "Would you like us to reach out? Provide your details below."},
                )
            else:
                # 4. Cache the successful answer for future hits
                try:
                    cache.set(
                        question,
                        {
                            "answer": answer,
                            "sources": sources,
                            "language": result.language,
                            "model": model,
                            "rewritten_query": result.rewritten_query,
                            "chunks_used": chunks_used,
                        },
                    )
                except Exception:
                    logger.warning("Failed to cache answer", exc_info=True)

                yield _sse("answer", {"token": answer})

            yield _sse("sources", {"sources": sources})
            yield _sse("done", {"model": model, "chunks_used": chunks_used})

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Feedback ────────────────────────────────────────────────────────────

    @app.post("/feedback")
    async def feedback(
        body: FeedbackRequest,
        store: Annotated[FeedbackStore, Depends(get_feedback_store)],
    ):
        try:
            store.store(
                question=body.question,
                answer=body.answer,
                thumbs_up=body.thumbs_up,
                chunks=body.chunks,
                reason=body.reason,
            )
        except Exception as exc:
            logger.exception("Failed to store feedback")
            raise HTTPException(status_code=500, detail="Failed to store feedback") from exc
        return {"status": "ok"}

    # ── Lead capture ─────────────────────────────────────────────────────────

    @app.post("/lead")
    async def lead(
        body: LeadRequest,
        store: Annotated[LeadStore, Depends(get_lead_store)],
    ):
        try:
            store.store(
                name=body.name,
                email=body.email,
                question=body.question,
                phone=body.phone,
                message=body.message,
            )
        except Exception as exc:
            logger.exception("Failed to store lead")
            raise HTTPException(status_code=500, detail="Failed to store lead") from exc
        return {"status": "ok"}

    return app


app = create_app()
handler = Mangum(app)


# ── No-answer detection ──────────────────────────────────────────────────────


def _is_no_answer(answer: str) -> bool:
    """Return True when the answer is a decline message, not a real answer."""
    lowered = answer.lower().strip()
    return (
        "don't have information" in lowered
        or "i don't have" in lowered
        or "no relevant context" in lowered
    )
