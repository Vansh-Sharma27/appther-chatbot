"""FastAPI streaming endpoint for the Appther RAG chatbot.

Exposes a scale-to-zero Lambda Function URL (no API Gateway) with:
  POST /chat       — Streaming SSE endpoint (cached or fresh RAG answer)
  POST /feedback   — 👍/👎 feedback tied to retrieved chunks
  POST /lead       — Capture leads from the no-answer fallback flow
  GET  /health     — Health check

Security:
  - API key authentication via X-API-Key header (disabled when empty/unset for dev)
  - Rate limiting via slowapi (disabled when REDIS_URL is unset for dev)
  - CORS origin whitelist from CORS_ORIGINS env var
  - Security headers (HSTS, X-Content-Type-Options, X-Frame-Options, CSP)

Key design choices:
  - The RAG query function is injected as a dependency so tests can override it
    with a mock, avoiding the need for a real LanceDB index or API keys.
  - The DynamoDB state layer is injected as a dependency (mockable in tests).
  - On cold start, the LanceDB index is downloaded from S3 to /tmp and cached
    there for the lifetime of the Lambda execution environment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from mangum import Mangum
from pydantic import BaseModel, EmailStr, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.rag import query as rag_query
from api.rag.types import Turn
from api.secrets import inject_env as _inject_secrets
from api.state import AnswerCache, ContentGapLog, FeedbackStore, LeadStore

# Resolve API keys from Secrets Manager at cold start. This runs once per
# execution environment lifetime — subsequent warm invocations reuse the
# cached secrets with zero additional latency.
_inject_secrets()

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_API_KEY: str = os.environ.get("API_AUTH_KEY", "")
_CORS_ORIGINS: list[str] = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS",
        "https://www.appther.com,https://appther.com,https://blog.appther.com",
    ).split(",")
    if o.strip()
]

# ── Pydantic schemas ──────────────────────────────────────────────────────────

_CHAT_QUESTION_MAX = 2000


class ChunkInfoSchema(BaseModel):
    """A single retrieved chunk tied to a feedback entry."""

    chunk_id: str = Field(..., max_length=200)
    url: str = Field(..., max_length=2048)
    score: float = Field(..., ge=0.0, le=1.0)


class TurnSchema(BaseModel):
    role: str = Field(pattern=r"^(user|assistant)$")
    content: str = Field(..., max_length=5000)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=_CHAT_QUESTION_MAX)
    history: list[TurnSchema] = Field(default_factory=list, max_length=50)


class FeedbackRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    answer: str = Field(..., min_length=1, max_length=50000)
    thumbs_up: bool
    chunks: list[ChunkInfoSchema] = Field(default_factory=list, max_length=50)
    reason: str | None = Field(None, max_length=2000)


class LeadRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    email: EmailStr
    question: str = Field(..., min_length=1, max_length=5000)
    phone: str | None = Field(None, max_length=50)
    message: str | None = Field(None, max_length=5000)


# ── Rate limiter ──────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)


# ── Auth dependency ───────────────────────────────────────────────────────────


def verify_api_key(request: Request) -> None:
    """Reject requests missing the required API key.

    When API_AUTH_KEY is empty/not set, authentication is disabled (dev mode).
    """
    if not _API_KEY:
        return  # auth disabled in dev
    header_key = request.headers.get("X-API-Key", "")
    if header_key != _API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key. Provide it via the X-API-Key header.",
        )


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
    The returned callable accepts the same keyword arguments as
    api.rag.query() — including rewritten_query for cache-key correctness.
    """
    return rag_query


# ── SSE helpers ──────────────────────────────────────────────────────────────


def _sse(event: str, data: object) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Token splitting for realistic streaming UX ────────────────────────────────


def _word_tokens(text: str) -> list[str]:
    """Split *text* into word-grouped tokens preserving newlines/markdown.

    Groups 1-2 words per token for natural-feeling streaming. Line breaks
    are preserved as separate "\\n" tokens so the client receives the correct
    structure (Sources list, markdown formatting, etc.).
    """
    lines = text.splitlines(keepends=True)
    tokens: list[str] = []
    for line in lines:
        if line.strip() == "":
            tokens.append("\n")
            continue
        if line == "\n":
            tokens.append("\n")
            continue
        words = line.split()
        i = 0
        while i < len(words):
            chunk = words[i]
            if i + 1 < len(words) and len(chunk) + len(words[i + 1]) + 1 < 40:
                chunk += " " + words[i + 1]
                i += 2
            else:
                i += 1
            tokens.append(chunk + " ")
    if tokens:
        tokens[-1] = tokens[-1].rstrip()
    return tokens


# ── Security headers middleware ────────────────────────────────────────────────


_SELF = "'self'"

_SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": (
        f"default-src {_SELF}; "
        f"script-src {_SELF} 'unsafe-inline'; "
        f"style-src {_SELF} 'unsafe-inline'; "
        f"img-src {_SELF} data: https:; "
        f"connect-src {_SELF} https:; "
        f"frame-ancestors {_SELF}; "
        f"base-uri {_SELF}"
    ),
    "Referrer-Policy": "strict-origin-when-cross-origin",
}


async def _add_security_headers(request: Request, call_next):
    response = await call_next(request)
    for header, value in _SECURITY_HEADERS.items():
        response.headers[header] = value
    return response


# ── App factory ──────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(title="Appther Chatbot API", version="0.1.0")

    # Rate limit read at app-creation time so tests can override via env vars
    rate_limit: str = os.environ.get("RATE_LIMIT", "10/minute")

    # ── Security middleware (order matters: run early) ──────────────────────
    app.middleware("http")(_add_security_headers)

    # ── CORS ────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_CORS_ORIGINS,
        allow_credentials=False,  # widget is cookieless; credentials=False avoids
        # the latent footgun where allow_origins=["*"] with credentials=True
        # would reject all cross-origin responses in the browser.
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    # ── Rate limiting ───────────────────────────────────────────────────────
    # NOTE: slowapi in-memory storage is process-local, NOT shared across
    # Lambda execution environments. In production, rate limiting is enforced
    # at the WAF level (CloudFront + AWS WAF rate-based rules). The app-level
    # limiter provides a coarse guard during local dev only. TODO: wire a
    # shared Redis/DynamoDB-backed store for production if WAF-level enforcement
    # is insufficient.
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Health (unauthenticated, unratelimited) ─────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "appther-chatbot"}

    # ── Chat (streaming) ────────────────────────────────────────────────────

    @app.post("/chat")
    @limiter.limit(rate_limit)
    async def chat(
        request: Request,
        body: ChatRequest,
        cache: Annotated[AnswerCache, Depends(get_cache_store)],
        gap_log: Annotated[ContentGapLog, Depends(get_gap_log)],
        query_fn: Annotated[Any, Depends(get_rag_query_fn)],
    ):
        verify_api_key(request)
        question = body.question
        history = [Turn(t.role, t.content) for t in body.history]  # type: ignore[arg-type]

        async def _stream() -> AsyncGenerator[str, None]:
            # 1. Rewrite to standalone question first (needed for cache key)
            from api.rag.rewrite import rewrite_query

            rewritten = rewrite_query(question, history)

            # 2. Check cache using the REWRITTEN query so follow-ups like
            #    "how much does it cost?" resolve correctly per conversation context.
            cached = cache.get(rewritten)
            if cached is not None:
                answer = cached.get("answer", "")
                sources = cached.get("sources", [])
                model = cached.get("model", "")
                chunks_used = cached.get("chunks_used", 0)
                for token in _word_tokens(answer):
                    yield _sse("answer", {"token": token})
                    await asyncio.sleep(0.015)
                yield _sse("sources", {"sources": sources})
                yield _sse("done", {"model": model, "chunks_used": chunks_used})
                return

            # 3. Run the RAG pipeline with pre-rewritten query so the pipeline
            #    doesn't rewrite again.
            try:
                result = await query_fn(
                    question=question,
                    history=history,
                    rewritten_query=rewritten,
                )
            except Exception:
                logger.exception("RAG query failed")
                yield _sse("error", {"detail": "Failed to generate answer"})
                return

            answer = result.answer
            sources = result.sources
            model = result.model
            chunks_used = result.chunks_used

            # 4. No-answer routing: use structured is_decline signal from RAG
            #    result (language-independent) over the brittle English substring check.
            if result.is_decline:
                gap_log.log(question=question, rewritten_query=result.rewritten_query)
                for token in _word_tokens(answer):
                    yield _sse("answer", {"token": token})
                    await asyncio.sleep(0.015)
                yield _sse("sources", {"sources": sources})
                yield _sse(
                    "lead_suggestion",
                    {"message": "Would you like us to reach out? Provide your details below."},
                )
            else:
                # 5. Cache the successful answer under the REWRITTEN query so
                #    future identical follow-ups produce a cache hit.
                try:
                    cache.set(
                        rewritten,
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

                # 6. Stream answer in word-sized tokens
                for token in _word_tokens(answer):
                    yield _sse("answer", {"token": token})
                    await asyncio.sleep(0.015)

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
    @limiter.limit(rate_limit)
    async def feedback(
        request: Request,
        body: FeedbackRequest,
        store: Annotated[FeedbackStore, Depends(get_feedback_store)],
    ):
        verify_api_key(request)
        try:
            store.store(
                question=body.question,
                answer=body.answer,
                thumbs_up=body.thumbs_up,
                chunks=[c.model_dump() for c in body.chunks],
                reason=body.reason,
            )
        except Exception as exc:
            logger.exception("Failed to store feedback")
            raise HTTPException(status_code=500, detail="Failed to store feedback") from exc
        return {"status": "ok"}

    # ── Lead capture ─────────────────────────────────────────────────────────

    @app.post("/lead")
    @limiter.limit(rate_limit)
    async def lead(
        request: Request,
        body: LeadRequest,
        store: Annotated[LeadStore, Depends(get_lead_store)],
    ):
        verify_api_key(request)
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

# Lambda handler via Mangum (ASGI adapter).
# NOTE: Mangum collects the full ASGI response body and returns it as a single
# Lambda payload, so the carefully chunked SSE tokens are NOT streamed
# incrementally in production — the client waits for the full response then
# receives it all at once. True streaming on Lambda Python requires:
#   - Lambda Function URL with InvokeMode=RESPONSE_STREAM
#   - A custom runtime or Lambda Web Adapter
# This is a known limitation; the streaming UX is accurate during local dev
# (uvicorn) and the code is structured for streaming readiness. See the
# graduation path in the architecture doc for the RESPONSE_STREAM upgrade.
handler = Mangum(app)
