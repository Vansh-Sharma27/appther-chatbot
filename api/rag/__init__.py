"""RAG query core — public API for the Step 5 library.

The query() function orchestrates all pipeline stages:
  rewrite -> embed -> hybrid retrieve -> rerank -> MMR -> generate

All stages are independently testable. This module is the integration point
for Step 6 (FastAPI Lambda endpoint).

Usage:
    from api.rag import query, RAGResult, Turn

    result = await query("What ERP services does Appther offer?")
    print(result.answer)
    print(result.sources)
"""

from __future__ import annotations

import logging
import os

from api.rag.embed import embed_query, provider_from_index_meta
from api.rag.generate import NO_CONTEXT_REPLY, detect_language, generate_answer, resolve_model
from api.rag.prompt import MAX_QUESTION_CHARS
from api.rag.rerank import voyage_rerank
from api.rag.retrieve import apply_page_type_boost, hybrid_retrieve, mmr_select
from api.rag.rewrite import rewrite_query
from api.rag.types import RAGResult, RetrievedChunk, Turn
from crawler.config import LANCE_JINA_TABLE_NAME, LANCE_TABLE_NAME

__all__ = ["query", "RAGResult", "Turn", "RetrievedChunk"]

logger = logging.getLogger(__name__)


async def query(
    question: str,
    history: list[Turn] | None = None,
    index_uri: str | None = None,
    table_name: str = LANCE_TABLE_NAME,
    voyage_api_key: str | None = None,
    gemini_api_key: str | None = None,
    top_k_retrieve: int = 20,
    top_n_rerank: int = 12,
    top_n_final: int = 6,
    rewritten_query: str | None = None,
    original_question: str | None = None,
) -> RAGResult:
    """Run the full RAG pipeline and return a RAGResult.

    Steps:
      1. Rewrite follow-up questions to standalone form (skipped when
         *rewritten_query* is provided by the caller).
      2. Embed the (possibly rewritten) query.
      3. Hybrid retrieve top_k_retrieve candidates (RRF fusion).
      4. Rerank with Voyage rerank-2.5, keep top_n_rerank.
      5. Page-type boost applied to rerank scores.
      6. MMR diversification on the boosted reranked set.
      7. Generate answer with Gemini (model tiered by complexity).

    *original_question* is the user's actual phrasing; when provided, it is
    used for language detection and escalation decisions instead of the
    rewritten query.
    """
    history = history or []
    uri = index_uri or os.environ.get("LANCE_INDEX_URI", "./lance_index")
    v_key = voyage_api_key or os.environ.get("VOYAGE_API_KEY")
    g_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")

    q_for_generation = original_question or question
    q_for_retrieval = rewritten_query or question

    # Guard: reject or truncate oversized questions before any paid API call
    if len(question) > MAX_QUESTION_CHARS:
        logger.warning("Question truncated from %d to %d chars", len(question), MAX_QUESTION_CHARS)
        question = question[:MAX_QUESTION_CHARS]

    # 1. Rewrite (only if the caller hasn't already done it)
    if rewritten_query is None:
        rewritten = rewrite_query(question, history, api_key=g_key)
    else:
        rewritten = rewritten_query
    logger.debug("Query rewritten: %r => %r", question, rewritten)

    # 2. Embed + retrieve (with Jina failover)
    import lancedb

    query_vec = None
    candidates = None
    provider = None
    table = None
    db = None
    used_jina = False

    try:
        # Build storage_options for S3 URIs (Bug 17 fix)
        storage_opts = None
        if uri.startswith("s3://"):
            storage_opts = {
                "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", ""),
                "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
                "aws_region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            }
        db = lancedb.connect(uri, storage_options=storage_opts)
        tbl = db.open_table(table_name)
        provider = provider_from_index_meta(uri, table_name, api_key=v_key)
        query_vec = embed_query(rewritten, provider=provider)
        candidates = hybrid_retrieve(query_vec, rewritten, tbl, top_k=top_k_retrieve)
    except Exception as primary_exc:
        logger.warning("Primary (Voyage) path failed, trying Jina standby: %s", primary_exc)
        try:
            jina_table = LANCE_JINA_TABLE_NAME
            jina_tbl = db.open_table(jina_table) if db else lancedb.connect(uri, storage_options=storage_opts).open_table(jina_table)
            jina_provider = provider_from_index_meta(uri, jina_table, api_key=v_key)
            query_vec = embed_query(rewritten, provider=jina_provider)
            candidates = hybrid_retrieve(query_vec, rewritten, jina_tbl, top_k=top_k_retrieve)
            used_jina = True
            logger.info("Jina standby path succeeded for query")
        except Exception as jina_exc:
            logger.error("Both Voyage and Jina paths failed: %s / %s", primary_exc, jina_exc)
            # Return a graceful decline
            return RAGResult(
                answer=NO_CONTEXT_REPLY,
                sources=[],
                language=detect_language(q_for_generation),
                model="",
                rewritten_query=rewritten,
                chunks_used=0,
                is_decline=True,
            )

    logger.debug("Retrieved %d candidates", len(candidates))

    # 4. Rerank
    if candidates:
        reranked = voyage_rerank(rewritten, candidates, top_n=top_n_rerank, api_key=v_key)

        # 5. Page-type boost applied AFTER rerank (Bug 11 fix)
        boosted = apply_page_type_boost(reranked)

        # 6. MMR diversification
        diverse = mmr_select(boosted, query_vec, k=top_n_final)
    else:
        diverse = []
    logger.debug("After MMR: %d chunks", len(diverse))

    # 7. Generate
    if not diverse:
        answer = NO_CONTEXT_REPLY
        sources = []
        language = detect_language(q_for_generation)
        model_used = ""
        is_decline = True
    else:
        answer_parts: list[str] = []
        async for token in generate_answer(q_for_generation, diverse, history, api_key=g_key):
            answer_parts.append(token)
        answer = "".join(answer_parts)
        sources = list(dict.fromkeys(c.url for c in diverse))
        language = detect_language(q_for_generation)
        model_used = resolve_model(q_for_generation, diverse)
        # Structured is_decline: check if the answer is a context-decline
        is_decline = (
            len(diverse) == 0
            or answer.startswith(NO_CONTEXT_REPLY[:30])
            or ("don't have information" in answer.lower() and len(answer) < len(NO_CONTEXT_REPLY) + 50)
        )

    return RAGResult(
        answer=answer,
        sources=sources,
        language=language,
        model=model_used,
        rewritten_query=rewritten,
        chunks_used=len(diverse),
        is_decline=is_decline,
    )
