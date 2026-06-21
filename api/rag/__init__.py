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
from api.rag.generate import detect_language, generate_answer, resolve_model
from api.rag.prompt import MAX_QUESTION_CHARS
from api.rag.rerank import voyage_rerank
from api.rag.retrieve import hybrid_retrieve, mmr_select
from api.rag.rewrite import rewrite_query
from api.rag.types import RAGResult, RetrievedChunk, Turn
from crawler.config import LANCE_TABLE_NAME

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
) -> RAGResult:
    """Run the full RAG pipeline and return a RAGResult.

    Steps:
      1. Rewrite follow-up questions to standalone form.
      2. Embed the (possibly rewritten) query.
      3. Hybrid retrieve top_k_retrieve candidates (RRF fusion).
      4. Rerank with Voyage rerank-2.5, keep top_n_rerank.
      5. MMR diversification on the reranked set.
      6. Generate answer with Gemini (model tiered by complexity).
    """
    history = history or []
    uri = index_uri or os.environ.get("LANCE_INDEX_URI", "./lance_index")
    v_key = voyage_api_key or os.environ.get("VOYAGE_API_KEY")
    g_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")

    # Guard: reject or truncate oversized questions before any paid API call
    if len(question) > MAX_QUESTION_CHARS:
        logger.warning("Question truncated from %d to %d chars", len(question), MAX_QUESTION_CHARS)
        question = question[:MAX_QUESTION_CHARS]

    # 1. Rewrite
    rewritten = rewrite_query(question, history, api_key=g_key)
    logger.debug("Query rewritten: %r => %r", question, rewritten)

    # 2. Embed -- resolve provider from index metadata so query model matches ingest
    import lancedb

    db = lancedb.connect(uri)
    tbl = db.open_table(table_name)
    provider = provider_from_index_meta(uri, table_name, api_key=v_key)
    query_vec = embed_query(rewritten, provider=provider)

    # 3. Hybrid retrieve
    candidates = hybrid_retrieve(query_vec, rewritten, tbl, top_k=top_k_retrieve)
    logger.debug("Retrieved %d candidates", len(candidates))

    # 4. Rerank -- keep a wider band so MMR has room to diversify
    reranked = voyage_rerank(rewritten, candidates, top_n=top_n_rerank, api_key=v_key)

    # 5. MMR diversification -- trim to the final diverse set
    diverse = mmr_select(reranked, query_vec, k=top_n_final)
    logger.debug("After MMR: %d chunks", len(diverse))

    # 6. Generate
    answer_parts: list[str] = []
    async for token in generate_answer(question, diverse, history, api_key=g_key):
        answer_parts.append(token)
    answer = "".join(answer_parts)

    sources = list(dict.fromkeys(c.url for c in diverse))
    language = detect_language(question)
    model_used = resolve_model(question, diverse)

    return RAGResult(
        answer=answer,
        sources=sources,
        language=language,
        model=model_used,
        rewritten_query=rewritten,
        chunks_used=len(diverse),
    )
