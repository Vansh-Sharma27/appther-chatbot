"""Voyage rerank-2.5 reranker.

Takes the RRF-fused candidates (top ~20) and reranks them using a cross-encoder
model that reads (query, chunk_text) pairs. This is the single biggest quality
lever: +~14% accuracy for minimal cost.

Public API:
    voyage_rerank(query, chunks, top_n, api_key) → list[RetrievedChunk]
        Returns the input chunks reordered by Voyage rerank-2.5, truncated to top_n.
        Preserves the chunk vectors (needed for subsequent MMR).
"""

from __future__ import annotations

import logging
import os

from api.rag.types import RetrievedChunk
from crawler.config import VOYAGE_RERANK_MODEL

logger = logging.getLogger(__name__)


def voyage_rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_n: int = 6,
    api_key: str | None = None,
) -> list[RetrievedChunk]:
    """Rerank chunks with Voyage rerank-2.5 and return the top_n.

    Falls back to the original RRF ordering if the rerank API is unavailable,
    so retrieval still works during testing without a live API key.
    """
    if not chunks:
        return []

    key = api_key or os.environ.get("VOYAGE_API_KEY", "")
    if not key:
        logger.warning("VOYAGE_API_KEY not set — skipping rerank, using RRF order.")
        return chunks[:top_n]

    try:
        import voyageai

        client = voyageai.Client(api_key=key)
        texts = [c.text for c in chunks]
        result = client.rerank(
            query=query,
            documents=texts,
            model=VOYAGE_RERANK_MODEL,
            top_k=min(top_n, len(chunks)),
        )

        reranked: list[RetrievedChunk] = []
        for item in result.results:
            original = chunks[item.index]
            reranked.append(
                RetrievedChunk(
                    chunk_id=original.chunk_id,
                    url=original.url,
                    title=original.title,
                    page_type=original.page_type,
                    text=original.text,
                    score=float(item.relevance_score),
                    is_faq=original.is_faq,
                    vector=original.vector,
                )
            )
        return reranked

    except Exception as exc:  # noqa: BLE001
        logger.warning("Voyage rerank failed (using RRF order): %s", exc)
        return chunks[:top_n]
