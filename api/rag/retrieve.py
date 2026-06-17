"""Hybrid retrieval: BM25 + vector search fused with RRF, plus MMR diversification.

Pipeline (per query):
  1. hybrid_retrieve(query_vector, fts_query, table, top_k=20)
       → Run LanceDB vector search + FTS search in parallel-ish fashion.
       → Fuse results with rrf_fuse() → top_k candidates ordered by RRF score.
  2. mmr_select(chunks, query_vector, k=6, lambda_=0.5)
       → Apply Maximum Marginal Relevance to the reranked candidates.
       → Diversifies by penalising chunks already similar to selected ones.

Public functions:
    rrf_fuse(vector_rows, fts_rows, k=60) → list[RetrievedChunk]
    mmr_select(chunks, query_vector, k, lambda_) → list[RetrievedChunk]
    hybrid_retrieve(query_vector, fts_query, table, top_k) → list[RetrievedChunk]
"""

from __future__ import annotations

import logging
import math

from api.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard RRF constant; higher → smoother rank weighting


# ── RRF fusion ────────────────────────────────────────────────────────────────


def rrf_fuse(
    vector_rows: list[dict],
    fts_rows: list[dict],
    k: int = _RRF_K,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion of vector-search and FTS result lists.

    score(doc) = Σ  1 / (k + rank_i)   for each list i where doc appears

    Returns chunks deduplicated by chunk_id, sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    rows_by_id: dict[str, dict] = {}

    for rank, row in enumerate(vector_rows, 1):
        cid = row["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        rows_by_id[cid] = row

    for rank, row in enumerate(fts_rows, 1):
        cid = row["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in rows_by_id:
            rows_by_id[cid] = row

    return sorted(
        [_row_to_chunk(rows_by_id[cid], scores[cid]) for cid in rows_by_id],
        key=lambda c: c.score,
        reverse=True,
    )


def _row_to_chunk(row: dict, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row["chunk_id"],
        url=row["url"],
        title=row.get("title", ""),
        page_type=row.get("page_type", ""),
        text=row.get("text", ""),
        score=score,
        is_faq=bool(row.get("is_faq", False)),
        vector=list(row.get("vector") or []),
    )


# ── MMR diversification ───────────────────────────────────────────────────────


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def mmr_select(
    chunks: list[RetrievedChunk],
    query_vector: list[float],
    k: int = 6,
    lambda_: float = 0.5,
) -> list[RetrievedChunk]:
    """Maximum Marginal Relevance selection.

    Greedily selects k chunks that balance relevance to the query with
    dissimilarity from already-selected chunks.

    mmr_score = lambda_ * relevance(chunk, query)
                - (1 - lambda_) * max(sim(chunk, selected_i))

    lambda_=1.0 → pure relevance; lambda_=0.0 → pure diversity.
    """
    if not chunks:
        return []
    if len(chunks) <= k:
        return list(chunks)

    # Pre-compute relevance scores (cosine sim to query) for each chunk.
    # If the chunk has no vector, fall back to the RRF score as relevance.
    relevance: dict[str, float] = {}
    for c in chunks:
        if c.vector:
            relevance[c.chunk_id] = _cosine_sim(c.vector, query_vector)
        else:
            relevance[c.chunk_id] = c.score

    selected: list[RetrievedChunk] = []
    remaining = list(chunks)

    while len(selected) < k and remaining:
        if not selected:
            # First pick: highest relevance
            best = max(remaining, key=lambda c: relevance[c.chunk_id])
        else:
            # Subsequent picks: maximise MMR score
            def mmr_score(c: RetrievedChunk) -> float:
                rel = relevance[c.chunk_id]
                max_sim = max(
                    (_cosine_sim(c.vector, s.vector) if c.vector and s.vector else 0.0)
                    for s in selected
                )
                return lambda_ * rel - (1.0 - lambda_) * max_sim

            best = max(remaining, key=mmr_score)

        selected.append(best)
        remaining.remove(best)

    return selected


# ── Hybrid retrieve ───────────────────────────────────────────────────────────


def hybrid_retrieve(
    query_vector: list[float],
    fts_query: str,
    table,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    """Run vector + FTS search, fuse with RRF, return top_k candidates.

    Both search types are attempted; an FTS failure is logged and vector
    results are returned alone (graceful degradation).
    """
    # Vector search
    vector_rows: list[dict] = (
        table.search(query_vector, query_type="vector").limit(top_k).to_list()
    )

    # FTS search — non-fatal if the index isn't ready yet
    fts_rows: list[dict] = []
    try:
        fts_rows = table.search(fts_query, query_type="fts").limit(top_k).to_list()
    except Exception as exc:  # noqa: BLE001
        logger.warning("FTS search failed (falling back to vector-only): %s", exc)

    return rrf_fuse(vector_rows, fts_rows)
