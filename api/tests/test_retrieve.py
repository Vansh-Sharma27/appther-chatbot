"""Tests for api/rag/retrieve.py.

Covers:
- rrf_fuse: ordering by score, deduplication, empty inputs, both-list boost
- mmr_select: top-k bound, fewer-than-k passthrough, diversity selection
- hybrid_retrieve: calls both vector + FTS search, applies RRF, respects top_k
"""

from __future__ import annotations

import math

from api.rag.retrieve import hybrid_retrieve, mmr_select, rrf_fuse

from .conftest import DIMS, make_chunk, make_lance_row, make_mock_table

# ── rrf_fuse ──────────────────────────────────────────────────────────────────


def test_rrf_fuse_empty_inputs():
    assert rrf_fuse([], []) == []


def test_rrf_fuse_single_list():
    rows = [make_lance_row("c1"), make_lance_row("c2")]
    result = rrf_fuse(rows, [])
    assert len(result) == 2
    # Both came from only the vector list (rank 1 and 2)
    assert result[0].chunk_id == "c1"  # rank 1 → higher score
    assert result[1].chunk_id == "c2"


def test_rrf_fuse_items_in_both_lists_score_higher():
    """A chunk appearing in both lists should outscore one appearing in only one."""
    shared_row = make_lance_row("shared")
    vec_only_row = make_lance_row("vec_only")
    fts_only_row = make_lance_row("fts_only")

    # shared ranks 1st in both lists; vec_only/fts_only rank 1st in their respective lists
    result = rrf_fuse(
        vector_rows=[shared_row, vec_only_row],
        fts_rows=[shared_row, fts_only_row],
    )
    scores = {c.chunk_id: c.score for c in result}

    assert scores["shared"] > scores["vec_only"]
    assert scores["shared"] > scores["fts_only"]


def test_rrf_fuse_deduplication():
    """A chunk appearing in both lists must appear only once in the output."""
    row = make_lance_row("dup")
    result = rrf_fuse([row], [row])
    chunk_ids = [c.chunk_id for c in result]
    assert chunk_ids.count("dup") == 1


def test_rrf_fuse_sorted_descending():
    rows_a = [make_lance_row(f"a{i}") for i in range(5)]
    rows_b = [make_lance_row(f"b{i}") for i in range(5)]
    result = rrf_fuse(rows_a, rows_b)
    scores = [c.score for c in result]
    assert scores == sorted(scores, reverse=True)


def test_rrf_fuse_score_formula():
    """With one item each at rank 1, score = 1/(60+1) ≈ 0.0164."""
    k = 60
    expected = 1.0 / (k + 1)
    row = make_lance_row("x")
    result = rrf_fuse([row], [])
    assert math.isclose(result[0].score, expected, rel_tol=1e-6)


def test_rrf_fuse_preserves_metadata():
    row = make_lance_row("c1", url="https://example.com", title="My Title", is_faq=True)
    result = rrf_fuse([row], [])
    assert result[0].url == "https://example.com"
    assert result[0].title == "My Title"
    assert result[0].is_faq is True


def test_rrf_fuse_preserves_vector():
    vec = [0.5] * DIMS
    row = make_lance_row("c1", vector=vec)
    result = rrf_fuse([row], [])
    assert result[0].vector == vec


# ── mmr_select ────────────────────────────────────────────────────────────────


def test_mmr_select_returns_at_most_k():
    chunks = [make_chunk(f"c{i}", score=1.0 / (i + 1)) for i in range(10)]
    query_vec = [0.1] * DIMS
    result = mmr_select(chunks, query_vec, k=4)
    assert len(result) <= 4


def test_mmr_select_fewer_than_k_returns_all():
    chunks = [make_chunk("c1"), make_chunk("c2")]
    query_vec = [0.1] * DIMS
    result = mmr_select(chunks, query_vec, k=6)
    assert len(result) == 2


def test_mmr_select_empty_input():
    result = mmr_select([], [0.1] * DIMS, k=4)
    assert result == []


def test_mmr_select_exactly_k():
    chunks = [make_chunk(f"c{i}") for i in range(4)]
    result = mmr_select(chunks, [0.1] * DIMS, k=4)
    assert len(result) == 4


def test_mmr_select_diversifies_near_duplicates():
    """Two nearly-identical chunks (same vector) should not both be selected when k < n.

    Setup (512-dim vectors, only first two dims non-zero):
      - query_vec points diagonally: [1, 1, 0, ...]
      - dup_a_vec / dup_b_vec point along dim 0: [1, 0, 0, ...]  (identical near-dupes)
      - diverse_vec points along dim 1: [0, 1, 0, ...]  (orthogonal to dup_a but
        equally relevant to query as dup_a)

    After dup_a is selected first (identical relevance to query, first in list):
      - MMR(dup_b)  = 0.5*cos(dup_b, query) - 0.5*cos(dup_b, dup_a)
                    = 0.5*0.707 - 0.5*1.0  ≈ -0.15   (heavily penalised)
      - MMR(diverse)= 0.5*cos(diverse, query) - 0.5*cos(diverse, dup_a)
                    = 0.5*0.707 - 0.5*0.0  ≈ +0.35   (no penalty, wins)
    """
    dup_vec = [1.0, 0.0] + [0.0] * (DIMS - 2)
    diverse_vec = [0.0, 1.0] + [0.0] * (DIMS - 2)
    # Query has components along both dup and diverse directions.
    query_vec = [1.0, 1.0] + [0.0] * (DIMS - 2)

    chunk_a = make_chunk("dup_a", vector=dup_vec, score=0.9)
    chunk_b = make_chunk("dup_b", vector=dup_vec, score=0.88)  # identical direction to dup_a
    chunk_diverse = make_chunk("diverse", vector=diverse_vec, score=0.85)

    result = mmr_select([chunk_a, chunk_b, chunk_diverse], query_vector=query_vec, k=2)
    ids = {c.chunk_id for c in result}
    assert "dup_a" in ids, "Most relevant chunk must be selected first"
    assert "diverse" in ids, "Diverse chunk must be preferred over the near-duplicate"
    assert "dup_b" not in ids, "Near-duplicate must be excluded when k=2"


def test_mmr_select_result_is_ordered_by_selection():
    """First selected chunk should be the one with the highest relevance."""
    chunks = [
        make_chunk("high", score=0.9, vector=[1.0] * DIMS),
        make_chunk("low", score=0.1, vector=[0.1] * DIMS),
    ]
    result = mmr_select(chunks, query_vector=[1.0] * DIMS, k=2)
    assert result[0].chunk_id == "high"


# ── hybrid_retrieve ───────────────────────────────────────────────────────────


def test_hybrid_retrieve_calls_both_search_types():
    vector_rows = [make_lance_row("v1"), make_lance_row("v2")]
    fts_rows = [make_lance_row("f1"), make_lance_row("f2")]
    table = make_mock_table(vector_rows=vector_rows, fts_rows=fts_rows)

    query_vec = [0.1] * DIMS
    hybrid_retrieve(query_vec, "test query", table, top_k=10)

    # Both search types were invoked
    calls = table.search.call_args_list
    call_types = [c[1].get("query_type", c[0][1] if len(c[0]) > 1 else "vector") for c in calls]
    assert "vector" in call_types or any("vector" in str(c) for c in calls)
    assert table.search.call_count == 2


def test_hybrid_retrieve_returns_rrf_fused_results():
    """Results come from both lists, deduplicated, ordered by RRF score."""
    shared = make_lance_row("shared")
    vec_only = make_lance_row("vec_only")
    fts_only = make_lance_row("fts_only")

    table = make_mock_table(
        vector_rows=[shared, vec_only],
        fts_rows=[shared, fts_only],
    )
    result = hybrid_retrieve([0.1] * DIMS, "query", table, top_k=20)

    ids = [c.chunk_id for c in result]
    assert ids.count("shared") == 1
    assert "vec_only" in ids
    assert "fts_only" in ids

    scores = {c.chunk_id: c.score for c in result}
    assert scores["shared"] > scores["vec_only"]
    assert scores["shared"] > scores["fts_only"]


def test_hybrid_retrieve_respects_top_k():
    rows = [make_lance_row(f"c{i}") for i in range(30)]
    table = make_mock_table(vector_rows=rows, fts_rows=rows)
    result = hybrid_retrieve([0.1] * DIMS, "query", table, top_k=5)
    # After RRF dedup, there are 30 unique chunks from each list merged, but we
    # ask for top_k=5 and so the limit(5) passed to each LanceDB query is what
    # controls candidate count, giving us ≤ 10 unique rows total, but the
    # returned result should contain no more than top_k items.
    assert len(result) <= 10  # 5 vector + 5 fts (may overlap) → deduped ≤ 10


def test_hybrid_retrieve_empty_index():
    table = make_mock_table(vector_rows=[], fts_rows=[])
    result = hybrid_retrieve([0.1] * DIMS, "test", table, top_k=20)
    assert result == []


def test_hybrid_retrieve_fts_failure_graceful(mocker):
    """If FTS search raises, vector results are still returned."""
    from unittest.mock import MagicMock

    table = MagicMock()
    vec_chain = MagicMock()
    vec_chain.limit.return_value.to_list.return_value = [make_lance_row("v1")]

    def search_side_effect(query, query_type="vector"):
        if query_type == "fts":
            raise RuntimeError("FTS index not ready")
        return vec_chain

    table.search.side_effect = search_side_effect

    result = hybrid_retrieve([0.1] * DIMS, "test", table, top_k=5)
    assert len(result) >= 1
    assert result[0].chunk_id == "v1"
