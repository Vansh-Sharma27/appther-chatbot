"""Tests for crawler/index.py.

All tests use a tmp_path LanceDB URI (local file system, no S3).
Validates:
- build_index creates table with correct schema (vector dim=512, all metadata cols)
- Index metadata (model, dims, built_at, chunk_count) is written and readable
- upsert_chunks adds new rows, updates changed rows, skips unchanged rows
- delete_chunks_for_urls removes the expected rows
- smoke_query returns both vector and fts results
- Incremental re-embed: only changed content_hash triggers re-embedding
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from crawler.config import LANCE_TABLE_NAME, VOYAGE_EMBED_DIMS
from crawler.embed import EmbeddedChunk
from crawler.index import (
    _table_exists,
    build_index,
    delete_chunks_for_urls,
    read_index_meta,
    smoke_query,
    upsert_chunks,
    write_index_meta,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_ec(
    chunk_id: str,
    url: str = "https://www.appther.com/faq",
    content_hash: str = "aabbcc",
    text: str = "Appther provides ERP and custom software.",
    page_type: str = "faq",
    is_faq: bool = True,
    vector: list[float] | None = None,
) -> EmbeddedChunk:
    if vector is None:
        vector = [0.0] * VOYAGE_EMBED_DIMS
    return EmbeddedChunk(
        chunk_id=chunk_id,
        url=url,
        title="Appther FAQ",
        page_type=page_type,
        content_hash=content_hash,
        text=text,
        chunk_index=0,
        source="sitemap",
        is_faq=is_faq,
        vector=vector,
        provider="voyage",
        model="voyage-3.5",
        dims=VOYAGE_EMBED_DIMS,
    )


@pytest.fixture
def lance_uri(tmp_path):
    return str(tmp_path / "lance_index")


@pytest.fixture
def sample_chunks():
    return [
        _make_ec(
            "c1",
            url="https://www.appther.com/services/odoo",
            text="Odoo ERP implementation services",
            page_type="service",
        ),
        _make_ec(
            "c2",
            url="https://www.appther.com/services/custom",
            text="Custom software dev",
            page_type="service",
        ),
        _make_ec(
            "c3",
            url="https://www.appther.com/faq",
            text="FAQ about pricing",
            page_type="faq",
            is_faq=True,
        ),
    ]


# ── Schema correctness ────────────────────────────────────────────────────────


def test_build_index_schema(lance_uri, sample_chunks):
    build_index(sample_chunks, uri=lance_uri)

    import lancedb

    db = lancedb.connect(lance_uri)
    assert _table_exists(db, LANCE_TABLE_NAME)
    tbl = db.open_table(LANCE_TABLE_NAME)
    schema = tbl.schema

    field_names = [f.name for f in schema]
    for required in (
        "chunk_id",
        "url",
        "title",
        "page_type",
        "content_hash",
        "text",
        "chunk_index",
        "source",
        "is_faq",
        "vector",
    ):
        assert required in field_names, f"Missing field: {required}"

    vector_field = schema.field("vector")
    assert isinstance(vector_field.type, pa.FixedSizeListType)
    assert vector_field.type.list_size == VOYAGE_EMBED_DIMS
    assert vector_field.type.value_type == pa.float32()


def test_build_index_row_count(lance_uri, sample_chunks):
    build_index(sample_chunks, uri=lance_uri)

    import lancedb

    db = lancedb.connect(lance_uri)
    tbl = db.open_table(LANCE_TABLE_NAME)
    assert tbl.count_rows() == len(sample_chunks)


# ── Index metadata ────────────────────────────────────────────────────────────


def test_build_index_writes_metadata(lance_uri, sample_chunks):
    build_index(sample_chunks, uri=lance_uri)
    meta = read_index_meta(lance_uri, LANCE_TABLE_NAME)

    assert meta["provider"] == "voyage"
    assert meta["model"] == "voyage-3.5"
    assert meta["dims"] == VOYAGE_EMBED_DIMS
    assert meta["chunk_count"] == len(sample_chunks)
    assert "built_at" in meta


def test_metadata_dims_pin(lance_uri, sample_chunks):
    """Metadata dims must equal VOYAGE_EMBED_DIMS — the invariant from the design doc."""
    build_index(sample_chunks, uri=lance_uri)
    meta = read_index_meta(lance_uri, LANCE_TABLE_NAME)
    assert meta["dims"] == 512


def test_write_read_meta_roundtrip(lance_uri):
    payload = {"provider": "voyage", "model": "voyage-3.5", "dims": 512, "chunk_count": 42}
    write_index_meta(lance_uri, LANCE_TABLE_NAME, payload)
    result = read_index_meta(lance_uri, LANCE_TABLE_NAME)
    assert result == payload


def test_read_meta_missing_returns_empty(lance_uri):
    meta = read_index_meta(lance_uri, "nonexistent_table")
    assert meta == {}


# ── build_index overwrites ────────────────────────────────────────────────────


def test_build_index_overwrites_existing(lance_uri, sample_chunks):
    build_index(sample_chunks, uri=lance_uri)
    # Rebuild with only 1 chunk — should wipe the first build
    build_index([sample_chunks[0]], uri=lance_uri)

    import lancedb

    db = lancedb.connect(lance_uri)
    tbl = db.open_table(LANCE_TABLE_NAME)
    assert tbl.count_rows() == 1


def test_build_index_empty_chunks_is_noop(lance_uri):
    build_index([], uri=lance_uri)
    import lancedb

    db = lancedb.connect(lance_uri)
    # Table should NOT have been created
    assert not _table_exists(db, LANCE_TABLE_NAME)


# ── upsert_chunks ─────────────────────────────────────────────────────────────


def test_upsert_creates_table_on_first_run(lance_uri, sample_chunks):
    result = upsert_chunks(sample_chunks, uri=lance_uri)
    assert result["added"] == len(sample_chunks)
    assert result["updated"] == 0
    assert result["skipped"] == 0


def test_upsert_skips_unchanged(lance_uri, sample_chunks):
    upsert_chunks(sample_chunks, uri=lance_uri)
    # Second run with identical chunks — all should be skipped
    result = upsert_chunks(sample_chunks, uri=lance_uri)
    assert result["skipped"] == len(sample_chunks)
    assert result["added"] == 0
    assert result["updated"] == 0


def test_upsert_updates_changed_content_hash(lance_uri, sample_chunks):
    upsert_chunks(sample_chunks, uri=lance_uri)

    # Mutate c1's content_hash and text to simulate a page update
    updated_c1 = _make_ec("c1", content_hash="new_hash", text="Updated Odoo content")
    result = upsert_chunks([updated_c1, sample_chunks[1], sample_chunks[2]], uri=lance_uri)

    assert result["updated"] == 1
    assert result["skipped"] == 2
    assert result["added"] == 0


def test_upsert_adds_new_chunk(lance_uri, sample_chunks):
    upsert_chunks(sample_chunks[:2], uri=lance_uri)
    new_chunk = _make_ec(
        "c99",
        url="https://www.appther.com/case-study/acme",
        page_type="case-study",
    )
    result = upsert_chunks(sample_chunks[:2] + [new_chunk], uri=lance_uri)
    assert result["added"] == 1
    assert result["skipped"] == 2


def test_upsert_empty_is_noop(lance_uri, sample_chunks):
    upsert_chunks(sample_chunks, uri=lance_uri)
    result = upsert_chunks([], uri=lance_uri)
    assert result == {"added": 0, "updated": 0, "skipped": 0}


def test_upsert_total_count_after_operations(lance_uri, sample_chunks):
    """After add + update, row count should equal unique chunk_ids."""
    upsert_chunks(sample_chunks, uri=lance_uri)
    updated = [_make_ec("c1", content_hash="changed")]
    new = [_make_ec("c4")]
    upsert_chunks(sample_chunks[1:] + updated + new, uri=lance_uri)

    import lancedb

    db = lancedb.connect(lance_uri)
    tbl = db.open_table(LANCE_TABLE_NAME)
    assert tbl.count_rows() == 4  # c1, c2, c3, c4


# ── delete_chunks_for_urls ────────────────────────────────────────────────────


def test_delete_chunks_for_urls(lance_uri, sample_chunks):
    build_index(sample_chunks, uri=lance_uri)
    deleted = delete_chunks_for_urls(["https://www.appther.com/faq"], uri=lance_uri)
    assert deleted == 1

    import lancedb

    db = lancedb.connect(lance_uri)
    tbl = db.open_table(LANCE_TABLE_NAME)
    assert tbl.count_rows() == 2


def test_delete_chunks_no_match(lance_uri, sample_chunks):
    build_index(sample_chunks, uri=lance_uri)
    deleted = delete_chunks_for_urls(["https://www.appther.com/nonexistent"], uri=lance_uri)
    assert deleted == 0


def test_delete_chunks_empty_urls(lance_uri, sample_chunks):
    build_index(sample_chunks, uri=lance_uri)
    deleted = delete_chunks_for_urls([], uri=lance_uri)
    assert deleted == 0


def test_delete_on_missing_table_returns_zero(lance_uri):
    deleted = delete_chunks_for_urls(["https://example.com"], uri=lance_uri)
    assert deleted == 0


# ── smoke_query ───────────────────────────────────────────────────────────────


def test_smoke_query_returns_expected_keys(lance_uri, sample_chunks):
    build_index(sample_chunks, uri=lance_uri)
    result = smoke_query(
        query_vector=[0.0] * VOYAGE_EMBED_DIMS,
        fts_query="Odoo ERP",
        uri=lance_uri,
        top_k=3,
    )
    assert "vector" in result
    assert "fts" in result


def test_smoke_query_vector_results_count(lance_uri, sample_chunks):
    build_index(sample_chunks, uri=lance_uri)
    result = smoke_query(
        query_vector=[0.0] * VOYAGE_EMBED_DIMS,
        fts_query="custom software",
        uri=lance_uri,
        top_k=2,
    )
    assert len(result["vector"]) <= 2
    assert len(result["vector"]) >= 1


def test_smoke_query_result_fields(lance_uri, sample_chunks):
    build_index(sample_chunks, uri=lance_uri)
    result = smoke_query(
        query_vector=[0.0] * VOYAGE_EMBED_DIMS,
        fts_query="pricing",
        uri=lance_uri,
    )
    for row in result["vector"]:
        for key in ("chunk_id", "url", "title", "page_type", "text"):
            assert key in row


# ── Jina standby index (separate table) ──────────────────────────────────────


def test_build_jina_standby_index(lance_uri, sample_chunks):
    """Jina standby index must use a different table name but same schema."""
    from crawler.config import LANCE_JINA_TABLE_NAME

    jina_chunks = [
        _make_ec(c.chunk_id, url=c.url, text=c.text, vector=[0.1] * VOYAGE_EMBED_DIMS)
        for c in sample_chunks
    ]
    for ec in jina_chunks:
        ec.provider = "jina"
        ec.model = "jina-embeddings-v3"

    build_index(jina_chunks, uri=lance_uri, table_name=LANCE_JINA_TABLE_NAME)

    import lancedb

    db = lancedb.connect(lance_uri)
    assert _table_exists(db, LANCE_JINA_TABLE_NAME)
    assert not _table_exists(db, LANCE_TABLE_NAME)

    tbl = db.open_table(LANCE_JINA_TABLE_NAME)
    assert tbl.count_rows() == 3

    meta = read_index_meta(lance_uri, LANCE_JINA_TABLE_NAME)
    assert meta["provider"] == "jina"
    assert meta["dims"] == VOYAGE_EMBED_DIMS  # must match for query compatibility


def test_voyage_and_jina_tables_coexist(lance_uri, sample_chunks):
    """Both tables can live under the same URI without interfering."""
    from crawler.config import LANCE_JINA_TABLE_NAME

    build_index(sample_chunks, uri=lance_uri, table_name=LANCE_TABLE_NAME)

    jina_chunks = [_make_ec(c.chunk_id) for c in sample_chunks[:2]]
    for ec in jina_chunks:
        ec.provider = "jina"
    build_index(jina_chunks, uri=lance_uri, table_name=LANCE_JINA_TABLE_NAME)

    import lancedb

    db = lancedb.connect(lance_uri)
    assert _table_exists(db, LANCE_TABLE_NAME)
    assert _table_exists(db, LANCE_JINA_TABLE_NAME)


# ── Metadata dims invariant ───────────────────────────────────────────────────


def test_metadata_model_pinned(lance_uri, sample_chunks):
    """The model name in metadata must match what was used to build the index."""
    build_index(sample_chunks, uri=lance_uri)
    meta = read_index_meta(lance_uri, LANCE_TABLE_NAME)
    # Voyage chunks carry model="voyage-3.5" from _make_ec
    assert meta["model"] == "voyage-3.5"


# ── C2: orphaned-chunk reconciliation on content edits ────────────────────────


def _ids_in_table(lance_uri) -> set[str]:
    import lancedb

    tbl = lancedb.connect(lance_uri).open_table(LANCE_TABLE_NAME)
    return {r["chunk_id"] for r in tbl.search().limit(100_000).to_list()}


def test_upsert_deletes_orphaned_chunks_on_edit(lance_uri):
    """A content edit re-chunks into NEW chunk_ids; old chunk_ids must be removed."""
    url = "https://www.appther.com/industry/odoo"
    v1 = [
        _make_ec("a1", url=url, content_hash="h1", text="section one"),
        _make_ec("a2", url=url, content_hash="h1", text="section two"),
        _make_ec("a3", url=url, content_hash="h1", text="section three"),
    ]
    upsert_chunks(v1, uri=lance_uri)

    # Page edited → brand-new chunk_ids + new content_hash for the same URL.
    v2 = [
        _make_ec("b1", url=url, content_hash="h2", text="new one"),
        _make_ec("b2", url=url, content_hash="h2", text="new two"),
    ]
    result = upsert_chunks(v2, uri=lance_uri)

    assert _ids_in_table(lance_uri) == {"b1", "b2"}  # a1,a2,a3 orphans removed
    assert result["added"] == 2


def test_upsert_shrink_removes_extra_chunks(lance_uri):
    """When a re-ingested URL produces fewer chunks, the extras are deleted."""
    url = "https://www.appther.com/faq"
    upsert_chunks(
        [
            _make_ec("k1", url=url, content_hash="h1", text="a"),
            _make_ec("k2", url=url, content_hash="h1", text="b"),
            _make_ec("k3", url=url, content_hash="h1", text="c"),
        ],
        uri=lance_uri,
    )
    # Only k1 survives the new chunking (unchanged); k2/k3 no longer produced.
    result = upsert_chunks([_make_ec("k1", url=url, content_hash="h1", text="a")], uri=lance_uri)
    assert _ids_in_table(lance_uri) == {"k1"}
    assert result["skipped"] == 1


def test_upsert_leaves_urls_absent_from_batch_untouched(lance_uri):
    """Reconciliation must never delete chunks for URLs not present in the batch."""
    url_a = "https://www.appther.com/a"
    url_b = "https://www.appther.com/b"
    upsert_chunks(
        [
            _make_ec("a1", url=url_a, content_hash="h1", text="a1"),
            _make_ec("b1", url=url_b, content_hash="h1", text="b1"),
            _make_ec("b2", url=url_b, content_hash="h1", text="b2"),
        ],
        uri=lance_uri,
    )
    # Re-ingest only url_a with an entirely new chunk set.
    upsert_chunks([_make_ec("a9", url=url_a, content_hash="h2", text="a9")], uri=lance_uri)
    # url_a reconciled to just a9; url_b fully preserved (it was not crawled).
    assert _ids_in_table(lance_uri) == {"a9", "b1", "b2"}


def test_upsert_orphan_count_recorded_in_meta(lance_uri):
    """Orphan deletions are observable via index metadata."""
    url = "https://www.appther.com/svc"
    upsert_chunks(
        [
            _make_ec("o1", url=url, content_hash="h1", text="x"),
            _make_ec("o2", url=url, content_hash="h1", text="y"),
        ],
        uri=lance_uri,
    )
    upsert_chunks([_make_ec("o3", url=url, content_hash="h2", text="z")], uri=lance_uri)
    meta = read_index_meta(lance_uri, LANCE_TABLE_NAME)
    assert meta["last_orphans_deleted"] == 2


def test_upsert_unchanged_recrawl_deletes_nothing(lance_uri):
    """Idempotency: re-ingesting identical content reconciles zero orphans."""
    url = "https://www.appther.com/stable"
    chunks = [
        _make_ec("s1", url=url, content_hash="h1", text="one"),
        _make_ec("s2", url=url, content_hash="h1", text="two"),
    ]
    upsert_chunks(chunks, uri=lance_uri)
    result = upsert_chunks(chunks, uri=lance_uri)
    assert result == {"added": 0, "updated": 0, "skipped": 2}
    assert _ids_in_table(lance_uri) == {"s1", "s2"}
