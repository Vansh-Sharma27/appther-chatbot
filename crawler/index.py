"""LanceDB index builder and query helpers.

Architecture:
- One primary LanceDB table ('chunks') built with Voyage embeddings (512-dim float32).
- One standby table ('chunks_jina') built with Jina embeddings (same 512-dim schema).
- Both tables live under *uri* (a local path or s3:// URI).
- Index metadata (.index_meta.json) pins model + dims + build timestamp so the query
  layer can assert it reads with the same model used at ingest.

Key functions:
  build_index(embedded_chunks, uri, table_name, …)
      Create or overwrite a LanceDB table from EmbeddedChunk objects.

  upsert_chunks(embedded_chunks, uri, table_name, …)
      Incremental upsert keyed by chunk_id: add new chunks, replace chunks whose
      content_hash changed, skip unchanged ones, and delete stale chunks that a
      re-ingested URL no longer produces (prevents orphaned chunks after a content
      edit, since chunk_id is derived from chunk text + index). URLs absent from the
      batch are left untouched — vanished-URL cleanup is delete_chunks_for_urls().
      Idempotent — re-running on identical content is a no-op.

  delete_chunks_for_urls(urls, uri, table_name)
      Remove all chunks for the given canonical URLs (called when a page 404s).

  smoke_query(query_vector, fts_query, uri, table_name, top_k)
      Run a vector search + FTS search and return the results (for CLI smoke test).

  read_index_meta(uri, table_name) → dict
  write_index_meta(uri, table_name, meta)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pyarrow as pa

from crawler.config import (
    LANCE_META_FILENAME,
    LANCE_TABLE_NAME,
    VECTOR_INDEX_ENABLED,
    VECTOR_INDEX_METRIC,
    VECTOR_INDEX_MIN_ROWS,
    VECTOR_INDEX_TYPE,
)
from crawler.embed import EmbeddedChunk

logger = logging.getLogger(__name__)

# ── PyArrow schema ────────────────────────────────────────────────────────────

# The vector column is intentionally float32 (not int8): LanceDB's searchable
# vector column is float, and the int8-equivalent saving is applied by the SQ
# index (see build_vector_index), so both providers share one float32 schema.
_VECTOR_DIM = 512


def _build_schema(dims: int = _VECTOR_DIM) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("url", pa.string()),
            pa.field("title", pa.string()),
            pa.field("page_type", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("text", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("source", pa.string()),
            pa.field("is_faq", pa.bool_()),
            pa.field("vector", pa.list_(pa.float32(), dims)),
        ]
    )


def _rows_from_embedded(embedded: list[EmbeddedChunk]) -> list[dict]:
    rows = []
    for e in embedded:
        rows.append(
            {
                "chunk_id": e.chunk_id,
                "url": e.url,
                "title": e.title,
                "page_type": e.page_type,
                "content_hash": e.content_hash,
                "text": e.text,
                "chunk_index": e.chunk_index,
                "source": e.source,
                "is_faq": e.is_faq,
                "vector": e.vector,
            }
        )
    return rows


# ── Index metadata ────────────────────────────────────────────────────────────


# keep metadata co-located with the index (S3 object, not /tmp)
def _meta_key(table_name: str) -> str:
    return f"{table_name}{LANCE_META_FILENAME}"


def _split_s3_uri(uri: str) -> tuple[str, str]:
    """Split 's3://bucket/key/...' into (bucket, key)."""
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def _write_s3_object(s3_uri: str, body: str) -> None:
    bucket, key = _split_s3_uri(s3_uri)
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))


def _read_s3_object(s3_uri: str) -> str | None:
    bucket, key = _split_s3_uri(s3_uri)
    client = boto3.client("s3")
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
    except client.exceptions.NoSuchKey:
        return None
    return resp["Body"].read().decode("utf-8")


def write_index_meta(uri: str, table_name: str, meta: dict) -> None:
    payload = json.dumps(meta, indent=2)
    if uri.startswith("s3://"):
        _write_s3_object(f"{uri.rstrip('/')}/{_meta_key(table_name)}", payload)
    else:
        base = Path(uri)
        base.mkdir(parents=True, exist_ok=True)
        (base / _meta_key(table_name)).write_text(payload, encoding="utf-8")


def read_index_meta(uri: str, table_name: str) -> dict:
    if uri.startswith("s3://"):
        raw = _read_s3_object(f"{uri.rstrip('/')}/{_meta_key(table_name)}")
        return json.loads(raw) if raw else {}
    path = Path(uri) / _meta_key(table_name)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _make_meta(embedded: list[EmbeddedChunk], dims: int) -> dict:
    if not embedded:
        return {"chunk_count": 0, "built_at": _utcnow()}
    first = embedded[0]
    return {
        "provider": first.provider,
        "model": first.model,
        "dims": dims,
        "chunk_count": len(embedded),
        "built_at": _utcnow(),
    }


def _utcnow() -> str:
    return datetime.now(tz=UTC).isoformat()


# ── LanceDB connection helper ─────────────────────────────────────────────────


def _connect(uri: str, storage_options: dict | None = None):
    import lancedb

    if uri.startswith("s3://"):
        aws_opts: dict = {
            "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", ""),
            "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
            "aws_region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        }
        aws_opts.update(storage_options or {})
        return lancedb.connect(uri, storage_options=aws_opts)
    return lancedb.connect(uri)


def _table_exists(db, table_name: str) -> bool:
    """Return True if *table_name* exists in *db*, handling different LanceDB return types."""
    result = db.list_tables()
    # LanceDB 0.33+ returns ListTablesResponse with a .tables attribute
    names = result.tables if hasattr(result, "tables") else list(result)
    return table_name in names


# ── Build (full overwrite) ────────────────────────────────────────────────────


def build_index(
    embedded: list[EmbeddedChunk],
    uri: str,
    table_name: str = LANCE_TABLE_NAME,
    dims: int = _VECTOR_DIM,
    storage_options: dict | None = None,
) -> None:
    """Create (or overwrite) *table_name* with all embedded chunks.

    Full rebuild — used on the first crawl or when the embedding model changes.
    For incremental updates use upsert_chunks().
    """
    if not embedded:
        logger.warning("build_index called with 0 chunks — skipping.")
        return

    db = _connect(uri, storage_options)
    schema = _build_schema(dims)
    rows = _rows_from_embedded(embedded)

    if _table_exists(db, table_name):
        db.drop_table(table_name)
        logger.info("Dropped existing table %r for full rebuild.", table_name)

    tbl = db.create_table(table_name, data=rows, schema=schema)
    _create_fts_index(tbl)
    build_vector_index(tbl)

    meta = _make_meta(embedded, dims)
    write_index_meta(uri, table_name, meta)
    logger.info(
        "Built index %r with %d chunks (model=%s, dims=%d)",
        table_name,
        len(embedded),
        meta.get("model"),
        dims,
    )


def _create_fts_index(tbl) -> None:
    try:
        tbl.create_fts_index("text", replace=True)
        logger.debug("FTS index created on 'text' column.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("FTS index creation failed (non-fatal): %s", exc)


def build_vector_index(
    tbl,
    enabled: bool = VECTOR_INDEX_ENABLED,
    min_rows: int = VECTOR_INDEX_MIN_ROWS,
    index_type: str = VECTOR_INDEX_TYPE,
    metric: str = VECTOR_INDEX_METRIC,
) -> bool:
    """Build a scalar-quantized ANN index on the vector column when worthwhile.

    This is the H3 cost lever: LanceDB scalar quantization (IVF_HNSW_SQ) shrinks
    the in-memory vector index ~4x (the int8-equivalent saving) on the float
    vectors -- we never store provider int8 (the float32 column + the Jina
    fallback can't; see crawler.config.VOYAGE_EMBED_DTYPE).

    Skipped below *min_rows*: at launch scale (~3k chunks) exact brute-force
    search is faster, 100% accurate, and SQ training needs enough rows anyway.
    Building the ANN index is a non-critical optimization, so a failure here is
    logged WITH CONTEXT and the table keeps serving exact search -- it is not
    swallowed silently and it never fails the pipeline.

    Returns True iff an index was built. NOTE: the create_index call is unverified
    against a live LanceDB in this environment; run verify/verify_vector_index.py
    where lancedb is installed.
    """
    if not enabled:
        return False
    n = tbl.count_rows()
    if n < min_rows:
        logger.info(
            "Vector ANN index skipped: %d rows < %d (exact search used at this scale)",
            n,
            min_rows,
        )
        return False
    num_partitions = max(1, int(n**0.5))
    try:
        tbl.create_index(
            metric=metric,
            vector_column_name="vector",
            index_type=index_type,
            num_partitions=num_partitions,
            replace=True,
        )
        logger.info(
            "Built %s vector index (metric=%s, num_partitions=%d) on %d rows",
            index_type,
            metric,
            num_partitions,
            n,
        )
        return True
    except Exception as exc:  # noqa: BLE001 -- ANN index is optional; flat search still works
        logger.warning(
            "Vector index build failed (non-fatal; falling back to exact search): %s",
            exc,
        )
        return False


# ── Upsert (incremental) ──────────────────────────────────────────────────────


def upsert_chunks(
    embedded: list[EmbeddedChunk],
    uri: str,
    table_name: str = LANCE_TABLE_NAME,
    dims: int = _VECTOR_DIM,
    storage_options: dict | None = None,
) -> dict[str, int]:
    """Incremental upsert: add/replace changed chunks; leave unchanged ones alone.

    Returns a summary dict: {"added": N, "updated": N, "skipped": N}.
    Creates the table if it doesn't exist yet (first run).
    """
    if not embedded:
        return {"added": 0, "updated": 0, "skipped": 0}

    db = _connect(uri, storage_options)
    schema = _build_schema(dims)

    if not _table_exists(db, table_name):
        tbl = db.create_table(table_name, data=_rows_from_embedded(embedded), schema=schema)
        _create_fts_index(tbl)
        build_vector_index(tbl)
        meta = _make_meta(embedded, dims)
        write_index_meta(uri, table_name, meta)
        return {"added": len(embedded), "updated": 0, "skipped": 0}

    tbl = db.open_table(table_name)
    existing = _fetch_existing_index(tbl)
    existing_hashes = {cid: ch for cid, (ch, _url) in existing.items()}

    # Map every URL the table currently holds to the set of chunk_ids it owns, so we
    # can detect chunks that a re-ingested URL's new chunking no longer produces.
    existing_ids_by_url: dict[str, set[str]] = {}
    for cid, (_ch, url) in existing.items():
        existing_ids_by_url.setdefault(url, set()).add(cid)

    to_add: list[EmbeddedChunk] = []
    to_update: list[EmbeddedChunk] = []
    skipped = 0
    incoming_ids_by_url: dict[str, set[str]] = {}

    for e in embedded:
        incoming_ids_by_url.setdefault(e.url, set()).add(e.chunk_id)
        prev_hash = existing_hashes.get(e.chunk_id)
        if prev_hash is None:
            to_add.append(e)
        elif prev_hash != e.content_hash:
            to_update.append(e)
        else:
            skipped += 1

    # C2 fix: for every URL re-ingested this run, delete chunk_ids the table still holds
    # for that URL but the new chunking did NOT produce. chunk_id is derived from chunk
    # text + index, so an edit that rewrites or shifts a chunk yields a new chunk_id and
    # orphans the old row. Only URLs present in this batch are reconciled — URLs absent
    # from the crawl (e.g. 404s) are cleaned up exclusively by delete_chunks_for_urls().
    stale_ids: list[str] = []
    for url, incoming_ids in incoming_ids_by_url.items():
        stale_ids.extend(existing_ids_by_url.get(url, set()) - incoming_ids)

    # Changed-hash chunk_ids must also be removed before their replacements are added.
    ids_to_delete = list(dict.fromkeys([e.chunk_id for e in to_update] + stale_ids))
    if ids_to_delete:
        _delete_by_chunk_ids(tbl, ids_to_delete)

    new_rows = _rows_from_embedded(to_add + to_update)
    if new_rows:
        tbl.add(new_rows)

    # Refresh the FTS index whenever the row set changed (additions OR deletions).
    if new_rows or ids_to_delete:
        _create_fts_index(tbl)
        build_vector_index(tbl)

    meta = read_index_meta(uri, table_name)
    meta.update(_make_meta(to_add + to_update, dims))
    meta["chunk_count"] = tbl.count_rows()
    meta["last_upsert"] = _utcnow()
    meta["last_added"] = len(to_add)
    meta["last_updated"] = len(to_update)
    meta["last_orphans_deleted"] = len(stale_ids)
    write_index_meta(uri, table_name, meta)

    logger.info(
        "Upsert complete — added=%d updated=%d skipped=%d orphaned=%d (table=%r)",
        len(to_add),
        len(to_update),
        skipped,
        len(stale_ids),
        table_name,
    )
    return {"added": len(to_add), "updated": len(to_update), "skipped": skipped}


def _fetch_existing_index(tbl) -> dict[str, tuple[str, str]]:
    """Return {chunk_id: (content_hash, url)} for every row currently in the table."""
    try:
        rows = tbl.search().select(["chunk_id", "content_hash", "url"]).to_list()
        return {r["chunk_id"]: (r["content_hash"], r["url"]) for r in rows}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch existing index: %s", exc)
        return {}


def _delete_by_chunk_ids(tbl, chunk_ids: list[str]) -> None:
    if not chunk_ids:
        return
    ids_sql = ", ".join("'" + cid.replace("'", "''") + "'" for cid in chunk_ids)
    tbl.delete(f"chunk_id IN ({ids_sql})")


# ── Delete by URL (called on 404/410) ────────────────────────────────────────


def delete_chunks_for_urls(
    urls: list[str],
    uri: str,
    table_name: str = LANCE_TABLE_NAME,
    storage_options: dict | None = None,
) -> int:
    """Delete all chunks whose *url* is in *urls*.  Returns the number of deleted rows.

    This is a pure exact-match primitive: *urls* must already be in the same
    canonical form chunks were indexed under (``canonicalize_url`` of the page's
    final URL).  Callers that begin from a raw or discovered URL must canonicalize
    first -- see ``crawler.verify.deindex_permanent_failures`` -- otherwise the
    ``url IN (...)`` filter matches nothing and silently deletes 0 rows (C3).
    """
    if not urls:
        return 0

    db = _connect(uri, storage_options)
    if not _table_exists(db, table_name):
        return 0

    tbl = db.open_table(table_name)
    urls_sql = ", ".join("'" + u.replace("'", "''") + "'" for u in urls)
    before = tbl.count_rows()
    tbl.delete(f"url IN ({urls_sql})")
    deleted = before - tbl.count_rows()
    logger.info("Deleted %d chunks for %d URLs from table %r", deleted, len(urls), table_name)
    return deleted


# ── Smoke query (CLI validation) ──────────────────────────────────────────────


def smoke_query(
    query_vector: list[float],
    fts_query: str,
    uri: str,
    table_name: str = LANCE_TABLE_NAME,
    top_k: int = 5,
    storage_options: dict | None = None,
) -> dict[str, list[dict]]:
    """Run vector + FTS searches and return their results (for --smoke CLI flag)."""
    db = _connect(uri, storage_options)
    tbl = db.open_table(table_name)

    vector_results = tbl.search(query_vector, query_type="vector").limit(top_k).to_list()
    try:
        fts_results = tbl.search(fts_query, query_type="fts").limit(top_k).to_list()
    except Exception as exc:  # noqa: BLE001
        logger.warning("FTS search failed: %s", exc)
        fts_results = []

    def _clean(rows: list[dict]) -> list[dict]:
        return [
            {
                "chunk_id": r.get("chunk_id"),
                "url": r.get("url"),
                "title": r.get("title"),
                "page_type": r.get("page_type"),
                "text": r.get("text", "")[:120],
                "score": r.get("_distance") or r.get("_score"),
            }
            for r in rows
        ]

    return {"vector": _clean(vector_results), "fts": _clean(fts_results)}


# ── CLI entry-point ───────────────────────────────────────────────────────────


def _cli_smoke(uri: str, table_name: str = LANCE_TABLE_NAME) -> None:
    import sys

    meta = read_index_meta(uri, table_name)
    if not meta:
        print(f"No index metadata found at {uri!r} for table {table_name!r}.")
        sys.exit(1)

    print(f"Index metadata: {json.dumps(meta, indent=2)}")

    db = _connect(uri)
    if not _table_exists(db, table_name):
        print(f"Table {table_name!r} not found.")
        sys.exit(1)

    tbl = db.open_table(table_name)
    count = tbl.count_rows()
    print(f"Table row count: {count}")

    # Use a zero vector as a stand-in (real smoke uses a real embedding)
    dims = meta.get("dims", _VECTOR_DIM)
    zero_vec = [0.0] * dims
    results = smoke_query(
        query_vector=zero_vec,
        fts_query="Appther ERP implementation",
        uri=uri,
        table_name=table_name,
    )
    print(f"Sample vector results ({len(results['vector'])} rows):")
    for r in results["vector"]:
        print(f"  [{r['page_type']}] {r['url']} — {r['text'][:80]!r}")
    print(f"Sample FTS results ({len(results['fts'])} rows):")
    for r in results["fts"]:
        print(f"  [{r['page_type']}] {r['url']} — {r['text'][:80]!r}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LanceDB index utilities")
    parser.add_argument("--smoke", action="store_true", help="Run smoke query against the index")
    parser.add_argument("--uri", default="./lance_index", help="LanceDB URI (local path or s3://)")
    parser.add_argument("--table", default=LANCE_TABLE_NAME, help="Table name")
    args = parser.parse_args()

    if args.smoke:
        _cli_smoke(uri=args.uri, table_name=args.table)
