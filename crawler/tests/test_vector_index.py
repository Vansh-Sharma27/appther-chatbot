"""H3: build_vector_index() size-gated scalar-quantization index.

Uses a fake table so no live LanceDB is required at test-collection time -- the
real create_index() call against LanceDB should additionally be smoke-tested via
verify/verify_fixes.py (lancedb section) on a machine with lancedb installed.
"""

from __future__ import annotations

import pytest

from crawler.index import build_vector_index


class _FakeTable:
    def __init__(self, n: int):
        self._n = n
        self.created: dict | None = None

    def count_rows(self) -> int:
        return self._n

    def create_index(self, **kwargs):
        self.created = kwargs


def test_skips_below_min_rows():
    tbl = _FakeTable(100)
    assert build_vector_index(tbl, min_rows=2048) is False
    assert tbl.created is None


def test_builds_sq_index_above_min_rows():
    tbl = _FakeTable(5000)
    assert build_vector_index(tbl, min_rows=2048) is True
    assert tbl.created["index_type"] == "IVF_HNSW_SQ"
    assert tbl.created["vector_column_name"] == "vector"
    assert tbl.created["metric"] == "l2"
    assert tbl.created["num_partitions"] == int(5000**0.5)


def test_disabled_never_builds():
    tbl = _FakeTable(5000)
    assert build_vector_index(tbl, enabled=False) is False
    assert tbl.created is None


def test_build_failure_is_non_fatal():
    class _Boom(_FakeTable):
        def create_index(self, **kwargs):
            raise RuntimeError("simulated lancedb failure")

    # Must swallow-with-logging (return False), never propagate.
    assert build_vector_index(_Boom(5000)) is False


@pytest.mark.parametrize("n,expected", [(2048, 45), (10000, 100)])
def test_num_partitions_is_sqrt(n, expected):
    tbl = _FakeTable(n)
    build_vector_index(tbl)
    assert tbl.created["num_partitions"] == expected
