"""H2: the llms.txt overview doc participates in dedupe + near-duplicate collapse.

Previously the overview was chunked and appended AFTER normalize_documents(), so
it never competed with real pages and _SOURCE_PRIORITY["overview"] was dead.
Now it is fed in via normalize_documents(..., extra_docs=[overview]).
"""

from __future__ import annotations

from crawler.chunk import build_overview_doc_from_text
from crawler.normalize import (
    NormalizedDoc,
    compute_content_hash,
    normalize_documents,
)


def _make_doc(url: str, text: str, source: str, priority: float = 1.0) -> NormalizedDoc:
    return NormalizedDoc(
        url=url,
        original_url=url,
        title="Test",
        markdown=text,
        page_type="other",
        content_hash=compute_content_hash(text),
        source=source,
        priority=priority,
    )


def test_build_overview_doc_from_text_tags_overview():
    doc = build_overview_doc_from_text("some overview body", "https://x/llms-full.txt")
    assert doc is not None
    assert doc.source == "overview"
    assert doc.page_type == "overview"
    assert doc.url == "https://x/llms-full.txt"


def test_build_overview_doc_from_empty_text_is_none():
    assert build_overview_doc_from_text("", "https://x/llms.txt") is None
    assert build_overview_doc_from_text("   \n  ", "https://x/llms.txt") is None


def test_overview_threaded_through_normalize():
    overview = build_overview_doc_from_text("overview body text", "https://x/llms-full.txt")
    out = normalize_documents([], extra_docs=[overview], collapse_dupes=False)
    assert [d.source for d in out] == ["overview"]
    # Without extra_docs nothing is produced from an empty triple list.
    assert normalize_documents([], collapse_dupes=False) == []


def test_overview_collapses_into_duplicate_page_and_loses():
    """When the overview is a near-duplicate of a real page, collapse keeps the
    higher-priority page (sitemap, rank 0) and drops the overview (rank 2)."""
    text = "Appther delivers Odoo ERP implementations for growing businesses. " * 20
    page = _make_doc("https://www.appther.com/services/odoo", text, source="sitemap")
    overview = build_overview_doc_from_text(text, "https://www.appther.com/llms-full.txt")
    out = normalize_documents(
        [], extra_docs=[page, overview], collapse_dupes=True, dup_threshold=0.5
    )
    assert len(out) == 1
    assert out[0].source == "sitemap"
    assert out[0].url == "https://www.appther.com/services/odoo"
