"""Blueprint chunk-size invariant: no chunk exceeds CHUNK_MAX_TOKENS.

Guards the MEDIUM fix to _split_section()'s tail handling (the old tail-merge cap
of max_c * 2 let a merged chunk reach ~2x the 400-600 token spec). These cover
both the single-big-paragraph path and the small-accumulation-then-big-paragraph
edge.
"""

from __future__ import annotations

from crawler.chunk import _split_section, chunk_document
from crawler.config import CHARS_PER_TOKEN, CHUNK_MAX_TOKENS
from crawler.normalize import NormalizedDoc, compute_content_hash

MAX_CHARS = CHUNK_MAX_TOKENS * CHARS_PER_TOKEN
_SENTENCE = "Appther delivers Odoo ERP implementations for growing businesses. "


def _make_doc(markdown: str) -> NormalizedDoc:
    return NormalizedDoc(
        url="https://www.appther.com/x",
        original_url="https://www.appther.com/x",
        title="X",
        markdown=markdown,
        page_type="other",
        content_hash=compute_content_hash(markdown),
        source="sitemap",
    )


def test_single_big_paragraph_within_max():
    for chunk in _split_section(_SENTENCE * 200):
        assert len(chunk) <= MAX_CHARS


def test_small_accumulation_then_big_paragraph_within_max():
    """A sub-min_c run followed by a large paragraph must not yield a >max_c tail."""
    small = "short intro paragraph. " * 8
    big = _SENTENCE * 40
    chunks = _split_section(small + "\n\n" + big)
    assert chunks
    for chunk in chunks:
        assert len(chunk) <= MAX_CHARS, f"chunk too long: {len(chunk)}"


def test_tiny_tail_merge_respects_max():
    chunks = _split_section(_SENTENCE * 30 + "\n\nx.")
    for chunk in chunks:
        assert len(chunk) <= MAX_CHARS


def test_chunk_document_large_doc_within_max():
    md = "# Overview\n\n" + _SENTENCE * 200
    for c in chunk_document(_make_doc(md)):
        assert len(c.text) <= MAX_CHARS


def test_no_content_lost_on_split():
    """Content preservation: every word of the source appears in the joined chunks.

    Uses overlap-aware reconstruction with unique text (not repetitive sentences)
    to avoid false-positive suffix-matching in the overlap removal.
    """
    words = [f"zeta{i} about Appther for growing businesses." for i in range(400)]
    unique_text = " ".join(words)
    parts = _split_section(unique_text)

    def _collapse(s: str) -> str:
        return "".join(s.split())

    rebuilt = ""
    for c in parts:
        cc = _collapse(c)
        if not rebuilt:
            rebuilt = cc
            continue
        maxk = min(len(rebuilt), len(cc))
        k = next((kk for kk in range(maxk, 0, -1) if rebuilt[-kk:] == cc[:kk]), 0)
        rebuilt += cc[k:]
    assert rebuilt == _collapse(unique_text)


def test_no_content_lost_on_split_short():
    """This test is to cover the case of lower overlap  -"""
    text = _SENTENCE * 10
    parts = _split_section(text)
    # _split_section should have exactly one chunk for this short text
    assert len(parts) == 1, f"Expected 1 chunk got {len(parts)}"
    assert parts[0] in text or text in parts[0]
