"""Shared fixtures for api tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from api.rag.types import RetrievedChunk, Turn

DIMS = 512


# ── Chunk factory helpers ─────────────────────────────────────────────────────


def make_chunk(
    chunk_id: str = "c1",
    url: str = "https://www.appther.com/faq",
    title: str = "Appther FAQ",
    page_type: str = "faq",
    text: str = "What does Appther do? Appther builds ERP and CRM solutions.",
    score: float = 1.0,
    is_faq: bool = True,
    vector: list[float] | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        url=url,
        title=title,
        page_type=page_type,
        text=text,
        score=score,
        is_faq=is_faq,
        vector=vector or [0.1] * DIMS,
    )


def make_lance_row(
    chunk_id: str = "c1",
    url: str = "https://www.appther.com/faq",
    title: str = "Appther FAQ",
    page_type: str = "faq",
    text: str = "What does Appther do?",
    is_faq: bool = True,
    distance: float = 0.1,
    fts_score: float = 1.5,
    vector: list[float] | None = None,
) -> dict:
    """Simulate a LanceDB row dict as returned by .to_list()."""
    return {
        "chunk_id": chunk_id,
        "url": url,
        "title": title,
        "page_type": page_type,
        "text": text,
        "is_faq": is_faq,
        "_distance": distance,
        "_score": fts_score,
        "vector": vector or [0.1] * DIMS,
    }


# ── Mock LanceDB table ────────────────────────────────────────────────────────


def make_mock_table(
    vector_rows: list[dict] | None = None,
    fts_rows: list[dict] | None = None,
) -> MagicMock:
    """Build a mock LanceDB table with configurable search results."""
    table = MagicMock()

    def search_side_effect(query, query_type="vector"):
        chain = MagicMock()
        rows = vector_rows or [] if query_type == "vector" else fts_rows or []

        def limit_side_effect(n):
            inner = MagicMock()
            inner.to_list.return_value = rows[:n]
            return inner

        chain.limit.side_effect = limit_side_effect
        return chain

    table.search.side_effect = search_side_effect
    return table


# ── History helpers ───────────────────────────────────────────────────────────


def make_history(*pairs: tuple[str, str]) -> list[Turn]:
    """Create a Turn list from (user, assistant) text pairs."""
    turns: list[Turn] = []
    for user_text, assistant_text in pairs:
        turns.append(Turn(role="user", content=user_text))
        turns.append(Turn(role="assistant", content=assistant_text))
    return turns
