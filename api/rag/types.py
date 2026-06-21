"""Shared data types for the RAG query pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Turn:
    """One conversational exchange (user or assistant)."""

    role: Literal["user", "assistant"]
    content: str


@dataclass
class RetrievedChunk:
    """A chunk returned from hybrid search, enriched with a retrieval score.

    The `vector` field is carried through the pipeline so MMR can compute
    pairwise similarity, but it is not intended for callers of the public API.
    """

    chunk_id: str
    url: str
    title: str
    page_type: str
    text: str
    score: float
    is_faq: bool
    vector: list[float] = field(default_factory=list, repr=False, compare=False)


@dataclass
class RAGResult:
    """The complete output of one RAG query."""

    answer: str
    sources: list[str]
    language: str
    model: str
    rewritten_query: str
    chunks_used: int
    is_decline: bool = False
