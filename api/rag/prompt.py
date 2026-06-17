"""Prompt assembly for the RAG pipeline.

Responsibilities:
- SYSTEM_PROMPT: the fixed instruction block for the LLM.
- format_context(): turn retrieved chunks into a numbered source block.
- format_history(): convert Turn list to Gemini content dicts (role: user/model).
- build_user_message(): compose the final user turn: context + question.
"""

from __future__ import annotations

from api.rag.types import RetrievedChunk, Turn

SYSTEM_PROMPT = """You are Appther's customer support assistant. Appther is a technology company \
specializing in ERP, CRM, and software implementation services across USA, Australia, Dubai, \
Delhi NCR, and Canada.

Answer ONLY from the context provided below. Follow these rules strictly:
1. If the answer is present in the context, provide a clear, helpful, and concise response.
2. If the answer is NOT in the context, respond with exactly: \
"I don't have information about that in my current knowledge. For detailed help, please \
visit https://www.appther.com/contact-us or book a free consultation."
3. Always cite the source URLs as a numbered list at the end of your answer \
(e.g. "Sources: [1] https://...").
4. Do not invent facts, prices, or timelines not present in the context.
5. If the question is in a language other than English, respond in that same language.
6. Keep your tone helpful, professional, and sales-aware — Appther is a trusted partner."""

MAX_HISTORY_TURNS = 3
MAX_CONTEXT_CHARS = 8000


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered reference block."""
    if not chunks:
        return "(No relevant context found.)"
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[{i}] {chunk.title} ({chunk.page_type}) — {chunk.url}"
        parts.append(f"{header}\n{chunk.text}")
    context = "\n\n---\n\n".join(parts)
    # Hard-cap to avoid runaway token cost; truncate rather than fail.
    return context[:MAX_CONTEXT_CHARS]


def format_history(history: list[Turn]) -> list[dict]:
    """Convert the last MAX_HISTORY_TURNS turns into Gemini content format.

    Gemini uses "user" / "model" roles (not "assistant").
    History must alternate; we pass what we have and let the SDK handle it.
    """
    recent = history[-MAX_HISTORY_TURNS:]
    result = []
    for turn in recent:
        role = "model" if turn.role == "assistant" else "user"
        result.append({"role": role, "parts": [{"text": turn.content}]})
    return result


def build_user_message(question: str, context_str: str) -> str:
    """Compose the final user message that embeds the context."""
    return f"Context:\n{context_str}\n\nQuestion: {question}"
