"""
app/graphs/traditional_rag.py

Mode 1 — Traditional Vector RAG.
Flow: Query -> pure dense similarity search on ChromaDB (k=5) -> grounded
synthesis with [Passage X] citations -> answer + full chunk metadata.

Data source: ChromaDB chunks only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.core.resilient_llm import get_chat_model
from app.graphs._prompt_budget import budget_passages
from app.schemas.chat import ChatResponse, ExecutionStep, RagMode, SourceMetadata
from app.tools.chroma_search import search_chunks_with_metadata

SYSTEM_PROMPT = """You are a grounded RAG assistant. Answer ONLY using the provided passages.
Cite every factual claim with its passage number in the form [Passage X]. If the passages do
not contain the answer, say so explicitly rather than guessing."""


def _build_context_block(passages: List[SourceMetadata]) -> str:
    lines = []
    for i, p in enumerate(passages, start=1):
        lines.append(f"[Passage {i}] (chunk_id={p.chunk_id}, score={p.score})\n{p.text}")
    return "\n\n".join(lines)


async def run_traditional_rag(
    query: str,
    collection_name: str | None = None,
    prompt_top_k: Optional[int] = None,
    prompt_char_limit: Optional[int] = None,
) -> ChatResponse:
    """`prompt_top_k`/`prompt_char_limit` are internal-only knobs for the eval
    runner's token-budget compression (see app/graphs/_prompt_budget.py).
    Left as None (the default), live /api/chat behavior is unchanged."""
    settings = get_settings()
    trace: List[ExecutionStep] = []

    passages = search_chunks_with_metadata(query, collection_name=collection_name, k=settings.TRADITIONAL_RAG_K)
    trace.append(ExecutionStep(
        node="dense_retrieval",
        description=f"Retrieved top-{settings.TRADITIONAL_RAG_K} chunks via ChromaDB dense similarity search.",
        detail={"k": settings.TRADITIONAL_RAG_K, "hits": len(passages)},
    ))

    context_block = _build_context_block(budget_passages(passages, prompt_top_k, prompt_char_limit))
    router = get_chat_model(role="reasoning")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Passages:\n\n{context_block}\n\nQuestion: {query}"},
    ]
    response = await router.ainvoke(messages)
    trace.append(ExecutionStep(
        node="synthesis",
        description="Generated grounded answer with citations.",
        detail={"provider": router.last_provider, "model": router.last_model},
    ))

    return ChatResponse(
        answer=getattr(response, "content", str(response)),
        mode=RagMode.TRADITIONAL_RAG,
        retrieved_context=passages,
        execution_trace=trace,
        model_used=router.last_model,
        provider_used=router.last_provider,
    )