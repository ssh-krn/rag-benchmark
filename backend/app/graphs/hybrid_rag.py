"""
app/graphs/hybrid_rag.py

Mode 2 — Hybrid Search RAG.
Flow: ChromaDB dense search (k=10) + BM25Okapi sparse search (k=10) ->
Reciprocal Rank Fusion (RRF) -> FlashRank reranking (top k=4) -> grounded
synthesis + metadata (dense_rank, bm25_rank, rrf_score, rerank_score).

Data source: ChromaDB chunks only.
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.config import get_settings
from app.core.embeddings import rerank
from app.core.resilient_llm import get_chat_model
from app.core.vector_store import get_vector_store_manager
from app.schemas.chat import ChatResponse, ExecutionStep, RagMode, RetrievalMethod, SourceMetadata

SYSTEM_PROMPT = """You are a grounded RAG assistant using hybrid (dense + lexical) retrieval.
Answer ONLY using the provided passages. Cite every factual claim with its passage number
in the form [Passage X]. If the passages do not contain the answer, say so explicitly."""


def _reciprocal_rank_fusion(dense_hits: List[dict], bm25_hits: List[dict], k_constant: int = 60) -> Dict[str, dict]:
    """Merge by chunk_id, combining rank-based RRF scores from both lists."""
    fused: Dict[str, dict] = {}

    for hit in dense_hits:
        cid = hit["chunk_id"]
        fused.setdefault(cid, {**hit, "rrf_score": 0.0})
        fused[cid]["rrf_score"] += 1.0 / (k_constant + hit["dense_rank"])
        fused[cid]["dense_rank"] = hit["dense_rank"]

    for hit in bm25_hits:
        cid = hit["chunk_id"]
        if cid not in fused:
            fused[cid] = {**hit, "rrf_score": 0.0}
        fused[cid]["rrf_score"] += 1.0 / (k_constant + hit["bm25_rank"])
        fused[cid]["bm25_rank"] = hit["bm25_rank"]
        fused[cid].setdefault("text", hit.get("text"))
        fused[cid].setdefault("metadata", hit.get("metadata"))

    return fused


def _build_context_block(passages: List[SourceMetadata]) -> str:
    lines = []
    for i, p in enumerate(passages, start=1):
        lines.append(f"[Passage {i}] (chunk_id={p.chunk_id}, rrf={p.rrf_score}, rerank={p.rerank_score})\n{p.text}")
    return "\n\n".join(lines)


async def run_hybrid_rag(query: str, collection_name: str | None = None) -> ChatResponse:
    settings = get_settings()
    trace: List[ExecutionStep] = []
    manager = get_vector_store_manager()
    handle = manager.get_collection(collection_name)

    dense_hits = handle.dense_search(query, k=settings.HYBRID_DENSE_K)
    bm25_hits = handle.bm25_search(query, k=settings.HYBRID_BM25_K)
    trace.append(ExecutionStep(
        node="dual_retrieval",
        description="Ran dense ChromaDB search and BM25Okapi sparse search in parallel.",
        detail={"dense_hits": len(dense_hits), "bm25_hits": len(bm25_hits)},
    ))

    fused = _reciprocal_rank_fusion(dense_hits, bm25_hits)
    fused_sorted = sorted(fused.values(), key=lambda h: h["rrf_score"], reverse=True)
    trace.append(ExecutionStep(
        node="reciprocal_rank_fusion",
        description="Fused dense + BM25 rankings via RRF.",
        detail={"fused_candidates": len(fused_sorted)},
    ))

    passages_for_rerank = [{"id": h["chunk_id"], "text": h.get("text", ""), **h} for h in fused_sorted]
    reranked = rerank(query, passages_for_rerank, top_k=settings.HYBRID_RERANK_K)
    trace.append(ExecutionStep(
        node="flashrank_rerank",
        description=f"Reranked fused candidates with FlashRank, keeping top {settings.HYBRID_RERANK_K}.",
        detail={"top_k": settings.HYBRID_RERANK_K},
    ))

    passages = [
        SourceMetadata(
            chunk_id=r.get("chunk_id"),
            source=(r.get("metadata") or {}).get("source"),
            page=(r.get("metadata") or {}).get("page"),
            collection_name=handle.name,
            score=r.get("rrf_score"),
            retrieval_method=RetrievalMethod.RERANKED,
            text=r.get("text"),
            dense_rank=r.get("dense_rank"),
            bm25_rank=r.get("bm25_rank"),
            rrf_score=r.get("rrf_score"),
            rerank_score=r.get("rerank_score"),
        )
        for r in reranked
    ]

    context_block = _build_context_block(passages)
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
        mode=RagMode.HYBRID_RAG,
        retrieved_context=passages,
        execution_trace=trace,
        model_used=router.last_model,
        provider_used=router.last_provider,
    )
