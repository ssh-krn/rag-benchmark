"""
app/tools/chroma_search.py

LangChain tool exposing read-only ChromaDB + FlashRank reranked search over
pre-populated chunks. Used directly by traditional/hybrid/corrective RAG
graphs and bound as a callable tool for the agentic_rag ReAct agent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from app.config import get_settings
from app.core.embeddings import rerank
from app.core.vector_store import get_vector_store_manager
from app.schemas.chat import RetrievalMethod, SourceMetadata


def search_chunks_with_metadata(
    query: str,
    collection_name: Optional[str] = None,
    k: int = 5,
    rerank_top_k: Optional[int] = None,
) -> List[SourceMetadata]:
    """Dense ChromaDB search, optionally reranked with FlashRank. Returns rich metadata."""
    manager = get_vector_store_manager()
    handle = manager.get_collection(collection_name)
    dense_hits = handle.dense_search(query, k=k)

    if rerank_top_k:
        passages = [{"id": h["chunk_id"], "text": h["text"], **h} for h in dense_hits]
        reranked = rerank(query, passages, top_k=rerank_top_k)
        return [
            SourceMetadata(
                chunk_id=r.get("chunk_id"),
                source=(r.get("metadata") or {}).get("source"),
                page=(r.get("metadata") or {}).get("page"),
                collection_name=handle.name,
                score=r.get("score"),
                retrieval_method=RetrievalMethod.RERANKED,
                text=r.get("text"),
                dense_rank=r.get("dense_rank"),
                rerank_score=r.get("rerank_score"),
            )
            for r in reranked
        ]

    return [
        SourceMetadata(
            chunk_id=h["chunk_id"],
            source=(h.get("metadata") or {}).get("source"),
            page=(h.get("metadata") or {}).get("page"),
            collection_name=handle.name,
            score=h.get("score"),
            retrieval_method=RetrievalMethod.DENSE,
            text=h.get("text"),
            dense_rank=h.get("dense_rank"),
        )
        for h in dense_hits
    ]


@tool("search_textbook_chunks")
def search_textbook_chunks(query: str) -> str:
    """Search the pre-populated textbook ChromaDB collection for relevant passages.
    Use this to find grounded factual content before answering. Input should be a
    focused natural-language query or keyword phrase."""
    settings = get_settings()
    results = search_chunks_with_metadata(
        query, collection_name=settings.DEFAULT_COLLECTION_NAME, k=8, rerank_top_k=4
    )
    truncate = settings.AGENTIC_CHUNK_TRUNCATE_CHARS
    lines = []
    for i, r in enumerate(results, start=1):
        text = (r.text or "")[:truncate]
        lines.append(f"[Passage {i} | chunk_id={r.chunk_id} | score={r.score}]\n{text}")
    return "\n\n".join(lines) if lines else "No relevant passages found."
