"""
app/core/embeddings.py
Singleton local embeddings model and FlashRank reranker. Both are CPU / iGPU
safe and loaded once per process.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger("embeddings")


@lru_cache
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    settings = get_settings()
    logger.info("Loading local embeddings model: %s (local_files_only=%s)",
                settings.EMBEDDING_MODEL_NAME, settings.LOCAL_FILES_ONLY)
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={"local_files_only": settings.LOCAL_FILES_ONLY, "device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache
def get_reranker():
    from flashrank import Ranker

    settings = get_settings()
    logger.info("Loading FlashRank reranker: %s", settings.RERANKER_MODEL_NAME)
    return Ranker(model_name=settings.RERANKER_MODEL_NAME, cache_dir=settings.RERANKER_CACHE_DIR)


def rerank(query: str, passages: list[dict], top_k: int) -> list[dict]:
    """
    passages: list of {"id": ..., "text": ...} (plus any extra keys, which are
    preserved and returned alongside the rerank score).
    """
    from flashrank import RerankRequest

    ranker = get_reranker()
    rr_request = RerankRequest(
        query=query,
        passages=[{"id": p.get("id") or i, "text": p["text"], "meta": p} for i, p in enumerate(passages)],
    )
    results = ranker.rerank(rr_request)
    output = []
    for r in results[:top_k]:
        meta = r.get("meta", {})
        output.append({**meta, "rerank_score": r.get("score")})
    return output
