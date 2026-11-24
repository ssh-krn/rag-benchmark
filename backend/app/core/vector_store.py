"""
app/core/vector_store.py

Read-only access to a pre-populated ChromaDB persistent store, plus an
in-memory BM25Okapi index built from each collection's chunks at startup
(for hybrid_rag's sparse-retrieval leg). This module never writes to or
ingests into ChromaDB — the store is assumed to already exist on disk.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.core.embeddings import get_embeddings

logger = logging.getLogger("vector_store")

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class ChromaCollectionHandle:
    def __init__(self, name: str, langchain_chroma, raw_docs: List[str], raw_metadatas: List[dict], raw_ids: List[str]):
        self.name = name
        self.store = langchain_chroma
        self.raw_docs = raw_docs
        self.raw_metadatas = raw_metadatas
        self.raw_ids = raw_ids
        self._bm25 = None
        self._tokenized_corpus: Optional[List[List[str]]] = None

    @property
    def chunk_count(self) -> int:
        return len(self.raw_docs)

    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            self._tokenized_corpus = [_tokenize(d) for d in self.raw_docs]
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            logger.info("Built in-memory BM25 index for collection '%s' (%d chunks)", self.name, self.chunk_count)
        return self._bm25

    def bm25_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        bm25 = self.bm25()
        scores = bm25.get_scores(_tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for rank, idx in enumerate(ranked_idx):
            results.append({
                "chunk_id": self.raw_ids[idx],
                "text": self.raw_docs[idx],
                "metadata": self.raw_metadatas[idx],
                "score": float(scores[idx]),
                "bm25_rank": rank + 1,
            })
        return results

    def dense_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        docs_with_scores = self.store.similarity_search_with_relevance_scores(query, k=k)
        results = []
        for rank, (doc, score) in enumerate(docs_with_scores):
            results.append({
                "chunk_id": doc.metadata.get("chunk_id") or doc.metadata.get("id"),
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
                "dense_rank": rank + 1,
            })
        return results


class VectorStoreManager:
    """Loads and caches read-only handles to every persisted Chroma collection."""

    def __init__(self):
        self.settings = get_settings()
        self._handles: Dict[str, ChromaCollectionHandle] = {}

    def _load_collection(self, name: str) -> ChromaCollectionHandle:
        from langchain_chroma import Chroma
        import chromadb

        client = chromadb.PersistentClient(path=self.settings.CHROMA_DIR)
        store = Chroma(
            client=client,
            collection_name=name,
            embedding_function=get_embeddings(),
        )
        raw = store.get()  # pulls ids, documents, metadatas (read-only)
        handle = ChromaCollectionHandle(
            name=name,
            langchain_chroma=store,
            raw_docs=raw.get("documents") or [],
            raw_metadatas=raw.get("metadatas") or [],
            raw_ids=raw.get("ids") or [],
        )
        logger.info("Loaded ChromaDB collection '%s' with %d chunks", name, handle.chunk_count)
        return handle

    def get_collection(self, name: Optional[str] = None) -> ChromaCollectionHandle:
        name = name or self.settings.DEFAULT_COLLECTION_NAME
        if name not in self._handles:
            self._handles[name] = self._load_collection(name)
        return self._handles[name]

    def list_collections(self) -> List[Dict[str, Any]]:
        import chromadb

        client = chromadb.PersistentClient(path=self.settings.CHROMA_DIR)
        out = []
        for col in client.list_collections():
            try:
                handle = self.get_collection(col.name)
                out.append({"name": col.name, "chunk_count": handle.chunk_count})
            except Exception as exc:  # noqa: BLE001
                out.append({"name": col.name, "chunk_count": None, "error": str(exc)})
        return out

    def health(self) -> Dict[str, Any]:
        try:
            import chromadb

            client = chromadb.PersistentClient(path=self.settings.CHROMA_DIR)
            cols = client.list_collections()
            return {"status": "ok", "collections": [c.name for c in cols], "path": self.settings.CHROMA_DIR}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": str(exc), "path": self.settings.CHROMA_DIR}


@lru_cache
def get_vector_store_manager() -> VectorStoreManager:
    return VectorStoreManager()
