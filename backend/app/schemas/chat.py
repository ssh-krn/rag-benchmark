"""
app/schemas/chat.py
Request/response contracts for /api/chat and /api/chat/stream, plus the
rich retrieval metadata objects shared across all five RAG modes.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RagMode(str, Enum):
    TRADITIONAL_RAG = "traditional_rag"
    HYBRID_RAG = "hybrid_rag"
    CORRECTIVE_RAG = "corrective_rag"
    AGENTIC_RAG = "agentic_rag"
    LIGHTRAG = "lightrag"


class RetrievalMethod(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID_RRF = "hybrid_rrf"
    RERANKED = "reranked"
    WEB_SEARCH = "web_search"
    GRAPH_TRAVERSAL = "graph_traversal"


class SourceMetadata(BaseModel):
    """Per-chunk / per-source metadata returned for every retrieved context item."""

    chunk_id: Optional[str] = None
    source: Optional[str] = None
    page: Optional[int] = None
    collection_name: Optional[str] = None
    score: Optional[float] = None
    retrieval_method: Optional[RetrievalMethod] = None
    text: Optional[str] = None

    # Hybrid-specific
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None

    # Web-specific
    url: Optional[str] = None

    # Graph-specific (LightRAG)
    entities: Optional[List[str]] = None
    relations: Optional[List[Dict[str, Any]]] = None


class ExecutionStep(BaseModel):
    node: str
    description: str
    detail: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: RagMode = RagMode.TRADITIONAL_RAG
    collection_name: Optional[str] = None
    graph_name: Optional[str] = None
    conversation_id: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    answer: str
    mode: RagMode
    retrieved_context: List[SourceMetadata]
    execution_trace: List[ExecutionStep]

    # LightRAG-only extras (also mirrored into retrieved_context entries)
    matched_entities: Optional[List[str]] = None
    relationships: Optional[List[Dict[str, Any]]] = None
    subgraph_nodes: Optional[List[Dict[str, Any]]] = None
    subgraph_edges: Optional[List[Dict[str, Any]]] = None

    model_used: Optional[str] = None
    provider_used: Optional[str] = None


class SSEEvent(BaseModel):
    """Envelope used for every Server-Sent Event frame on /api/chat/stream."""

    event: str  # one of: "trace" | "context" | "token" | "final" | "error"
    data: Dict[str, Any]
