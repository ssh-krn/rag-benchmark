"""
app/graphs/lightrag.py

Mode 5 — LightRAG Knowledge Graph.
Flow: Dual-level keyword/entity extraction via resilient LLM/Ollama
(deepseek-v4.1-flash:cloud with automatic fallback to Groq/Gemini on 429 or
quota exhaustion) -> 1-hop/2-hop NetworkX graph traversal + ChromaDB
entity/relation lookup -> synthesis returning matched_entities, relationships,
subgraph_nodes, subgraph_edges.

Data source: pre-built NetworkX Knowledge Graph + ChromaDB entity/relationship
stores. This is the ONLY mode that queries the knowledge graph.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, TypedDict

from app.config import get_settings
from app.core.graph_store import get_graph_store_manager
from app.core.resilient_llm import ResilientChatRouter, build_candidate_chain
from app.schemas.chat import ChatResponse, ExecutionStep, RagMode, RetrievalMethod, SourceMetadata

KEYWORD_EXTRACTION_PROMPT = """Extract the key entities and concepts (dual-level: both
high-level themes and low-level specific terms) from the question below that would be
useful for searching a knowledge graph. Return ONLY a JSON array of short keyword strings,
no commentary.

Question: {question}"""

SYNTHESIS_SYSTEM_PROMPT = """You are a knowledge-graph-grounded RAG assistant. Answer using
the matched entities and relationships provided. Reference entity names directly in your answer
where relevant."""


class LightRAGState(TypedDict):
    query: str
    graph_name: Optional[str]
    keywords: List[str]
    matched_entities: List[str]
    subgraph_nodes: List[Dict[str, Any]]
    subgraph_edges: List[Dict[str, Any]]
    answer: Optional[str]
    trace: List[ExecutionStep]
    last_provider: Optional[str]
    last_model: Optional[str]


def _ollama_first_chain() -> ResilientChatRouter:
    """Build a router prioritizing Ollama :cloud (as per spec: LightRAG keyword
    extraction prefers deepseek-v4.1-flash:cloud) with Groq/Gemini as fallback."""
    full_chain = build_candidate_chain(role="reasoning")
    ollama_first = [c for c in full_chain if c.provider == "ollama_cloud"]
    others = [c for c in full_chain if c.provider != "ollama_cloud"]
    return ResilientChatRouter(role="reasoning", candidates=ollama_first + others)


def _extract_json_array(text: str) -> List[str]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return [w.strip() for w in re.split(r"[,\n]", text) if w.strip()][:10]
    try:
        return [str(x) for x in json.loads(match.group(0))]
    except json.JSONDecodeError:
        return [w.strip() for w in re.split(r"[,\n]", text) if w.strip()][:10]


async def node_extract_keywords(state: LightRAGState) -> LightRAGState:
    router = _ollama_first_chain()
    prompt = KEYWORD_EXTRACTION_PROMPT.format(question=state["query"])
    response = await router.ainvoke(prompt)
    text = getattr(response, "content", str(response))
    keywords = _extract_json_array(text)
    state["keywords"] = keywords
    state["last_provider"], state["last_model"] = router.last_provider, router.last_model
    state["trace"].append(ExecutionStep(
        node="dual_level_keyword_extraction",
        description=f"Extracted {len(keywords)} dual-level keywords/entities.",
        detail={"provider": router.last_provider, "model": router.last_model, "keywords": keywords},
    ))
    return state


async def node_graph_traversal(state: LightRAGState) -> LightRAGState:
    settings = get_settings()
    manager = get_graph_store_manager()
    handle = manager.get_graph(state["graph_name"])

    seeds = handle.find_matching_entities(state["keywords"], limit=15)
    nodes, edges = handle.traverse(seeds, hops=settings.LIGHTRAG_HOPS)

    state["matched_entities"] = seeds
    state["subgraph_nodes"] = [handle.node_payload(n) for n in nodes]
    state["subgraph_edges"] = edges
    state["trace"].append(ExecutionStep(
        node="graph_traversal",
        description=f"Matched {len(seeds)} seed entities; {settings.LIGHTRAG_HOPS}-hop traversal expanded to "
                     f"{len(nodes)} nodes / {len(edges)} edges.",
    ))
    return state


async def node_synthesize(state: LightRAGState) -> LightRAGState:
    router = ResilientChatRouter(role="reasoning")
    entities_block = "\n".join(f"- {n['id']}: {n.get('description', '')}" for n in state["subgraph_nodes"][:30])
    relations_block = "\n".join(
        f"- {e['source']} --[{e.get('relation', 'related_to')}]--> {e['target']}" for e in state["subgraph_edges"][:40]
    )
    prompt = (
        f"{SYNTHESIS_SYSTEM_PROMPT}\n\nMatched entities:\n{entities_block}\n\n"
        f"Relationships:\n{relations_block}\n\nQuestion: {state['query']}"
    )
    response = await router.ainvoke(prompt)
    state["answer"] = getattr(response, "content", str(response))
    state["last_provider"], state["last_model"] = router.last_provider, router.last_model
    state["trace"].append(ExecutionStep(
        node="graph_synthesis",
        description="Synthesized answer from matched entities and relationships.",
        detail={"provider": router.last_provider, "model": router.last_model},
    ))
    return state


async def run_lightrag(query: str, graph_name: Optional[str] = None) -> ChatResponse:
    state: LightRAGState = {
        "query": query,
        "graph_name": graph_name,
        "keywords": [],
        "matched_entities": [],
        "subgraph_nodes": [],
        "subgraph_edges": [],
        "answer": None,
        "trace": [],
        "last_provider": None,
        "last_model": None,
    }
    state = await node_extract_keywords(state)
    state = await node_graph_traversal(state)
    state = await node_synthesize(state)

    retrieved_context = [
        SourceMetadata(
            chunk_id=None,
            source=n["id"],
            collection_name=graph_name,
            retrieval_method=RetrievalMethod.GRAPH_TRAVERSAL,
            text=n.get("description"),
            entities=[n["id"]],
        )
        for n in state["subgraph_nodes"][:30]
    ]

    return ChatResponse(
        answer=state["answer"] or "Unable to synthesize an answer from the knowledge graph.",
        mode=RagMode.LIGHTRAG,
        retrieved_context=retrieved_context,
        execution_trace=state["trace"],
        matched_entities=state["matched_entities"],
        relationships=state["subgraph_edges"],
        subgraph_nodes=state["subgraph_nodes"],
        subgraph_edges=state["subgraph_edges"],
        model_used=state["last_model"],
        provider_used=state["last_provider"],
    )
