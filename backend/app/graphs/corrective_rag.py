"""
app/graphs/corrective_rag.py

Mode 3 — Self-Reflective / Corrective RAG (LangGraph StateGraph).
Flow: retrieve & FlashRank -> relevance grader (fast grader model) ->
query-rewriter loop (max_retries=2) -> generator -> hallucination /
groundedness grader. On a rejected draft, the previous AIMessage draft is
popped from message history before a corrective HumanMessage retry is
appended, so the model never "sees" its own rejected answer as context.

Data source: ChromaDB chunks only (no web fallback).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.config import get_settings
from app.core.resilient_llm import get_chat_model
from app.graphs._prompt_budget import budget_passages
from app.schemas.chat import ChatResponse, ExecutionStep, RagMode, SourceMetadata
from app.tools.chroma_search import search_chunks_with_metadata

GENERATION_SYSTEM_PROMPT = """You are a grounded RAG assistant. Answer ONLY using the provided
passages. Cite every factual claim with its passage number in the form [Passage X]."""


class CorrectiveRagState(TypedDict):
    original_query: str
    current_query: str
    collection_name: Optional[str]
    prompt_top_k: Optional[int]
    prompt_char_limit: Optional[int]
    passages: List[SourceMetadata]
    messages: List[BaseMessage]
    retries: int
    max_retries: int
    relevance_ok: bool
    groundedness_ok: bool
    final_answer: Optional[str]
    trace: List[ExecutionStep]
    last_provider: Optional[str]
    last_model: Optional[str]


def _context_block(state: CorrectiveRagState) -> str:
    budgeted = budget_passages(state["passages"], state.get("prompt_top_k"), state.get("prompt_char_limit"))
    return "\n\n".join(
        f"[Passage {i}] (chunk_id={p.chunk_id})\n{p.text}" for i, p in enumerate(budgeted, start=1)
    )


async def node_retrieve(state: CorrectiveRagState) -> CorrectiveRagState:
    passages = search_chunks_with_metadata(
        state["current_query"], collection_name=state["collection_name"], k=10, rerank_top_k=5
    )
    state["passages"] = passages
    state["trace"].append(ExecutionStep(
        node="retrieve_and_rerank",
        description=f"Retrieved+reranked {len(passages)} passages for query: '{state['current_query']}'",
    ))
    return state


async def node_grade_relevance(state: CorrectiveRagState) -> CorrectiveRagState:
    grader = get_chat_model(role="grader")
    context = _context_block(state)
    prompt = (
        "You are a strict relevance grader. Given the passages and the question, respond with "
        "exactly one word: 'yes' if the passages contain enough information to answer the question, "
        f"or 'no' otherwise.\n\nPassages:\n{context}\n\nQuestion: {state['original_query']}"
    )
    response = await grader.ainvoke(prompt)
    verdict = getattr(response, "content", str(response)).strip().lower()
    state["relevance_ok"] = verdict.startswith("y")
    state["trace"].append(ExecutionStep(
        node="relevance_grader",
        description=f"Relevance grade: {'RELEVANT' if state['relevance_ok'] else 'NOT RELEVANT'}",
        detail={"provider": grader.last_provider, "model": grader.last_model},
    ))
    return state


async def node_rewrite_query(state: CorrectiveRagState) -> CorrectiveRagState:
    router = get_chat_model(role="reasoning")
    prompt = (
        f"The following search query returned insufficient results: '{state['current_query']}'. "
        f"Rewrite it to better capture the user's intent for the original question: "
        f"'{state['original_query']}'. Return ONLY the rewritten query."
    )
    response = await router.ainvoke(prompt)
    new_query = getattr(response, "content", str(response)).strip()
    state["current_query"] = new_query
    state["retries"] += 1
    state["trace"].append(ExecutionStep(
        node="query_rewriter",
        description=f"Rewrote query (attempt {state['retries']}/{state['max_retries']}): '{new_query}'",
    ))
    return state


async def node_generate(state: CorrectiveRagState) -> CorrectiveRagState:
    router = get_chat_model(role="reasoning")
    context = _context_block(state)

    if not state["messages"]:
        state["messages"].append(SystemMessage(content=GENERATION_SYSTEM_PROMPT))
        state["messages"].append(HumanMessage(content=f"Passages:\n\n{context}\n\nQuestion: {state['original_query']}"))

    response = await router.ainvoke(state["messages"])
    draft = getattr(response, "content", str(response))
    state["messages"].append(AIMessage(content=draft))
    state["final_answer"] = draft
    state["last_provider"], state["last_model"] = router.last_provider, router.last_model
    state["trace"].append(ExecutionStep(
        node="generate",
        description="Generated draft answer.",
        detail={"provider": router.last_provider, "model": router.last_model},
    ))
    return state


async def node_grade_groundedness(state: CorrectiveRagState) -> CorrectiveRagState:
    grader = get_chat_model(role="grader")
    context = _context_block(state)
    prompt = (
        "You are a strict hallucination grader. Given the source passages and a draft answer, "
        "respond with exactly one word: 'grounded' if every claim in the draft is supported by the "
        f"passages, or 'hallucinated' otherwise.\n\nPassages:\n{context}\n\nDraft answer:\n{state['final_answer']}"
    )
    response = await grader.ainvoke(prompt)
    verdict = getattr(response, "content", str(response)).strip().lower()
    state["groundedness_ok"] = verdict.startswith("grounded")
    state["trace"].append(ExecutionStep(
        node="groundedness_grader",
        description=f"Groundedness grade: {'GROUNDED' if state['groundedness_ok'] else 'HALLUCINATED'}",
        detail={"provider": grader.last_provider, "model": grader.last_model},
    ))
    return state


async def node_pop_rejected_draft(state: CorrectiveRagState) -> CorrectiveRagState:
    """Remove the rejected AIMessage draft, then append a corrective HumanMessage
    instructing the model to retry grounded strictly in the (possibly refreshed) passages."""
    if state["messages"] and isinstance(state["messages"][-1], AIMessage):
        state["messages"].pop()
    context = _context_block(state)
    state["messages"].append(HumanMessage(
        content=(
            "Your previous draft was not fully grounded in the source passages and was discarded. "
            f"Re-answer using ONLY these passages, citing [Passage X] for every claim:\n\n{context}\n\n"
            f"Question: {state['original_query']}"
        )
    ))
    state["retries"] += 1
    state["trace"].append(ExecutionStep(
        node="pop_rejected_draft",
        description="Popped rejected AIMessage draft and appended corrective retry prompt.",
    ))
    return state


def _route_after_relevance(state: CorrectiveRagState) -> str:
    if state["relevance_ok"]:
        return "generate"
    if state["retries"] < state["max_retries"]:
        return "rewrite_query"
    return "generate"  # exhausted retries: generate best-effort with what we have


def _route_after_groundedness(state: CorrectiveRagState) -> str:
    if state["groundedness_ok"]:
        return "end"
    if state["retries"] < state["max_retries"]:
        return "pop_and_retry"
    return "end"  # exhausted retries: return the (possibly imperfect) draft


def build_corrective_rag_graph():
    graph = StateGraph(CorrectiveRagState)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("grade_relevance", node_grade_relevance)
    graph.add_node("rewrite_query", node_rewrite_query)
    graph.add_node("generate", node_generate)
    graph.add_node("grade_groundedness", node_grade_groundedness)
    graph.add_node("pop_and_retry", node_pop_rejected_draft)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_relevance")
    graph.add_conditional_edges("grade_relevance", _route_after_relevance, {
        "generate": "generate", "rewrite_query": "rewrite_query",
    })
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", "grade_groundedness")
    graph.add_conditional_edges("grade_groundedness", _route_after_groundedness, {
        "end": END, "pop_and_retry": "pop_and_retry",
    })
    graph.add_edge("pop_and_retry", "generate")
    return graph.compile()


_COMPILED_GRAPH = None


def get_compiled_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_corrective_rag_graph()
    return _COMPILED_GRAPH


async def run_corrective_rag(
    query: str,
    collection_name: str | None = None,
    prompt_top_k: Optional[int] = None,
    prompt_char_limit: Optional[int] = None,
) -> ChatResponse:
    settings = get_settings()
    app = get_compiled_graph()
    initial_state: CorrectiveRagState = {
        "original_query": query,
        "current_query": query,
        "collection_name": collection_name,
        "prompt_top_k": prompt_top_k,
        "prompt_char_limit": prompt_char_limit,
        "passages": [],
        "messages": [],
        "retries": 0,
        "max_retries": settings.CORRECTIVE_MAX_RETRIES,
        "relevance_ok": False,
        "groundedness_ok": False,
        "final_answer": None,
        "trace": [],
        "last_provider": None,
        "last_model": None,
    }
    final_state = await app.ainvoke(initial_state)

    return ChatResponse(
        answer=final_state["final_answer"] or "Unable to generate a grounded answer.",
        mode=RagMode.CORRECTIVE_RAG,
        retrieved_context=final_state["passages"],
        execution_trace=final_state["trace"],
        model_used=final_state["last_model"],
        provider_used=final_state["last_provider"],
    )