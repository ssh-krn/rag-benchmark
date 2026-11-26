"""
app/graphs/agentic_rag.py

Mode 4 — Autonomous Agentic RAG (LangGraph ReAct + Audit Loop).
Flow: Agent Reasoner node (bound with search_textbook_chunks + search_web_tavily)
-> ToolNode (max_tool_calls=4) -> Groundedness Audit node (popping rejected
AIMessage drafts on retry).

Data source: ChromaDB chunks (search_textbook_chunks) + live web (search_web_tavily).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.config import get_settings
from app.core.resilient_llm import get_chat_model
from app.graphs._prompt_budget import budget_passages
from app.schemas.chat import ChatResponse, ExecutionStep, RagMode, SourceMetadata
from app.tools.chroma_search import search_chunks_with_metadata, search_textbook_chunks
from app.tools.web_search import search_web_tavily, web_search_with_metadata

AGENT_SYSTEM_PROMPT = """You are an autonomous research agent. You have two tools:
- search_textbook_chunks: search a pre-populated textbook knowledge base (ChromaDB). Prefer this first.
- search_web_tavily: search the live web. Use only if the textbook search is insufficient.

Reason step by step, call tools as needed (you may call tools multiple times), then produce a final
answer that cites each supporting source as [Passage X] (textbook) or [Web X] (url) inline."""

TOOLS = [search_textbook_chunks, search_web_tavily]


class AgenticRagState(TypedDict):
    original_query: str
    collection_name: Optional[str]
    prompt_top_k: Optional[int]
    prompt_char_limit: Optional[int]
    messages: List[BaseMessage]
    tool_call_count: int
    max_tool_calls: int
    retries: int
    max_retries: int
    groundedness_ok: bool
    final_answer: Optional[str]
    trace: List[ExecutionStep]
    last_provider: Optional[str]
    last_model: Optional[str]
    collected_sources: List[SourceMetadata]


async def node_agent_reasoner(state: AgenticRagState) -> AgenticRagState:
    router = get_chat_model(role="reasoning")
    bound = router.bind_tools(TOOLS)

    if not state["messages"]:
        state["messages"].append(SystemMessage(content=AGENT_SYSTEM_PROMPT))
        state["messages"].append(HumanMessage(content=state["original_query"]))

    response = await bound.ainvoke(state["messages"])
    state["messages"].append(response)
    state["last_provider"], state["last_model"] = router.last_provider, router.last_model

    tool_calls = getattr(response, "tool_calls", None) or []
    state["trace"].append(ExecutionStep(
        node="agent_reasoner",
        description=f"Agent step produced {'a tool call' if tool_calls else 'a final answer'}.",
        detail={"provider": router.last_provider, "model": router.last_model, "tool_calls": len(tool_calls)},
    ))
    if not tool_calls:
        state["final_answer"] = getattr(response, "content", str(response))
    return state


async def node_tools(state: AgenticRagState) -> AgenticRagState:
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", []) or []
    prev_len = len(state["messages"])
    tool_node = ToolNode(TOOLS)
    result = await tool_node.ainvoke({"messages": state["messages"]})
    new_messages = result["messages"]

    # Trim newly appended ToolMessages only when prompt_char_limit is explicitly requested (eval mode)
    char_limit = state.get("prompt_char_limit")
    if char_limit is not None:
        max_tool_chars = max(char_limit * (state.get("prompt_top_k") or 3), 1200)
        for msg in new_messages[prev_len:]:
            if isinstance(msg, ToolMessage) and isinstance(msg.content, str) and len(msg.content) > max_tool_chars:
                msg.content = msg.content[:max_tool_chars] + "\n...[truncated for prompt budget]"

    state["messages"] = new_messages
    state["tool_call_count"] += len(tool_calls)

    for tc in tool_calls:
        if tc["name"] == "search_textbook_chunks":
            state["collected_sources"].extend(
                search_chunks_with_metadata(tc["args"].get("query", ""), collection_name=state["collection_name"],
                                             k=8, rerank_top_k=4)
            )
        elif tc["name"] == "search_web_tavily":
            state["collected_sources"].extend(web_search_with_metadata(tc["args"].get("query", "")))

    state["trace"].append(ExecutionStep(
        node="tool_node",
        description=f"Executed {len(tool_calls)} tool call(s). Total tool calls so far: {state['tool_call_count']}.",
    ))
    return state


async def node_groundedness_audit(state: AgenticRagState) -> AgenticRagState:
    grader = get_chat_model(role="grader")
    budgeted_sources = budget_passages(
        state["collected_sources"], state.get("prompt_top_k"), state.get("prompt_char_limit")
    )
    sources_block = "\n\n".join(f"[{i}] {s.text}" for i, s in enumerate(budgeted_sources, start=1))
    prompt = (
        "You are a strict hallucination grader for an agentic RAG system. Given the gathered sources "
        "and the agent's draft answer, respond with exactly one word: 'grounded' if every claim in the "
        f"draft is supported by the sources, or 'hallucinated' otherwise.\n\nSources:\n{sources_block}\n\n"
        f"Draft answer:\n{state['final_answer']}"
    )
    response = await grader.ainvoke(prompt)
    verdict = getattr(response, "content", str(response)).strip().lower()
    state["groundedness_ok"] = verdict.startswith("grounded")
    state["trace"].append(ExecutionStep(
        node="groundedness_audit",
        description=f"Audit verdict: {'GROUNDED' if state['groundedness_ok'] else 'HALLUCINATED'}",
        detail={"provider": grader.last_provider, "model": grader.last_model},
    ))
    return state


async def node_pop_and_retry(state: AgenticRagState) -> AgenticRagState:
    if state["messages"] and isinstance(state["messages"][-1], AIMessage):
        state["messages"].pop()
    state["messages"].append(HumanMessage(
        content="Your previous answer was not fully grounded in the gathered sources and was discarded. "
                "Re-answer strictly using only the sources already gathered (gather more via tools if needed), "
                "citing [Passage X] or [Web X] for every claim."
    ))
    state["retries"] += 1
    state["final_answer"] = None
    state["trace"].append(ExecutionStep(
        node="pop_and_retry",
        description="Popped rejected AIMessage draft and appended corrective retry prompt.",
    ))
    return state


def _route_after_agent(state: AgenticRagState) -> str:
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", []) or []
    if tool_calls and state["tool_call_count"] < state["max_tool_calls"]:
        return "tools"
    if not state.get("final_answer"):
        return "audit"
    return "audit"


def _route_after_tools(state: AgenticRagState) -> str:
    return "agent"


def _route_after_audit(state: AgenticRagState) -> str:
    if state["groundedness_ok"]:
        return "end"
    if state["retries"] < state["max_retries"]:
        return "pop_and_retry"
    return "end"


def build_agentic_rag_graph():
    graph = StateGraph(AgenticRagState)
    graph.add_node("agent", node_agent_reasoner)
    graph.add_node("tools", node_tools)
    graph.add_node("audit", node_groundedness_audit)
    graph.add_node("pop_and_retry", node_pop_and_retry)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _route_after_agent, {"tools": "tools", "audit": "audit"})
    graph.add_conditional_edges("tools", _route_after_tools, {"agent": "agent"})
    graph.add_conditional_edges("audit", _route_after_audit, {"end": END, "pop_and_retry": "pop_and_retry"})
    graph.add_edge("pop_and_retry", "agent")
    return graph.compile()


_COMPILED_GRAPH = None


def get_compiled_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_agentic_rag_graph()
    return _COMPILED_GRAPH


async def run_agentic_rag(
    query: str,
    collection_name: str | None = None,
    prompt_top_k: Optional[int] = None,
    prompt_char_limit: Optional[int] = None,
) -> ChatResponse:
    settings = get_settings()
    app = get_compiled_graph()
    initial_state: AgenticRagState = {
        "original_query": query,
        "collection_name": collection_name,
        "prompt_top_k": prompt_top_k,
        "prompt_char_limit": prompt_char_limit,
        "messages": [],
        "tool_call_count": 0,
        "max_tool_calls": settings.AGENTIC_MAX_TOOL_CALLS,
        "retries": 0,
        "max_retries": 2,
        "groundedness_ok": False,
        "final_answer": None,
        "trace": [],
        "last_provider": None,
        "last_model": None,
        "collected_sources": [],
    }
    final_state = await app.ainvoke(initial_state, config={"recursion_limit": 50})

    return ChatResponse(
        answer=final_state["final_answer"] or "Unable to generate a grounded answer within the tool-call budget.",
        mode=RagMode.AGENTIC_RAG,
        retrieved_context=final_state["collected_sources"],
        execution_trace=final_state["trace"],
        model_used=final_state["last_model"],
        provider_used=final_state["last_provider"],
    )