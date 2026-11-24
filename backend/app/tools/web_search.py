"""
app/tools/web_search.py

Tavily-backed web search tool, used exclusively by agentic_rag. Truncates
results to keep Groq's 8K TPM free-tier limits from tripping 429s.
"""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.tools import tool

from app.config import get_settings
from app.schemas.chat import RetrievalMethod, SourceMetadata


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


def web_search_with_metadata(query: str) -> List[SourceMetadata]:
    settings = get_settings()
    if not settings.TAVILY_API_KEY:
        return []
    from langchain_tavily import TavilySearch

    search = TavilySearch(max_results=settings.AGENTIC_WEB_MAX_RESULTS, tavily_api_key=settings.TAVILY_API_KEY)
    raw = search.invoke({"query": query})
    results = raw.get("results", []) if isinstance(raw, dict) else raw

    out = []
    for r in results:
        out.append(
            SourceMetadata(
                source=r.get("title"),
                url=r.get("url"),
                score=r.get("score"),
                retrieval_method=RetrievalMethod.WEB_SEARCH,
                text=_truncate(r.get("content", ""), settings.AGENTIC_CHUNK_TRUNCATE_CHARS),
            )
        )
    return out


@tool("search_web_tavily")
def search_web_tavily(query: str) -> str:
    """Search the live web via Tavily for current information not found in the
    textbook collection. Use sparingly and only when ChromaDB passages are
    insufficient. Input should be a focused search query."""
    results = web_search_with_metadata(query)
    if not results:
        return "No web results available (Tavily not configured or returned nothing)."
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[Web {i} | {r.source} | {r.url}]\n{r.text}")
    return "\n\n".join(lines)
