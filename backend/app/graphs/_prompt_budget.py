"""
app/graphs/_prompt_budget.py

Prompt-only passage budgeting helper used by the RAG graph modules.
Slices and truncates copies of `SourceMetadata` for LLM prompt construction
without mutating the full `SourceMetadata` objects returned to the frontend.
"""
from __future__ import annotations

from typing import List, Optional

from app.schemas.chat import SourceMetadata


def budget_passages(
    passages: List[SourceMetadata],
    prompt_top_k: Optional[int] = None,
    prompt_char_limit: Optional[int] = None,
) -> List[SourceMetadata]:
    """Return a budgeted copy of `passages` for prompt assembly.

    When both `prompt_top_k` and `prompt_char_limit` are None (the default for
    live `/api/chat` and `/api/chat/stream` requests), returns `passages` unchanged.
    """
    if prompt_top_k is None and prompt_char_limit is None:
        return passages

    selected = passages[:prompt_top_k] if prompt_top_k is not None else passages
    if prompt_char_limit is None:
        return list(selected)

    budgeted: List[SourceMetadata] = []
    for p in selected:
        text = p.text or ""
        if len(text) > prompt_char_limit:
            text = text[:prompt_char_limit] + "..."
        budgeted.append(p.model_copy(update={"text": text}))
    return budgeted