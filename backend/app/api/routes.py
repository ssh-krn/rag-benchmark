"""
app/api/routes.py
All HTTP routes for the RAG platform backend.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.core.graph_store import get_graph_store_manager
from app.core.model_validator import get_cached_or_refresh_report
from app.core.resilient_llm import COOLDOWNS
from app.core.vector_store import get_vector_store_manager
from app.evals.runner import get_checkpoint, run_benchmark
from app.graphs.agentic_rag import run_agentic_rag
from app.graphs.corrective_rag import run_corrective_rag
from app.graphs.hybrid_rag import run_hybrid_rag
from app.graphs.lightrag import run_lightrag
from app.graphs.traditional_rag import run_traditional_rag
from app.schemas.chat import ChatRequest, ChatResponse, RagMode, SSEEvent
from app.schemas.evals import CheckpointReport, EvalRunRequest

logger = logging.getLogger("routes")
router = APIRouter()

MODE_DISPATCH = {
    RagMode.TRADITIONAL_RAG: lambda req: run_traditional_rag(req.query, collection_name=req.collection_name),
    RagMode.HYBRID_RAG: lambda req: run_hybrid_rag(req.query, collection_name=req.collection_name),
    RagMode.CORRECTIVE_RAG: lambda req: run_corrective_rag(req.query, collection_name=req.collection_name),
    RagMode.AGENTIC_RAG: lambda req: run_agentic_rag(req.query, collection_name=req.collection_name),
    RagMode.LIGHTRAG: lambda req: run_lightrag(req.query, graph_name=req.graph_name),
}


@router.get("/api/health")
async def health():
    settings = get_settings()
    vector_health = get_vector_store_manager().health()
    graph_health = get_graph_store_manager().health()
    verification = await get_cached_or_refresh_report()

    return {
        "status": "ok",
        "chromadb": vector_health,
        "lightrag_graph": graph_health,
        "active_cooldowns": COOLDOWNS.snapshot(),
        "preflight_verification": verification.to_dict(),
    }


@router.get("/api/collections")
async def list_collections():
    manager = get_vector_store_manager()
    graph_manager = get_graph_store_manager()
    return {
        "collections": manager.list_collections(),
        "lightrag_graphs": graph_manager.list_graphs(),
    }


@router.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if req.mode not in MODE_DISPATCH:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {req.mode}")
    try:
        return await MODE_DISPATCH[req.mode](req)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    if req.mode not in MODE_DISPATCH:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {req.mode}")

    async def event_generator():
        yield {"event": "trace", "data": json.dumps({"node": "dispatch", "description": f"Starting mode={req.mode}"})}
        try:
            response = await MODE_DISPATCH[req.mode](req)
            for step in response.execution_trace:
                yield {"event": "trace", "data": step.model_dump_json()}
            yield {"event": "context", "data": json.dumps([c.model_dump() for c in response.retrieved_context], default=str)}
            # Stream the final answer as a single token chunk (true token streaming
            # would require per-provider streaming callbacks wired into each graph).
            yield {"event": "token", "data": json.dumps({"text": response.answer})}
            yield {"event": "final", "data": response.model_dump_json()}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming chat request failed")
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}

    return EventSourceResponse(event_generator())


@router.post("/api/evals/run")
async def start_eval_run(req: EvalRunRequest, background_tasks: BackgroundTasks):
    from app.evals.runner import new_run_id
    from app.evals.checkpoint_manager import CheckpointManager

    run_id = req.resume_run_id or new_run_id()
    background_tasks.add_task(
        run_benchmark,
        dataset_path=req.dataset_path,
        target_modes=req.target_modes,
        resume_run_id=run_id,
        batch_size=req.batch_size,
        sleep_between_batches=req.sleep_between_batches,
    )
    return {"run_id": run_id, "status": "started"}


@router.get("/api/evals/runs/{run_id}", response_model=CheckpointReport)
async def get_eval_run(run_id: str):
    checkpoint = get_checkpoint(run_id)
    meta = checkpoint.read_meta()
    total = meta.get("total", 0)
    return checkpoint.build_report(total=total)
