"""
app/api/routes.py
All HTTP routes for the RAG platform backend.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.core.graph_store import get_graph_store_manager
from app.core.model_validator import get_cached_or_refresh_report
from app.core.resilient_llm import COOLDOWNS
from app.core.vector_store import get_vector_store_manager
from app.evals.checkpoint_manager import new_run_id
from app.evals.runner import get_checkpoint, run_benchmark
from app.graphs.agentic_rag import run_agentic_rag
from app.graphs.corrective_rag import run_corrective_rag
from app.graphs.hybrid_rag import run_hybrid_rag
from app.graphs.lightrag import run_lightrag
from app.graphs.traditional_rag import run_traditional_rag
from app.schemas.chat import ChatRequest, ChatResponse, RagMode
from app.schemas.evals import CheckpointReport, EvalRunRequest

logger = logging.getLogger("routes")
router = APIRouter()

_LIVE_EVAL_SEMAPHORE = asyncio.Semaphore(1)
FRONTEND_EVAL_SUMMARY_PATH = "./eval_results/frontend_eval_summary.json"

MODE_DISPATCH = {
    RagMode.TRADITIONAL_RAG: lambda req: run_traditional_rag(req.query, collection_name=req.collection_name),
    RagMode.HYBRID_RAG: lambda req: run_hybrid_rag(req.query, collection_name=req.collection_name),
    RagMode.CORRECTIVE_RAG: lambda req: run_corrective_rag(req.query, collection_name=req.collection_name),
    RagMode.AGENTIC_RAG: lambda req: run_agentic_rag(req.query, collection_name=req.collection_name),
    RagMode.LIGHTRAG: lambda req: run_lightrag(req.query, graph_name=req.graph_name),
}


@router.get("/api/health")
async def health():
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
            yield {"event": "token", "data": json.dumps({"text": response.answer})}
            yield {"event": "final", "data": response.model_dump_json()}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming chat request failed")
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}

    return EventSourceResponse(event_generator())


async def _guarded_live_eval_task(
    dataset_path: str,
    target_modes: Optional[list[str]],
    resume_run_id: str,
    batch_size: Optional[int],
    sleep_between_batches: Optional[float],
    cleanup_temp_path: Optional[str] = None,
) -> None:
    try:
        async with _LIVE_EVAL_SEMAPHORE:
            await run_benchmark(
                dataset_path=dataset_path,
                target_modes=target_modes,
                resume_run_id=resume_run_id,
                batch_size=batch_size,
                sleep_between_batches=sleep_between_batches,
                live_mode=True,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Guarded live eval encountered an error: %s", exc)
    finally:
        if cleanup_temp_path and os.path.exists(cleanup_temp_path):
            try:
                os.remove(cleanup_temp_path)
            except OSError:
                pass


@router.post("/api/evals/run")
async def start_eval_run(req: EvalRunRequest, background_tasks: BackgroundTasks):
    if _LIVE_EVAL_SEMAPHORE.locked():
        raise HTTPException(
            status_code=429,
            detail="A live evaluation is already in progress. Please view the pre-computed golden benchmark or retry in a moment.",
        )

    # Guardrail: cap live browser-triggered evaluations to 1 question & at most 2 strategies
    capped_modes = (req.target_modes or ["traditional_rag"])[:2]
    dataset_path = req.dataset_path
    temp_dataset_path: Optional[str] = None

    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_ds = json.load(f)
            goldens = raw_ds.get("goldens", [])
            if len(goldens) > 1:
                sliced_ds = {**raw_ds, "goldens": goldens[:1]}
                fd, temp_dataset_path = tempfile.mkstemp(prefix="live_eval_", suffix=".json")
                with os.fdopen(fd, "w", encoding="utf-8") as tf:
                    json.dump(sliced_ds, tf)
                dataset_path = temp_dataset_path
        except Exception:  # noqa: BLE001
            pass

    run_id = req.resume_run_id or new_run_id()
    background_tasks.add_task(
        _guarded_live_eval_task,
        dataset_path=dataset_path,
        target_modes=capped_modes,
        resume_run_id=run_id,
        batch_size=1,
        sleep_between_batches=max(req.sleep_between_batches or 1.5, 1.5),
        cleanup_temp_path=temp_dataset_path,
    )
    return {
        "run_id": run_id,
        "status": "started",
        "capped_modes": capped_modes,
        "live_guardrail": "Scoped to 1 question and Faithfulness + AnswerRelevancy metrics to protect free-tier quotas.",
    }


@router.get("/api/evals/runs/{run_id}", response_model=CheckpointReport)
async def get_eval_run(run_id: str):
    checkpoint = get_checkpoint(run_id)
    meta = checkpoint.read_meta()
    total = meta.get("total", 0)
    return checkpoint.build_report(total=total)


@router.get("/api/evals/summary")
@router.get("/api/eval/summary")
async def get_precomputed_eval_summary():
    if not os.path.exists(FRONTEND_EVAL_SUMMARY_PATH):
        raise HTTPException(
            status_code=404,
            detail="Pre-computed benchmark summary not found yet. Run `python -m app.evals.runner` first.",
        )
    with open(FRONTEND_EVAL_SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)