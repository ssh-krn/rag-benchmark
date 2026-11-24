"""
app/evals/runner.py

Batched DeepEval benchmark runner. Processes evaluation datasets in small
batches with inter-batch cooldowns to stay within free-tier RPM/TPM and
:cloud session/concurrency limits, deduplicating + caching aggressively and
checkpointing incrementally so a crashed/interrupted run can resume.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional

from app.config import get_settings
from app.core.resilient_llm import ResilientDeepEvalJudge
from app.evals.eval_cache import dedupe_test_cases, get_metrics_cache, get_rag_output_cache, make_key
from app.evals.checkpoint_manager import CheckpointManager, new_run_id
from app.graphs.agentic_rag import run_agentic_rag
from app.graphs.corrective_rag import run_corrective_rag
from app.graphs.hybrid_rag import run_hybrid_rag
from app.graphs.lightrag import run_lightrag
from app.graphs.traditional_rag import run_traditional_rag
from app.schemas.evals import EvalStatus, MetricResult, TestCaseResult

logger = logging.getLogger("eval_runner")

MODE_RUNNERS = {
    "traditional_rag": lambda q, collection, graph: run_traditional_rag(q, collection_name=collection),
    "hybrid_rag": lambda q, collection, graph: run_hybrid_rag(q, collection_name=collection),
    "corrective_rag": lambda q, collection, graph: run_corrective_rag(q, collection_name=collection),
    "agentic_rag": lambda q, collection, graph: run_agentic_rag(q, collection_name=collection),
    "lightrag": lambda q, collection, graph: run_lightrag(q, graph_name=graph),
}

_RUN_REGISTRY: Dict[str, CheckpointManager] = {}


def _load_dataset(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def _run_deepeval_metrics(question: str, actual_output: str, retrieval_context: List[str],
                                 expected_output: Optional[str]) -> List[MetricResult]:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    judge = ResilientDeepEvalJudge(role="grader")
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=expected_output,
    )

    metrics = [
        FaithfulnessMetric(model=judge, include_reason=True),
        AnswerRelevancyMetric(model=judge, include_reason=True),
    ]
    if expected_output:
        metrics.append(ContextualPrecisionMetric(model=judge, include_reason=True))
        metrics.append(ContextualRecallMetric(model=judge, include_reason=True))

    results: List[MetricResult] = []
    for metric in metrics:
        try:
            await metric.a_measure(test_case)
            results.append(MetricResult(
                metric_name=metric.__name__ if hasattr(metric, "__name__") else metric.__class__.__name__,
                score=metric.score,
                reason=getattr(metric, "reason", None),
                success=metric.is_successful() if hasattr(metric, "is_successful") else None,
            ))
        except Exception as exc:  # noqa: BLE001
            results.append(MetricResult(
                metric_name=metric.__class__.__name__,
                error=str(exc),
            ))
    return results


async def _process_single_case(golden: dict, mode: str, collection_name: Optional[str], graph_name: Optional[str],
                                checkpoint: CheckpointManager) -> TestCaseResult:
    question = golden["input"]
    expected_output = golden.get("expected_output")
    rag_cache = get_rag_output_cache()
    metrics_cache = get_metrics_cache()

    try:
        runner_fn = MODE_RUNNERS[mode]
        chat_response = await runner_fn(question, collection_name, graph_name)
        retrieved_texts = [c.text for c in chat_response.retrieved_context if c.text]

        output_key = make_key(question, mode, chat_response.model_used or "unknown", retrieved_texts)
        cached_output = rag_cache.get(output_key)
        cache_hit = cached_output is not None
        if not cache_hit:
            rag_cache.set(output_key, {
                "answer": chat_response.answer,
                "retrieved_texts": retrieved_texts,
                "model_used": chat_response.model_used,
                "provider_used": chat_response.provider_used,
            })
        else:
            chat_response.answer = cached_output["answer"]
            retrieved_texts = cached_output["retrieved_texts"]

        metrics_key = make_key(question, mode, chat_response.model_used or "unknown", retrieved_texts)
        cached_metrics = metrics_cache.get(metrics_key)
        if cached_metrics is not None:
            metrics = [MetricResult(**m) for m in cached_metrics]
        else:
            metrics = await _run_deepeval_metrics(question, chat_response.answer, retrieved_texts, expected_output)
            metrics_cache.set(metrics_key, [m.model_dump() for m in metrics])

        result = TestCaseResult(
            golden_id=golden["id"],
            mode=mode,
            question=question,
            cache_hit=cache_hit,
            rag_output=chat_response.answer,
            retrieved_contexts=retrieved_texts,
            metrics=metrics,
            provider_used=chat_response.provider_used,
            model_used=chat_response.model_used,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Test case failed: golden_id=%s mode=%s", golden.get("id"), mode)
        result = TestCaseResult(
            golden_id=golden["id"], mode=mode, question=question, error=str(exc),
        )

    checkpoint.append_result(result)
    return result


async def run_benchmark(
    dataset_path: str,
    target_modes: Optional[List[str]] = None,
    resume_run_id: Optional[str] = None,
    batch_size: Optional[int] = None,
    sleep_between_batches: Optional[float] = None,
) -> str:
    """Kicks off (or resumes) a benchmark run and returns its run_id. Intended to
    be scheduled as a FastAPI BackgroundTask; progress is polled via checkpoint."""
    settings = get_settings()
    batch_size = batch_size or settings.EVAL_BATCH_SIZE
    sleep_between_batches = sleep_between_batches if sleep_between_batches is not None else settings.EVAL_BATCH_SLEEP_SECONDS

    run_id = resume_run_id or new_run_id()
    checkpoint = CheckpointManager(run_id)
    _RUN_REGISTRY[run_id] = checkpoint

    dataset = _load_dataset(dataset_path)
    goldens = dataset.get("goldens", [])
    collection_name = dataset.get("collection_name")
    graph_name = dataset.get("graph_name")

    deduped_cases = dedupe_test_cases(goldens, target_modes or [])
    already_completed = checkpoint.load_completed_keys()

    checkpoint.write_meta({
        "status": EvalStatus.RUNNING,
        "started_at": checkpoint.read_meta().get("started_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_path": dataset_path,
        "total": len(deduped_cases),
    })

    pending = [
        case for case in deduped_cases
        if CheckpointManager.case_key(case["id"], case["_mode"]) not in already_completed
    ]
    logger.info("Run %s: %d total cases, %d already completed, %d pending",
                run_id, len(deduped_cases), len(already_completed), len(pending))

    try:
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            await asyncio.gather(*[
                _process_single_case(case, case["_mode"], collection_name, graph_name, checkpoint)
                for case in batch
            ])
            checkpoint.write_meta({
                **checkpoint.read_meta(),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
            if i + batch_size < len(pending):
                await asyncio.sleep(sleep_between_batches)

        checkpoint.write_meta({**checkpoint.read_meta(), "status": EvalStatus.COMPLETED,
                                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")})
    except Exception as exc:  # noqa: BLE001
        checkpoint.write_meta({**checkpoint.read_meta(), "status": EvalStatus.FAILED, "error": str(exc),
                                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")})
        logger.exception("Benchmark run %s failed", run_id)

    return run_id


def get_checkpoint(run_id: str) -> CheckpointManager:
    if run_id not in _RUN_REGISTRY:
        _RUN_REGISTRY[run_id] = CheckpointManager(run_id)
    return _RUN_REGISTRY[run_id]
