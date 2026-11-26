"""
app/evals/runner.py

Batched DeepEval benchmark runner. Processes evaluation datasets in small
batches with inter-batch cooldowns to stay within free-tier RPM/TPM and
:cloud session/concurrency limits, deduplicating + caching aggressively,
checkpointing incrementally, and exporting frontend-ready summary artifacts.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.core.resilient_llm import ResilientDeepEvalJudge, disable_cache_globally
from app.evals.checkpoint_manager import CheckpointManager, new_run_id
from app.evals.eval_cache import dedupe_test_cases, get_metrics_cache, get_rag_output_cache, make_key
from app.graphs.agentic_rag import run_agentic_rag
from app.graphs.corrective_rag import run_corrective_rag
from app.graphs.hybrid_rag import run_hybrid_rag
from app.graphs.lightrag import run_lightrag
from app.graphs.traditional_rag import run_traditional_rag
from app.schemas.evals import EvalStatus, MetricResult, TestCaseResult

logger = logging.getLogger("eval_runner")

EVAL_PROMPT_TOP_K = 3
EVAL_PROMPT_CHAR_LIMIT = 1200

MODE_RUNNERS = {
    "traditional_rag": lambda q, collection, graph: run_traditional_rag(
        q, collection_name=collection, prompt_top_k=EVAL_PROMPT_TOP_K, prompt_char_limit=EVAL_PROMPT_CHAR_LIMIT
    ),
    "hybrid_rag": lambda q, collection, graph: run_hybrid_rag(
        q, collection_name=collection, prompt_top_k=EVAL_PROMPT_TOP_K, prompt_char_limit=EVAL_PROMPT_CHAR_LIMIT
    ),
    "corrective_rag": lambda q, collection, graph: run_corrective_rag(
        q, collection_name=collection, prompt_top_k=EVAL_PROMPT_TOP_K, prompt_char_limit=EVAL_PROMPT_CHAR_LIMIT
    ),
    "agentic_rag": lambda q, collection, graph: run_agentic_rag(
        q, collection_name=collection, prompt_top_k=EVAL_PROMPT_TOP_K, prompt_char_limit=EVAL_PROMPT_CHAR_LIMIT
    ),
    "lightrag": lambda q, collection, graph: run_lightrag(
        q, graph_name=graph, prompt_top_k=EVAL_PROMPT_TOP_K, prompt_char_limit=EVAL_PROMPT_CHAR_LIMIT
    ),
}

_RUN_REGISTRY: Dict[str, CheckpointManager] = {}


def _load_dataset(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def _run_deepeval_metrics(
    question: str,
    actual_output: str,
    retrieval_context: List[str],
    expected_output: Optional[str],
    live_mode: bool = False,
) -> List[MetricResult]:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    judge = ResilientDeepEvalJudge(role="eval_judge")
    budgeted_contexts = [c[:1000] for c in retrieval_context[:EVAL_PROMPT_TOP_K] if c]

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        retrieval_context=budgeted_contexts,
        expected_output=None if live_mode else expected_output,
    )

    metrics = [
        FaithfulnessMetric(model=judge, include_reason=True),
        AnswerRelevancyMetric(model=judge, include_reason=True),
    ]
    if expected_output and not live_mode:
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


async def _process_single_case(
    golden: dict,
    mode: str,
    collection_name: Optional[str],
    graph_name: Optional[str],
    checkpoint: CheckpointManager,
    live_mode: bool = False,
) -> TestCaseResult:
    question = golden["input"]
    expected_output = golden.get("expected_output")
    rag_cache = get_rag_output_cache()
    metrics_cache = get_metrics_cache()

    try:
        # Stage 1: Check cached RAG output by (question, mode, collection/graph) before invoking RAG graph
        gen_lookup_key = make_key(question, mode, f"{collection_name or ''}:{graph_name or ''}", [])
        cached_rag = rag_cache.get(gen_lookup_key)

        if cached_rag is not None:
            cache_hit = True
            answer = cached_rag["answer"]
            retrieved_texts = cached_rag["retrieved_texts"]
            model_used = cached_rag.get("model_used")
            provider_used = cached_rag.get("provider_used")
            latency_ms = cached_rag.get("latency_ms", 0)
            trace_steps = cached_rag.get("execution_trace", [])
        else:
            cache_hit = False
            t0 = time.perf_counter()
            runner_fn = MODE_RUNNERS[mode]
            chat_response = await runner_fn(question, collection_name, graph_name)
            latency_ms = round((time.perf_counter() - t0) * 1000)
            answer = chat_response.answer
            retrieved_texts = [c.text for c in chat_response.retrieved_context if c.text]
            model_used = chat_response.model_used
            provider_used = chat_response.provider_used
            trace_steps = [s.model_dump() for s in chat_response.execution_trace]

            rag_cache.set(gen_lookup_key, {
                "answer": answer,
                "retrieved_texts": retrieved_texts,
                "model_used": model_used,
                "provider_used": provider_used,
                "latency_ms": latency_ms,
                "execution_trace": trace_steps,
            })

        # Stage 2: Check cached DeepEval scores
        metric_scope = "live2" if live_mode else "full4"
        metrics_key = make_key(question, f"{mode}:{metric_scope}", model_used or "unknown", retrieved_texts[:EVAL_PROMPT_TOP_K])
        cached_metrics = metrics_cache.get(metrics_key)
        if cached_metrics is not None:
            metrics = [MetricResult(**m) for m in cached_metrics]
        else:
            metrics = await _run_deepeval_metrics(question, answer, retrieved_texts, expected_output, live_mode=live_mode)
            if any(m.score is not None for m in metrics):
                metrics_cache.set(metrics_key, [m.model_dump() for m in metrics])

        result = TestCaseResult(
            golden_id=golden["id"],
            mode=mode,
            question=question,
            cache_hit=cache_hit,
            rag_output=answer,
            retrieved_contexts=retrieved_texts,
            metrics=metrics,
            provider_used=provider_used,
            model_used=model_used,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Test case failed: golden_id=%s mode=%s", golden.get("id"), mode)
        result = TestCaseResult(
            golden_id=golden["id"], mode=mode, question=question, error=str(exc),
        )

    checkpoint.append_result(result)
    return result


def export_frontend_eval_summary(checkpoint: CheckpointManager, dataset: dict, output_dir: str = "./eval_results") -> dict:
    """Aggregates checkpoint results into ./eval_results/frontend_eval_summary.json
    and ./eval_results/EVAL_REPORT.md for direct display in the frontend About & Lab tabs."""
    os.makedirs(output_dir, exist_ok=True)
    rag_cache = get_rag_output_cache()
    goldens_by_id = {g["id"]: g for g in dataset.get("goldens", [])}

    all_results: List[TestCaseResult] = []
    if os.path.exists(checkpoint.results_path):
        with open(checkpoint.results_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        all_results.append(TestCaseResult(**json.loads(line)))
                    except Exception:  # noqa: BLE001
                        continue

    def _mean(vals: List[float]) -> Optional[float]:
        return round(sum(vals) / len(vals), 4) if vals else None

    by_mode: Dict[str, List[TestCaseResult]] = {}
    by_question: Dict[str, Dict[str, TestCaseResult]] = {}
    for r in all_results:
        if r.error is not None:
            continue
        by_mode.setdefault(r.mode, []).append(r)
        by_question.setdefault(r.golden_id, {})[r.mode] = r

    strategies_summary: Dict[str, Any] = {}
    for mode, items in by_mode.items():
        metric_scores: Dict[str, List[float]] = {}
        cat_scores: Dict[str, List[float]] = {}
        latencies: List[float] = []
        hits: List[int] = []

        for r in items:
            g = goldens_by_id.get(r.golden_id, {})
            cat = g.get("additional_metadata", {}).get("category", "general")
            case_vals: List[float] = []

            for m in r.metrics:
                if m.score is not None:
                    metric_scores.setdefault(m.metric_name, []).append(m.score)
                    case_vals.append(m.score)

            if case_vals:
                cat_scores.setdefault(cat, []).append(sum(case_vals) / len(case_vals))

            hits.append(len(r.retrieved_contexts))
            gen_key = make_key(
                r.question, mode, f"{dataset.get('collection_name') or ''}:{dataset.get('graph_name') or ''}", []
            )
            cached = rag_cache.get(gen_key)
            if cached and cached.get("latency_ms"):
                latencies.append(float(cached["latency_ms"]))

        f_avg = _mean(metric_scores.get("Faithfulness", []))
        ar_avg = _mean(metric_scores.get("Answer Relevancy", []))
        cp_avg = _mean(metric_scores.get("Contextual Precision", []))
        cr_avg = _mean(metric_scores.get("Contextual Recall", []))
        present = [v for v in (f_avg, ar_avg, cp_avg, cr_avg) if v is not None]

        p95 = None
        if latencies:
            s_lat = sorted(latencies)
            p95 = round(s_lat[min(len(s_lat) - 1, int(len(s_lat) * 0.95))])

        strategies_summary[mode] = {
            "evaluated_count": len(items),
            "faithfulness": f_avg,
            "answer_relevancy": ar_avg,
            "contextual_precision": cp_avg,
            "contextual_recall": cr_avg,
            "overall_score": _mean(present),
            "avg_latency_ms": round(statistics.mean(latencies)) if latencies else None,
            "p95_latency_ms": p95,
            "avg_retrieval_hits": round(statistics.mean(hits), 1) if hits else None,
            "category_breakdown": {cat: _mean(vals) for cat, vals in cat_scores.items()},
        }

    # Build up to 4 representative cross-strategy sample comparisons
    sample_comparisons: List[Dict[str, Any]] = []
    for gid in list(by_question.keys())[:4]:
        g = goldens_by_id.get(gid, {})
        mode_entries = {}
        for mode, res in by_question[gid].items():
            scores = {m.metric_name: m.score for m in res.metrics if m.score is not None}
            mode_entries[mode] = {
                "answer_preview": (res.rag_output or "")[:280],
                "scores": scores,
                "provider_used": res.provider_used,
                "model_used": res.model_used,
            }
        sample_comparisons.append({
            "id": gid,
            "category": g.get("additional_metadata", {}).get("category", "general"),
            "question": g.get("input", ""),
            "strategies": mode_entries,
        })

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": checkpoint.run_id,
        "corpus": {
            "title": "The Hundred-Page Machine Learning Book",
            "author": "Andriy Burkov",
            "collection_name": dataset.get("collection_name", "machine-learning"),
            "chunks": 376,
            "graph_nodes": 2188,
            "graph_edges": 1989,
        },
        "dataset_size": len(dataset.get("goldens", [])),
        "judge_role": "eval_judge (Gemini -> Mistral -> Groq -> Ollama)",
        "strategies": strategies_summary,
        "sample_comparisons": sample_comparisons,
    }

    summary_path = os.path.join(output_dir, "frontend_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Also write human-readable Markdown report
    md_lines = [
        f"# DeepEval Benchmark Report (`{checkpoint.run_id}`)",
        "",
        f"- **Corpus:** {summary['corpus']['title']} ({summary['corpus']['chunks']} chunks, {summary['corpus']['graph_nodes']} graph nodes / {summary['corpus']['graph_edges']} edges)",
        f"- **Dataset Size:** {summary['dataset_size']} stratified questions",
        f"- **Generated At:** {summary['generated_at']}",
        "",
        "| Strategy | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Overall | Avg Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for mode, s in strategies_summary.items():
        md_lines.append(
            f"| `{mode}` | {s['faithfulness']} | {s['answer_relevancy']} | "
            f"{s['contextual_precision']} | {s['contextual_recall']} | **{s['overall_score']}** | {s['avg_latency_ms']} |"
        )
    report_path = os.path.join(output_dir, "EVAL_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    logger.info("Exported frontend summary to %s and %s", summary_path, report_path)
    return summary


async def run_benchmark(
    dataset_path: str,
    target_modes: Optional[List[str]] = None,
    resume_run_id: Optional[str] = None,
    batch_size: Optional[int] = None,
    sleep_between_batches: Optional[float] = None,
    live_mode: bool = False,
) -> str:
    """Kicks off (or resumes) a benchmark run and returns its run_id."""
    settings = get_settings()
    batch_size = 1 if live_mode else (batch_size or min(settings.EVAL_BATCH_SIZE, 2))
    sleep_between_batches = (
        sleep_between_batches if sleep_between_batches is not None else max(settings.EVAL_BATCH_SLEEP_SECONDS, 1.5)
    )

    run_id = resume_run_id or new_run_id()
    checkpoint = CheckpointManager(run_id)
    _RUN_REGISTRY[run_id] = checkpoint

    dataset = _load_dataset(dataset_path)
    goldens = dataset.get("goldens", [])
    collection_name = dataset.get("collection_name")
    graph_name = dataset.get("graph_name")

    deduped_cases = dedupe_test_cases(goldens, target_modes or list(MODE_RUNNERS.keys()))
    already_completed = checkpoint.load_completed_keys()

    checkpoint.write_meta({
        "status": EvalStatus.RUNNING,
        "started_at": checkpoint.read_meta().get("started_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_path": dataset_path,
        "total": len(deduped_cases),
        "live_mode": live_mode,
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
                _process_single_case(case, case["_mode"], collection_name, graph_name, checkpoint, live_mode=live_mode)
                for case in batch
            ])
            checkpoint.write_meta({
                **checkpoint.read_meta(),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
            if i + batch_size < len(pending):
                await asyncio.sleep(sleep_between_batches)

        checkpoint.write_meta({
            **checkpoint.read_meta(),
            "status": EvalStatus.COMPLETED,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        if not live_mode:
            export_frontend_eval_summary(checkpoint, dataset)
    except Exception as exc:  # noqa: BLE001
        checkpoint.write_meta({
            **checkpoint.read_meta(),
            "status": EvalStatus.FAILED,
            "error": str(exc),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        logger.exception("Benchmark run %s failed", run_id)

    return run_id


def get_checkpoint(run_id: str) -> CheckpointManager:
    if run_id not in _RUN_REGISTRY:
        _RUN_REGISTRY[run_id] = CheckpointManager(run_id)
    return _RUN_REGISTRY[run_id]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run offline DeepEval benchmark")
    parser.add_argument("--dataset", default="data/eval_dataset_mini.json", help="Path to golden dataset JSON")
    parser.add_argument("--modes", nargs="*", default=None, help="Subset of RAG modes to run")
    parser.add_argument("--resume", default=None, help="Existing run_id to resume")
    parser.add_argument("--no-cache", action="store_true", help="Disable LLM completion cache")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if args.no_cache:
        disable_cache_globally()

    rid = asyncio.run(run_benchmark(dataset_path=args.dataset, target_modes=args.modes, resume_run_id=args.resume))
    print(f"Completed benchmark run_id={rid}")