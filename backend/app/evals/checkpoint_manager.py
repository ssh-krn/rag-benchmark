"""
app/evals/checkpoint_manager.py

Writes per-test-case results incrementally to ./eval_runs/{run_id}/results.jsonl
after every single item completes, and supports resuming a run by skipping
already-succeeded test cases and isolating only the failed/pending subset.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Dict, Iterable, List, Optional

from app.config import get_settings
from app.schemas.evals import CheckpointReport, EvalStatus, ModeMetricsSummary, TestCaseResult


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]


class CheckpointManager:
    def __init__(self, run_id: str):
        settings = get_settings()
        self.run_id = run_id
        self.run_dir = os.path.join(settings.EVAL_RUNS_DIR, run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.results_path = os.path.join(self.run_dir, "results.jsonl")
        self.meta_path = os.path.join(self.run_dir, "meta.json")

    # ---- case-level test key ----
    @staticmethod
    def case_key(golden_id: str, mode: str) -> str:
        return f"{golden_id}::{mode}"

    def load_completed_keys(self) -> Dict[str, TestCaseResult]:
        completed: Dict[str, TestCaseResult] = {}
        if not os.path.exists(self.results_path):
            return completed
        with open(self.results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    result = TestCaseResult(**obj)
                    if result.error is None:
                        completed[self.case_key(result.golden_id, result.mode)] = result
                except (json.JSONDecodeError, TypeError):
                    continue
        return completed

    def append_result(self, result: TestCaseResult) -> None:
        with open(self.results_path, "a", encoding="utf-8") as f:
            f.write(result.model_dump_json() + "\n")

    def write_meta(self, meta: dict) -> None:
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def read_meta(self) -> dict:
        if not os.path.exists(self.meta_path):
            return {}
        with open(self.meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_report(self, total: int, active_provider: Optional[str] = None) -> CheckpointReport:
        all_results: List[TestCaseResult] = []
        if os.path.exists(self.results_path):
            with open(self.results_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            all_results.append(TestCaseResult(**json.loads(line)))
                        except (json.JSONDecodeError, TypeError):
                            continue

        completed = sum(1 for r in all_results if r.error is None)
        failed = sum(1 for r in all_results if r.error is not None)
        cached = sum(1 for r in all_results if r.cache_hit)

        by_mode: Dict[str, List[TestCaseResult]] = {}
        for r in all_results:
            by_mode.setdefault(r.mode, []).append(r)

        def _avg(values: list) -> Optional[float]:
            values = [v for v in values if v is not None]
            return round(sum(values) / len(values), 4) if values else None

        summaries = []
        for mode, results in by_mode.items():
            metric_map: Dict[str, list] = {}
            for r in results:
                for m in r.metrics:
                    metric_map.setdefault(m.metric_name, []).append(m.score)
            summaries.append(ModeMetricsSummary(
                mode=mode,
                count=len(results),
                avg_faithfulness=_avg(metric_map.get("Faithfulness", [])),
                avg_answer_relevancy=_avg(metric_map.get("Answer Relevancy", [])),
                avg_contextual_precision=_avg(metric_map.get("Contextual Precision", [])),
                avg_contextual_recall=_avg(metric_map.get("Contextual Recall", [])),
            ))

        meta = self.read_meta()
        status = EvalStatus.COMPLETED if completed + failed >= total and total > 0 else (
            EvalStatus.RUNNING if (completed + failed) > 0 else EvalStatus.PENDING
        )

        return CheckpointReport(
            run_id=self.run_id,
            status=meta.get("status", status),
            total=total,
            completed=completed,
            failed=failed,
            cached_hits=cached,
            active_provider=active_provider,
            metrics_summary=summaries,
            started_at=meta.get("started_at"),
            updated_at=meta.get("updated_at"),
            error=meta.get("error"),
        )
