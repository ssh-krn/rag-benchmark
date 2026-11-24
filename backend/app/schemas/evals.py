"""
app/schemas/evals.py
Contracts for DeepEval benchmark triggering/status endpoints.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvalStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvalRunRequest(BaseModel):
    dataset_path: str = Field(default="./data/golden_eval_dataset.json")
    target_modes: Optional[List[str]] = None  # defaults to each golden's own target_modes
    resume_run_id: Optional[str] = None
    batch_size: Optional[int] = None
    sleep_between_batches: Optional[float] = None


class MetricResult(BaseModel):
    metric_name: str
    score: Optional[float] = None
    reason: Optional[str] = None
    success: Optional[bool] = None
    error: Optional[str] = None


class TestCaseResult(BaseModel):
    golden_id: str
    mode: str
    question: str
    cache_hit: bool = False
    rag_output: Optional[str] = None
    retrieved_contexts: Optional[List[str]] = None
    metrics: List[MetricResult] = Field(default_factory=list)
    provider_used: Optional[str] = None
    model_used: Optional[str] = None
    error: Optional[str] = None


class ModeMetricsSummary(BaseModel):
    mode: str
    count: int = 0
    avg_faithfulness: Optional[float] = None
    avg_answer_relevancy: Optional[float] = None
    avg_contextual_precision: Optional[float] = None
    avg_contextual_recall: Optional[float] = None


class CheckpointReport(BaseModel):
    run_id: str
    status: EvalStatus
    total: int
    completed: int
    failed: int
    cached_hits: int
    active_provider: Optional[str] = None
    metrics_summary: List[ModeMetricsSummary] = Field(default_factory=list)
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    error: Optional[str] = None
