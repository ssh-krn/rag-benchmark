"""
app/evals/eval_cache.py

Disk-backed JSON caches keyed by sha256(question + mode + model + retrieved_contexts)
so re-running an evaluation never wastes free-tier or :cloud quota on cases that
were already generated / scored.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any, Dict, Optional

from app.config import get_settings

_lock = threading.Lock()


def _cache_key(question: str, mode: str, model: str, retrieved_contexts: list[str]) -> str:
    payload = json.dumps(
        {"question": question, "mode": mode, "model": model, "contexts": retrieved_contexts},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JSONDiskCache:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _read_all(self) -> Dict[str, Any]:
        with open(self.path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}

    def _write_all(self, data: Dict[str, Any]) -> None:
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.path)

    def get(self, key: str) -> Optional[Any]:
        with _lock:
            return self._read_all().get(key)

    def set(self, key: str, value: Any) -> None:
        with _lock:
            data = self._read_all()
            data[key] = value
            self._write_all(data)


def get_rag_output_cache() -> JSONDiskCache:
    settings = get_settings()
    return JSONDiskCache(os.path.join(settings.EVAL_CACHE_DIR, "rag_outputs_cache.json"))


def get_metrics_cache() -> JSONDiskCache:
    settings = get_settings()
    return JSONDiskCache(os.path.join(settings.EVAL_CACHE_DIR, "deepeval_metrics_cache.json"))


def make_key(question: str, mode: str, model: str, retrieved_contexts: list[str]) -> str:
    return _cache_key(question, mode, model, retrieved_contexts)


def dedupe_test_cases(goldens: list[dict], target_modes: list[str]) -> list[dict]:
    """Deduplicate identical (question, mode, collection_name) combinations before
    running RAG generation, preserving the first occurrence's metadata."""
    seen = set()
    deduped = []
    for golden in goldens:
        modes = target_modes or golden.get("additional_metadata", {}).get("target_modes", [])
        for mode in modes:
            key = (golden["input"], mode, golden.get("collection_name", ""))
            if key in seen:
                continue
            seen.add(key)
            deduped.append({**golden, "_mode": mode})
    return deduped
