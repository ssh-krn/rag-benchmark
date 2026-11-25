"""
app/config.py
Centralized configuration via pydantic-settings. All values are overridable
through environment variables / a .env file (see .env.example).
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- Primary / fallback free-tier LLM providers ----
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    TAVILY_API_KEY: str = ""

    # ---- Ollama cloud ----
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_API_KEY: str = ""
    OLLAMA_LIGHTRAG_MODEL: str = "deepseek-v4.1-flash:cloud"
    OLLAMA_JUDGE_FALLBACK_MODEL: str = "deepseek-v4.1-flash:cloud"
    OLLAMA_COOLDOWN_SECONDS: int = 300
    OLLAMA_LOCAL_MODEL: str = "qwen2.5:3b"

    # ---- Verified free-tier model preferences ----
    GROQ_REASONING_MODEL: str = "openai/gpt-oss-120b"
    GROQ_REASONING_MODEL_FALLBACKS: List[str] = ["openai/gpt-oss-20b"]
    GROQ_FAST_GRADER_MODEL: str = "openai/gpt-oss-20b"
    GEMINI_FALLBACK_MODEL: str = "gemini-3.8-flash"
    GEMINI_FALLBACK_MODEL_CHAIN: List[str] = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
    OLLAMA_CLOUD_MODEL_CHAIN: List[str] = [
        "deepseek-v4.1-flash:cloud",
        "gpt-oss:120b-cloud",
        "kimi-k3.1-pro:cloud",
    ]

    # ---- Storage paths (pre-populated offline) ----
    CHROMA_DIR: str = "./chroma_db"
    DEFAULT_COLLECTION_NAME: str = "machine-learning"
    GRAPH_DIR: str = "./lightrag_graphs"
    DEFAULT_GRAPH_NAME: str = "machine-learning_graph.graphml"

    # ---- Local embeddings / reranker ----
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL_NAME: str = "ms-marco-MiniLM-L-12-v2"
    RERANKER_CACHE_DIR: str = "./cache"
    LOCAL_FILES_ONLY: bool = True

    # ---- DeepEval rate-limit / caching settings ----
    EVAL_CACHE_DIR: str = "./eval_cache"
    EVAL_RUNS_DIR: str = "./eval_runs"
    EVAL_BATCH_SIZE: int = 3
    EVAL_BATCH_SLEEP_SECONDS: float = 15.0
    LLM_MAX_RETRIES: int = 3
    LLM_TIMEOUT_SECONDS: float = 45.0

    # ---- CORS ----
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ---- RAG tuning knobs ----
    TRADITIONAL_RAG_K: int = 5
    HYBRID_DENSE_K: int = 10
    HYBRID_BM25_K: int = 10
    HYBRID_RERANK_K: int = 4
    CORRECTIVE_MAX_RETRIES: int = 2
    AGENTIC_MAX_TOOL_CALLS: int = 4
    AGENTIC_CHUNK_TRUNCATE_CHARS: int = 1200
    AGENTIC_WEB_MAX_RESULTS: int = 2
    LIGHTRAG_HOPS: int = 2

    @field_validator("CORS_ORIGINS", "GROQ_REASONING_MODEL_FALLBACKS", "GEMINI_FALLBACK_MODEL_CHAIN",
                      "OLLAMA_CLOUD_MODEL_CHAIN", mode="before")
    @classmethod
    def _parse_json_list(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [item.strip() for item in v.split(",") if item.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
