"""
app/core/resilient_llm.py

Role-based, quota-aware multi-provider fallback router: Groq -> Mistral ->
Gemini -> local Ollama (per-role chain order varies — see build_candidate_chain).

Responsibilities:
  * Three roles, each with its own provider chain and sampling defaults:
      - "fast"       (alias: "grader")   — graders, routers, keyword extraction.
                                            High call volume, short output.
      - "synthesis"  (alias: "reasoning")— final grounded answer generation.
      - "eval_judge"                      — DeepEval strict-JSON-schema scoring.
  * Classify failures into three buckets, each with its own cooldown:
      - quota-exhausted 429 (session/hourly/weekly limit)  -> long cooldown
        (OLLAMA_COOLDOWN_SECONDS), skip straight to next candidate.
      - transient 429 (burst/concurrency throttle)          -> in-process
        retry with exponential backoff + jitter, no cooldown.
      - empty content (reasoning-token exhaustion)          -> short cooldown
        (RATE_LIMIT_COOLDOWN_SECONDS), next candidate.
      - anything else (404/deprecated model, timeout, 503,
        "all connection attempts failed", crashed local daemon, ...)
                                                              -> long cooldown
        (HARD_FAILURE_COOLDOWN_SECONDS), next candidate. This is the fix for
        a dead/crashed local Ollama model being silently re-tried on every
        single subsequent call instead of being avoided for a while.
  * SHA-256 disk cache (sqlite-backed): every candidate attempt is preceded by
    a cache lookup keyed on sha256(role:model:system_prompt:user_prompt:
    temperature), so an identical call against the same candidate model never
    pays for a real completion twice — not even across process restarts.
    Cache hits return in ~0ms and are logged as "0ms, 0 tokens". Disabled
    automatically for tool-bound (bind_tools) calls, where caching raw text
    content would silently discard structured tool_calls.
  * Expose `get_chat_model(role)` for LangGraph nodes needing a plain chat
    model with automatic failover baked in via `ResilientChatRouter`.
  * Expose `ResilientDeepEvalJudge`, a `DeepEvalBaseLLM` implementation that
    supports DeepEval's injected Pydantic `schema` argument for both sync and
    async generation, with `.with_structured_output()`, a JSON-extraction
    fallback for models lacking native structured output, and the same
    disk cache applied to structured (schema) results.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from app.config import get_settings

logger = logging.getLogger("resilient_llm")

# ---------------------------------------------------------------------------
# Role aliasing — "reasoning"/"grader" are the original role names used
# throughout app/graphs/*.py; they keep working unchanged, now mapped onto
# the new "synthesis"/"fast" role-chain machinery.
# ---------------------------------------------------------------------------

_ROLE_ALIASES = {"reasoning": "synthesis", "grader": "fast"}


def _normalize_role(role: str) -> str:
    return _ROLE_ALIASES.get(role, role)


def _role_sampling_params(role: str) -> Tuple[float, Optional[int]]:
    """(temperature, max_tokens) defaults for a role — also folded into the
    LLM cache key, so changing these invalidates old cache entries safely."""
    settings = get_settings()
    role = _normalize_role(role)
    if role == "fast":
        return settings.FAST_ROLE_TEMPERATURE, settings.FAST_ROLE_MAX_TOKENS
    if role == "synthesis":
        return settings.SYNTHESIS_ROLE_TEMPERATURE, settings.SYNTHESIS_ROLE_MAX_TOKENS
    if role == "eval_judge":
        return settings.EVAL_JUDGE_ROLE_TEMPERATURE, None  # judge needs full-length JSON, no cap
    return settings.SYNTHESIS_ROLE_TEMPERATURE, None

# ---------------------------------------------------------------------------
# 429 classification
# ---------------------------------------------------------------------------

QUOTA_EXHAUSTION_PATTERNS = [
    r"session usage limit",
    r"hourly usage limit",
    r"weekly usage limit",
    r"quota exceeded",
    r"insufficient_quota",
    r"upgrade for higher limits",
    r"billing",
    r"exceeded your current quota",
    r"daily limit",
]
_QUOTA_RE = re.compile("|".join(QUOTA_EXHAUSTION_PATTERNS), re.IGNORECASE)


def classify_429(error_message: str) -> str:
    """Return 'quota_exhausted' or 'transient_throttle' for a 429-style error."""
    if _QUOTA_RE.search(error_message or ""):
        return "quota_exhausted"
    return "transient_throttle"


def is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc)
    status_code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return status_code == 429 or "429" in msg or "rate limit" in msg.lower() or "too many requests" in msg.lower()


class EmptyContentError(RuntimeError):
    """Raised when a model returns a successful-but-empty response (common
    with reasoning models like gpt-oss-*/deepseek-* when max_tokens is too
    low and the whole budget is consumed by hidden reasoning tokens)."""


def assert_non_empty_content(result: Any) -> None:
    content = getattr(result, "content", None)
    if content is None:
        raise EmptyContentError("Model response has no `.content` attribute")
    if isinstance(content, str) and content.strip() == "":
        raise EmptyContentError("Model returned an empty string (possible reasoning-token exhaustion)")
    if isinstance(content, list) and len(content) == 0:
        raise EmptyContentError("Model returned an empty content list")


# ---------------------------------------------------------------------------
# SHA-256 disk cache for LLM completions (sqlite-backed, zero extra deps)
# ---------------------------------------------------------------------------


def _canonicalize_messages(messages: Any) -> Tuple[str, str]:
    """Best-effort extraction of (system_prompt, user_prompt) text from
    whatever shape `messages` is — a plain string, a list of
    {"role":..., "content":...} dicts, or LangChain BaseMessage objects.
    Used ONLY to build a stable cache key; never sent to a model."""
    if isinstance(messages, str):
        return "", messages

    system_parts: List[str] = []
    user_parts: List[str] = []
    try:
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = str(m.get("content", ""))
            else:
                role = getattr(m, "type", "user")  # BaseMessage.type: "system"|"human"|"ai"|"tool"
                content = str(getattr(m, "content", ""))
            if role == "system":
                system_parts.append(content)
            elif role in ("user", "human"):
                user_parts.append(content)
            else:
                user_parts.append(f"[{role}] {content}")
    except TypeError:
        return "", str(messages)

    return "\n".join(system_parts), "\n".join(user_parts)


def _cache_key(role: str, model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    raw = f"{role}:{model}:{system_prompt}:{user_prompt}:{temperature}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class _CachedResponse:
    """Mimics the subset of a LangChain AIMessage that every call site reads
    (`getattr(response, "content", str(response))`), for a cache hit."""
    content: str


class LLMCompletionCache:
    """SHA-256-keyed, sqlite-backed cache of LLM completions. One row per
    (role, model, system_prompt, user_prompt, temperature) combination —
    deliberately per-candidate-model so a cache hit never silently credits a
    different model's past answer to one that's actually being invoked now."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS llm_cache ("
            "key TEXT PRIMARY KEY, content TEXT NOT NULL, provider TEXT, model TEXT, created_at TEXT)"
        )
        self._conn.commit()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute("SELECT content FROM llm_cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, key: str, content: str, provider: str, model: str) -> None:
        if not content:
            return  # never cache an empty response
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, content, provider, model, created_at) VALUES (?,?,?,?,?)",
                (key, content, provider, model, datetime.now(timezone.utc).isoformat()),
            )
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]


_cache_instance: Optional[LLMCompletionCache] = None
_cache_lock = threading.Lock()


def get_llm_cache() -> Optional[LLMCompletionCache]:
    """Returns the process-wide LLM cache, or None if disabled via
    settings.LLM_CACHE_ENABLED / disable_cache_globally()."""
    global _cache_instance
    settings = get_settings()
    if not settings.LLM_CACHE_ENABLED:
        return None
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = LLMCompletionCache(settings.LLM_CACHE_PATH)
    return _cache_instance


def disable_cache_globally() -> None:
    """Used by the `--no-cache` CLI flag on the offline eval runner — flips
    the (process-wide, lru_cache'd) Settings singleton so every subsequent
    get_llm_cache() call returns None for the rest of this process."""
    get_settings().LLM_CACHE_ENABLED = False
    logger.info("LLM completion cache disabled for this process (--no-cache).")


# ---------------------------------------------------------------------------
# Circuit breaker / cooldown bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class ModelCooldown:
    provider: str
    model: str
    cooldown_until: float = 0.0
    reason: str = ""

    def active(self) -> bool:
        return time.time() < self.cooldown_until


class CooldownRegistry:
    """Tracks which (provider, model) pairs are temporarily disabled."""

    def __init__(self) -> None:
        self._cooldowns: Dict[Tuple[str, str], ModelCooldown] = {}

    def trip(self, provider: str, model: str, seconds: float, reason: str) -> None:
        key = (provider, model)
        self._cooldowns[key] = ModelCooldown(provider, model, time.time() + seconds, reason)
        logger.warning("Cooldown tripped for %s/%s (%ss): %s", provider, model, seconds, reason)

    def is_blocked(self, provider: str, model: str) -> bool:
        cd = self._cooldowns.get((provider, model))
        return bool(cd and cd.active())

    def snapshot(self) -> List[Dict[str, Any]]:
        return [
            {
                "provider": cd.provider,
                "model": cd.model,
                "seconds_remaining": max(0, int(cd.cooldown_until - time.time())),
                "reason": cd.reason,
            }
            for cd in self._cooldowns.values()
            if cd.active()
        ]


COOLDOWNS = CooldownRegistry()


# ---------------------------------------------------------------------------
# Provider candidate chain
# ---------------------------------------------------------------------------


@dataclass
class ModelCandidate:
    provider: str  # "groq" | "gemini" | "ollama_cloud"
    model_name: str
    factory: Callable[[], Any]


def _models_for_role(role: str, chains: Dict[str, List[str]], legacy: List[str]) -> List[str]:
    """Picks the right per-role model chain, falling back to the pre-role-based
    `legacy` list for any role name not in `chains` (keeps unknown/custom role
    strings working rather than silently returning no candidates)."""
    norm = _normalize_role(role)
    return chains.get(norm, legacy)


def _build_groq_candidates(role: str, temperature: float, max_tokens: Optional[int]) -> List[ModelCandidate]:
    settings = get_settings()
    if not settings.GROQ_API_KEY:
        return []
    from langchain_groq import ChatGroq

    models = _models_for_role(
        role,
        {
            "fast": settings.GROQ_FAST_MODEL_CHAIN,
            "synthesis": settings.GROQ_SYNTHESIS_MODEL_CHAIN,
            "eval_judge": settings.GROQ_JUDGE_MODEL_CHAIN,
        },
        legacy=[settings.GROQ_REASONING_MODEL, *settings.GROQ_REASONING_MODEL_FALLBACKS],
    )
    candidates = []
    for m in models:
        candidates.append(
            ModelCandidate(
                provider="groq",
                model_name=m,
                factory=lambda m=m: ChatGroq(
                    model=m,
                    api_key=settings.GROQ_API_KEY,
                    max_retries=0,  # we handle retries ourselves
                    timeout=settings.LLM_TIMEOUT_SECONDS,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            )
        )
    return candidates


def _build_mistral_candidates(role: str, temperature: float, max_tokens: Optional[int]) -> List[ModelCandidate]:
    settings = get_settings()
    if not settings.MISTRAL_API_KEY:
        return []
    from langchain_mistralai import ChatMistralAI

    models = _models_for_role(
        role,
        {
            "fast": settings.MISTRAL_FAST_MODEL_CHAIN,
            "synthesis": settings.MISTRAL_SYNTHESIS_MODEL_CHAIN,
            "eval_judge": settings.MISTRAL_JUDGE_MODEL_CHAIN,
        },
        legacy=settings.MISTRAL_SYNTHESIS_MODEL_CHAIN,
    )
    candidates = []
    for m in models:
        candidates.append(
            ModelCandidate(
                provider="mistral",
                model_name=m,
                factory=lambda m=m: ChatMistralAI(
                    model=m,
                    api_key=settings.MISTRAL_API_KEY,
                    max_retries=0,
                    timeout=settings.LLM_TIMEOUT_SECONDS,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            )
        )
    return candidates


def _build_gemini_candidates(role: str, temperature: float, max_tokens: Optional[int]) -> List[ModelCandidate]:
    settings = get_settings()
    if not settings.GEMINI_API_KEY:
        return []
    from langchain_google_genai import ChatGoogleGenerativeAI

    models = _models_for_role(
        role,
        {
            "fast": settings.GEMINI_FAST_MODEL_CHAIN,
            "synthesis": settings.GEMINI_SYNTHESIS_MODEL_CHAIN,
            "eval_judge": settings.GEMINI_JUDGE_MODEL_CHAIN,
        },
        legacy=settings.GEMINI_FALLBACK_MODEL_CHAIN,
    )
    candidates = []
    for m in models:
        candidates.append(
            ModelCandidate(
                provider="gemini",
                model_name=m,
                factory=lambda m=m: ChatGoogleGenerativeAI(
                    model=m,
                    google_api_key=settings.GEMINI_API_KEY,
                    max_retries=0,
                    timeout=settings.LLM_TIMEOUT_SECONDS,
                    temperature=temperature,
                    max_output_tokens=max_tokens,  # Gemini's constructor kwarg name differs from Groq/Mistral
                ),
            )
        )
    return candidates


def _build_ollama_cloud_candidates(role: str, temperature: float = 0.0, max_tokens: Optional[int] = None) -> List[ModelCandidate]:
    settings = get_settings()
    from langchain_ollama import ChatOllama

    base_url = settings.OLLAMA_BASE_URL
    headers = {}
    if settings.OLLAMA_API_KEY:
        base_url = "https://ollama.com"
        headers["Authorization"] = f"Bearer {settings.OLLAMA_API_KEY}"

    candidates = []
    for m in settings.OLLAMA_CLOUD_MODEL_CHAIN:
        candidates.append(
            ModelCandidate(
                provider="ollama_cloud",
                model_name=m,
                factory=lambda m=m: ChatOllama(
                    model=m,
                    base_url=base_url,
                    client_kwargs={"headers": headers} if headers else None,
                    timeout=settings.LLM_TIMEOUT_SECONDS,
                    temperature=temperature,
                    num_predict=max_tokens,  # Ollama's constructor kwarg name differs again
                ),
            )
        )
    return candidates


def build_candidate_chain(role: str = "reasoning") -> List[ModelCandidate]:
    """Ordered chain per role:
      fast (alias "grader")        -> Groq -> Mistral -> Gemini -> local Ollama (small/fast model)
      synthesis (alias "reasoning")-> Groq -> Gemini -> Mistral -> Ollama(:cloud)
      eval_judge                   -> Gemini -> Mistral -> Groq -> Ollama(:cloud)
    Gemini is primary for eval_judge: largest context window of the three
    cloud providers, which matters for multi-chunk faithfulness verification
    prompts that would otherwise truncate against a tighter TPM/context cap.
    """
    settings = get_settings()
    norm = _normalize_role(role)
    temperature, max_tokens = _role_sampling_params(role)

    if norm == "fast":
        return (
            _build_groq_candidates(role, temperature, max_tokens)
            + _build_mistral_candidates(role, temperature, max_tokens)
            + _build_gemini_candidates(role, temperature, max_tokens)
            + _build_ollama_local_candidates([settings.OLLAMA_FAST_LOCAL_MODEL])
        )
    if norm == "eval_judge":
        return (
            _build_gemini_candidates(role, temperature, max_tokens)
            + _build_mistral_candidates(role, temperature, max_tokens)
            + _build_groq_candidates(role, temperature, max_tokens)
            + _build_ollama_cloud_candidates(role, temperature, max_tokens)
        + _build_ollama_local_candidates([settings.OLLAMA_LOCAL_MODEL], temperature=temperature, max_tokens=max_tokens)
        )
    # "synthesis" (and any unrecognized role string, for safety)
    return (
        _build_groq_candidates(role, temperature, max_tokens)
        + _build_gemini_candidates(role, temperature, max_tokens)
        + _build_mistral_candidates(role, temperature, max_tokens)
        + _build_ollama_cloud_candidates(role, temperature, max_tokens)
        + _build_ollama_local_candidates([settings.OLLAMA_LOCAL_MODEL], temperature=temperature, max_tokens=max_tokens)
    )


def _build_ollama_local_candidates(models: List[str], base_url: Optional[str] = None,
                                    temperature: float = 0.0, max_tokens: Optional[int] = None) -> List[ModelCandidate]:
    """Genuinely local Ollama models (no `:cloud` suffix) — always talks to
    the local daemon directly, ignoring OLLAMA_API_KEY's cloud-routing
    override entirely. Requires `ollama pull <model>` to have been run."""
    from langchain_ollama import ChatOllama

    settings = get_settings()
    url = base_url or settings.OLLAMA_BASE_URL
    candidates = []
    for m in models:
        candidates.append(
            ModelCandidate(
                provider="ollama_local",
                model_name=m,
                factory=lambda m=m: ChatOllama(
                    model=m, base_url=url, timeout=settings.LLM_TIMEOUT_SECONDS,
                    temperature=temperature, num_predict=max_tokens,
                ),
            )
        )
    return candidates


def build_local_ollama_router(
    models: Optional[List[str]] = None,
    base_url: Optional[str] = None,
    role: str = "reasoning",
) -> "ResilientChatRouter":
    """Router restricted to local Ollama model(s) ONLY — no Groq, no Gemini,
    no Mistral, no Ollama :cloud fallback. For offline tooling (e.g.
    scripts/offline_build_graph.py) that should never touch a rate-limited
    or paywalled API at all. Pass multiple `models` to fail over between
    several locally-pulled models (e.g. a 14B model then a smaller 7B one)."""
    settings = get_settings()
    temperature, max_tokens = _role_sampling_params(role)
    model_list = models or [settings.OLLAMA_LOCAL_MODEL]
    candidates = _build_ollama_local_candidates(model_list, base_url=base_url, temperature=temperature, max_tokens=max_tokens)
    return ResilientChatRouter(role=role, candidates=candidates)


# ---------------------------------------------------------------------------
# Resilient router: handles retries + failover for plain chat invocations
# ---------------------------------------------------------------------------


class AllProvidersExhaustedError(RuntimeError):
    pass


class ResilientChatRouter:
    """
    Wraps an ordered chain of model candidates. `invoke`/`ainvoke` try the
    current candidate with bounded retries for transient 429s; a quota-type
    429, an empty-content response, or any other hard failure trips a
    cooldown (of varying length) and advances to the next candidate. Every
    attempt is preceded by a disk-cache lookup keyed on this exact
    (role, candidate model, prompt, temperature) combination.
    """

    def __init__(
        self,
        role: str = "reasoning",
        candidates: Optional[List[ModelCandidate]] = None,
        cache_disabled: bool = False,
    ):
        self.role = role
        self.candidates = candidates if candidates is not None else build_candidate_chain(role)
        self.settings = get_settings()
        self.temperature, self.max_tokens = _role_sampling_params(role)
        self.cache_disabled = cache_disabled
        self.last_provider: Optional[str] = None
        self.last_model: Optional[str] = None
        self.last_cache_hit: bool = False

    def _available_candidates(self) -> List[ModelCandidate]:
        return [c for c in self.candidates if not COOLDOWNS.is_blocked(c.provider, c.model_name)]

    def _backoff_sleep(self, attempt: int) -> float:
        base = min(2 ** attempt, 20)
        return base + random.uniform(0, 1.5)

    def _cache(self) -> Optional[LLMCompletionCache]:
        return None if self.cache_disabled else get_llm_cache()

    def invoke(self, messages: Any, **kwargs) -> Any:
        return asyncio.get_event_loop().run_until_complete(self.ainvoke(messages, **kwargs)) \
            if not asyncio.get_event_loop().is_running() else self._sync_invoke(messages, **kwargs)

    def _sync_invoke(self, messages: Any, **kwargs) -> Any:
        system_prompt, user_prompt = _canonicalize_messages(messages)
        cache = self._cache()
        last_exc: Optional[Exception] = None

        for candidate in self._available_candidates():
            cache_key = (
                _cache_key(self.role, candidate.model_name, system_prompt, user_prompt, self.temperature)
                if cache else None
            )
            if cache is not None and cache_key is not None:
                cached = cache.get(cache_key)
                if cached is not None:
                    logger.info(
                        "LLM cache HIT role=%s provider=%s model=%s — 0ms, 0 tokens",
                        self.role, candidate.provider, candidate.model_name,
                    )
                    self.last_provider, self.last_model, self.last_cache_hit = candidate.provider, candidate.model_name, True
                    return _CachedResponse(content=cached)

            llm = candidate.factory()
            for attempt in range(self.settings.LLM_MAX_RETRIES):
                try:
                    result = llm.invoke(messages, **kwargs)
                    assert_non_empty_content(result)
                    self.last_provider, self.last_model, self.last_cache_hit = candidate.provider, candidate.model_name, False
                    if cache is not None and cache_key is not None:
                        cache.set(cache_key, getattr(result, "content", str(result)), candidate.provider, candidate.model_name)
                    return result
                except EmptyContentError as exc:
                    last_exc = exc
                    logger.warning("Empty content from %s/%s: %s", candidate.provider, candidate.model_name, exc)
                    COOLDOWNS.trip(candidate.provider, candidate.model_name, self.settings.RATE_LIMIT_COOLDOWN_SECONDS, "empty content")
                    break  # next candidate — an empty response rarely fixes itself on immediate retry
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if is_rate_limit_error(exc):
                        kind = classify_429(str(exc))
                        if kind == "quota_exhausted":
                            COOLDOWNS.trip(
                                candidate.provider, candidate.model_name,
                                self.settings.OLLAMA_COOLDOWN_SECONDS, str(exc)[:300],
                            )
                            break  # move to next candidate, no more retries here
                        time.sleep(self._backoff_sleep(attempt))
                        continue
                    else:
                        logger.warning("Non-rate-limit error on %s/%s: %s", candidate.provider, candidate.model_name, exc)
                        COOLDOWNS.trip(candidate.provider, candidate.model_name, self.settings.HARD_FAILURE_COOLDOWN_SECONDS, str(exc)[:300])
                        break
        raise AllProvidersExhaustedError(f"All providers exhausted for role={self.role}: {last_exc}")

    async def ainvoke(self, messages: Any, **kwargs) -> Any:
        system_prompt, user_prompt = _canonicalize_messages(messages)
        cache = self._cache()
        last_exc: Optional[Exception] = None

        for candidate in self._available_candidates():
            cache_key = (
                _cache_key(self.role, candidate.model_name, system_prompt, user_prompt, self.temperature)
                if cache else None
            )
            if cache is not None and cache_key is not None:
                cached = cache.get(cache_key)
                if cached is not None:
                    logger.info(
                        "LLM cache HIT role=%s provider=%s model=%s — 0ms, 0 tokens",
                        self.role, candidate.provider, candidate.model_name,
                    )
                    self.last_provider, self.last_model, self.last_cache_hit = candidate.provider, candidate.model_name, True
                    return _CachedResponse(content=cached)

            llm = candidate.factory()
            for attempt in range(self.settings.LLM_MAX_RETRIES):
                try:
                    result = await llm.ainvoke(messages, **kwargs)
                    assert_non_empty_content(result)
                    self.last_provider, self.last_model, self.last_cache_hit = candidate.provider, candidate.model_name, False
                    if cache is not None and cache_key is not None:
                        cache.set(cache_key, getattr(result, "content", str(result)), candidate.provider, candidate.model_name)
                    return result
                except EmptyContentError as exc:
                    last_exc = exc
                    logger.warning("Empty content from %s/%s: %s", candidate.provider, candidate.model_name, exc)
                    COOLDOWNS.trip(candidate.provider, candidate.model_name, self.settings.RATE_LIMIT_COOLDOWN_SECONDS, "empty content")
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if is_rate_limit_error(exc):
                        kind = classify_429(str(exc))
                        if kind == "quota_exhausted":
                            COOLDOWNS.trip(
                                candidate.provider, candidate.model_name,
                                self.settings.OLLAMA_COOLDOWN_SECONDS, str(exc)[:300],
                            )
                            break
                        await asyncio.sleep(self._backoff_sleep(attempt))
                        continue
                    else:
                        logger.warning("Non-rate-limit error on %s/%s: %s", candidate.provider, candidate.model_name, exc)
                        COOLDOWNS.trip(candidate.provider, candidate.model_name, self.settings.HARD_FAILURE_COOLDOWN_SECONDS, str(exc)[:300])
                        break
        raise AllProvidersExhaustedError(f"All providers exhausted for role={self.role}: {last_exc}")

    def bind_tools(self, tools: List[Any]):
        """Returns a thin proxy whose .invoke/.ainvoke route through failover
        while delegating tool-binding to each underlying candidate lazily.
        Caching is disabled here deliberately: a tool-bound response's
        `.tool_calls` structure would be silently lost if we only cached
        `.content` text, which would corrupt agentic_rag's tool-routing."""
        return _ToolBoundRouter(self, tools)


class _ToolBoundRouter:
    def __init__(self, router: ResilientChatRouter, tools: List[Any]):
        self._router = router
        self._tools = tools

    def _bound_candidates(self) -> List[ModelCandidate]:
        bound = []
        for c in self._router.candidates:
            def factory(c=c):
                return c.factory().bind_tools(self._tools)
            bound.append(ModelCandidate(c.provider, c.model_name, factory))
        return bound

    def invoke(self, messages: Any, **kwargs) -> Any:
        router = ResilientChatRouter(self._router.role, self._bound_candidates(), cache_disabled=True)
        result = router._sync_invoke(messages, **kwargs)
        self._router.last_provider, self._router.last_model = router.last_provider, router.last_model
        return result

    async def ainvoke(self, messages: Any, **kwargs) -> Any:
        router = ResilientChatRouter(self._router.role, self._bound_candidates(), cache_disabled=True)
        result = await router.ainvoke(messages, **kwargs)
        self._router.last_provider, self._router.last_model = router.last_provider, router.last_model
        return result


def get_chat_model(role: str = "reasoning") -> ResilientChatRouter:
    """Factory used by LangGraph nodes. role in {"reasoning"/"synthesis",
    "grader"/"fast", "eval_judge"} — "reasoning" and "grader" are the
    original role names and keep working exactly as before."""
    return ResilientChatRouter(role=role)


# ---------------------------------------------------------------------------
# ResilientDeepEvalJudge
# ---------------------------------------------------------------------------

try:
    from deepeval.models.base_model import DeepEvalBaseLLM
except ImportError:  # pragma: no cover - allows import without deepeval installed
    class DeepEvalBaseLLM:  # type: ignore
        def __init__(self, model=None):
            self.model = model

        def load_model(self):
            return self.model

        def get_model_name(self):
            return "resilient-judge"


def _extract_json_block(text: str) -> str:
    """Best-effort extraction of a JSON object from free-form model text."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


class ResilientDeepEvalJudge(DeepEvalBaseLLM):
    """
    A DeepEval-compatible judge model that fails over across
    Gemini -> Mistral -> Groq -> Ollama(:cloud) (the "eval_judge" role chain)
    and supports DeepEval's schema-aware `generate(prompt, schema)` /
    `a_generate(prompt, schema)` signatures, with the same SHA-256 disk cache
    applied to structured (schema) results as plain completions get.
    """

    def __init__(self, role: str = "eval_judge"):
        self.router = ResilientChatRouter(role=role)
        super().__init__(model=self.router)

    def load_model(self):
        return self.router

    def get_model_name(self) -> str:
        return f"ResilientDeepEvalJudge({self.router.last_provider or 'unrouted'}/{self.router.last_model or 'unrouted'})"

    # -- structured generation helpers --
    def _structured_sync(self, prompt: str, schema: Optional[Type[BaseModel]]):
        if schema is None:
            response = self.router._sync_invoke(prompt)
            return getattr(response, "content", str(response))

        cache = self.router._cache()
        schema_tag = f"schema:{schema.__name__}"
        last_exc = None

        for candidate in self.router._available_candidates():
            cache_key = _cache_key(self.router.role, candidate.model_name, schema_tag, prompt, self.router.temperature) if cache else None
            if cache is not None and cache_key is not None:
                cached = cache.get(cache_key)
                if cached is not None:
                    try:
                        parsed = schema(**json.loads(cached))
                        logger.info("LLM cache HIT (judge) role=%s provider=%s model=%s schema=%s — 0ms, 0 tokens",
                                    self.router.role, candidate.provider, candidate.model_name, schema.__name__)
                        self.router.last_provider, self.router.last_model = candidate.provider, candidate.model_name
                        return parsed
                    except Exception:  # noqa: BLE001
                        pass  # stale/corrupt cache entry — fall through to a real call

            llm = candidate.factory()
            try:
                structured_llm = llm.with_structured_output(schema)
                result = structured_llm.invoke(prompt)
                self.router.last_provider, self.router.last_model = candidate.provider, candidate.model_name
                if cache is not None and cache_key is not None:
                    cache.set(cache_key, result.model_dump_json(), candidate.provider, candidate.model_name)
                return result
            except NotImplementedError:
                pass
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if is_rate_limit_error(exc) and classify_429(str(exc)) == "quota_exhausted":
                    COOLDOWNS.trip(candidate.provider, candidate.model_name,
                                    get_settings().OLLAMA_COOLDOWN_SECONDS, str(exc)[:300])
                    continue
            # Fallback: raw generation + manual JSON parse against schema
            try:
                raw = llm.invoke(prompt)
                text = getattr(raw, "content", str(raw))
                if not text or not text.strip():
                    raise EmptyContentError("empty judge fallback content")
                parsed = json.loads(_extract_json_block(text))
                self.router.last_provider, self.router.last_model = candidate.provider, candidate.model_name
                result = schema(**parsed)
                if cache is not None and cache_key is not None:
                    cache.set(cache_key, result.model_dump_json(), candidate.provider, candidate.model_name)
                return result
            except Exception as exc2:  # noqa: BLE001
                last_exc = exc2
                COOLDOWNS.trip(candidate.provider, candidate.model_name,
                                get_settings().HARD_FAILURE_COOLDOWN_SECONDS, str(exc2)[:200])
                continue
        raise AllProvidersExhaustedError(f"Judge generation failed on all providers: {last_exc}")

    async def _structured_async(self, prompt: str, schema: Optional[Type[BaseModel]]):
        if schema is None:
            response = await self.router.ainvoke(prompt)
            return getattr(response, "content", str(response))

        cache = self.router._cache()
        schema_tag = f"schema:{schema.__name__}"
        last_exc = None

        for candidate in self.router._available_candidates():
            cache_key = _cache_key(self.router.role, candidate.model_name, schema_tag, prompt, self.router.temperature) if cache else None
            if cache is not None and cache_key is not None:
                cached = cache.get(cache_key)
                if cached is not None:
                    try:
                        parsed = schema(**json.loads(cached))
                        logger.info("LLM cache HIT (judge) role=%s provider=%s model=%s schema=%s — 0ms, 0 tokens",
                                    self.router.role, candidate.provider, candidate.model_name, schema.__name__)
                        self.router.last_provider, self.router.last_model = candidate.provider, candidate.model_name
                        return parsed
                    except Exception:  # noqa: BLE001
                        pass

            llm = candidate.factory()
            try:
                structured_llm = llm.with_structured_output(schema)
                result = await structured_llm.ainvoke(prompt)
                self.router.last_provider, self.router.last_model = candidate.provider, candidate.model_name
                if cache is not None and cache_key is not None:
                    cache.set(cache_key, result.model_dump_json(), candidate.provider, candidate.model_name)
                return result
            except NotImplementedError:
                pass
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if is_rate_limit_error(exc) and classify_429(str(exc)) == "quota_exhausted":
                    COOLDOWNS.trip(candidate.provider, candidate.model_name,
                                    get_settings().OLLAMA_COOLDOWN_SECONDS, str(exc)[:300])
                    continue
            try:
                raw = await llm.ainvoke(prompt)
                text = getattr(raw, "content", str(raw))
                if not text or not text.strip():
                    raise EmptyContentError("empty judge fallback content")
                parsed = json.loads(_extract_json_block(text))
                self.router.last_provider, self.router.last_model = candidate.provider, candidate.model_name
                result = schema(**parsed)
                if cache is not None and cache_key is not None:
                    cache.set(cache_key, result.model_dump_json(), candidate.provider, candidate.model_name)
                return result
            except Exception as exc2:  # noqa: BLE001
                last_exc = exc2
                COOLDOWNS.trip(candidate.provider, candidate.model_name,
                                get_settings().HARD_FAILURE_COOLDOWN_SECONDS, str(exc2)[:200])
                continue
        raise AllProvidersExhaustedError(f"Judge generation failed on all providers: {last_exc}")

    # -- DeepEvalBaseLLM required interface --
    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> Any:
        return self._structured_sync(prompt, schema)

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> Any:
        return await self._structured_async(prompt, schema)