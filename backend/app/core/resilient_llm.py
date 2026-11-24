"""
app/core/resilient_llm.py

Quota-aware multi-provider fallback router for Groq -> Gemini -> Ollama(:cloud).

Responsibilities:
  * Classify 429 errors as either (a) transient burst/concurrency throttling
    (retry with exponential backoff + jitter) or (b) session/hourly/weekly
    quota exhaustion (skip retries on that model, trip a cooldown, fail over
    immediately to the next candidate in the chain).
  * Expose `get_chat_model(role)` for LangGraph nodes needing a plain chat
    model with automatic failover baked in via `ResilientChatRouter`.
  * Expose `ResilientDeepEvalJudge`, a `DeepEvalBaseLLM` implementation that
    supports DeepEval's injected Pydantic `schema` argument for both sync and
    async generation, with `.with_structured_output()` and a JSON-extraction
    fallback for models lacking native structured output.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from app.config import get_settings

logger = logging.getLogger("resilient_llm")

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


def _build_groq_candidates(role: str) -> List[ModelCandidate]:
    settings = get_settings()
    if not settings.GROQ_API_KEY:
        return []
    from langchain_groq import ChatGroq

    models = (
        [settings.GROQ_FAST_GRADER_MODEL]
        if role == "grader"
        else [settings.GROQ_REASONING_MODEL, *settings.GROQ_REASONING_MODEL_FALLBACKS]
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
                ),
            )
        )
    return candidates


def _build_gemini_candidates(role: str) -> List[ModelCandidate]:
    settings = get_settings()
    if not settings.GEMINI_API_KEY:
        return []
    from langchain_google_genai import ChatGoogleGenerativeAI

    candidates = []
    for m in settings.GEMINI_FALLBACK_MODEL_CHAIN:
        candidates.append(
            ModelCandidate(
                provider="gemini",
                model_name=m,
                factory=lambda m=m: ChatGoogleGenerativeAI(
                    model=m,
                    google_api_key=settings.GEMINI_API_KEY,
                    max_retries=0,
                    timeout=settings.LLM_TIMEOUT_SECONDS,
                ),
            )
        )
    return candidates


def _build_ollama_cloud_candidates(role: str) -> List[ModelCandidate]:
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
                ),
            )
        )
    return candidates


def build_candidate_chain(role: str = "reasoning") -> List[ModelCandidate]:
    """Ordered chain: Groq -> Gemini -> Ollama(:cloud)."""
    return (
        _build_groq_candidates(role)
        + _build_gemini_candidates(role)
        + _build_ollama_cloud_candidates(role)
    )


# ---------------------------------------------------------------------------
# Resilient router: handles retries + failover for plain chat invocations
# ---------------------------------------------------------------------------


class AllProvidersExhaustedError(RuntimeError):
    pass


class ResilientChatRouter:
    """
    Wraps an ordered chain of model candidates. `invoke`/`ainvoke` try the
    current candidate with bounded retries for transient 429s; a quota-type
    429 immediately trips a cooldown and advances to the next candidate.
    """

    def __init__(self, role: str = "reasoning", candidates: Optional[List[ModelCandidate]] = None):
        self.role = role
        self.candidates = candidates if candidates is not None else build_candidate_chain(role)
        self.settings = get_settings()
        self.last_provider: Optional[str] = None
        self.last_model: Optional[str] = None

    def _available_candidates(self) -> List[ModelCandidate]:
        return [c for c in self.candidates if not COOLDOWNS.is_blocked(c.provider, c.model_name)]

    def _backoff_sleep(self, attempt: int) -> float:
        base = min(2 ** attempt, 20)
        return base + random.uniform(0, 1.5)

    def invoke(self, messages: Any, **kwargs) -> Any:
        return asyncio.get_event_loop().run_until_complete(self.ainvoke(messages, **kwargs)) \
            if not asyncio.get_event_loop().is_running() else self._sync_invoke(messages, **kwargs)

    def _sync_invoke(self, messages: Any, **kwargs) -> Any:
        last_exc: Optional[Exception] = None
        for candidate in self._available_candidates():
            llm = candidate.factory()
            for attempt in range(self.settings.LLM_MAX_RETRIES):
                try:
                    result = llm.invoke(messages, **kwargs)
                    self.last_provider, self.last_model = candidate.provider, candidate.model_name
                    return result
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
                        break
        raise AllProvidersExhaustedError(f"All providers exhausted for role={self.role}: {last_exc}")

    async def ainvoke(self, messages: Any, **kwargs) -> Any:
        last_exc: Optional[Exception] = None
        for candidate in self._available_candidates():
            llm = candidate.factory()
            for attempt in range(self.settings.LLM_MAX_RETRIES):
                try:
                    result = await llm.ainvoke(messages, **kwargs)
                    self.last_provider, self.last_model = candidate.provider, candidate.model_name
                    return result
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
                        break
        raise AllProvidersExhaustedError(f"All providers exhausted for role={self.role}: {last_exc}")

    def bind_tools(self, tools: List[Any]):
        """Returns a thin proxy whose .invoke/.ainvoke route through failover
        while delegating tool-binding to each underlying candidate lazily."""
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
        router = ResilientChatRouter(self._router.role, self._bound_candidates())
        result = router._sync_invoke(messages, **kwargs)
        self._router.last_provider, self._router.last_model = router.last_provider, router.last_model
        return result

    async def ainvoke(self, messages: Any, **kwargs) -> Any:
        router = ResilientChatRouter(self._router.role, self._bound_candidates())
        result = await router.ainvoke(messages, **kwargs)
        self._router.last_provider, self._router.last_model = router.last_provider, router.last_model
        return result


def get_chat_model(role: str = "reasoning") -> ResilientChatRouter:
    """Factory used by LangGraph nodes. role in {"reasoning", "grader"}."""
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
    Groq -> Gemini -> Ollama(:cloud) and supports DeepEval's schema-aware
    `generate(prompt, schema)` / `a_generate(prompt, schema)` signatures.
    """

    def __init__(self, role: str = "grader"):
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

        last_exc = None
        for candidate in self.router._available_candidates():
            llm = candidate.factory()
            try:
                structured_llm = llm.with_structured_output(schema)
                result = structured_llm.invoke(prompt)
                self.router.last_provider, self.router.last_model = candidate.provider, candidate.model_name
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
                parsed = json.loads(_extract_json_block(text))
                self.router.last_provider, self.router.last_model = candidate.provider, candidate.model_name
                return schema(**parsed)
            except Exception as exc2:  # noqa: BLE001
                last_exc = exc2
                continue
        raise AllProvidersExhaustedError(f"Judge generation failed on all providers: {last_exc}")

    async def _structured_async(self, prompt: str, schema: Optional[Type[BaseModel]]):
        if schema is None:
            response = await self.router.ainvoke(prompt)
            return getattr(response, "content", str(response))

        last_exc = None
        for candidate in self.router._available_candidates():
            llm = candidate.factory()
            try:
                structured_llm = llm.with_structured_output(schema)
                result = await structured_llm.ainvoke(prompt)
                self.router.last_provider, self.router.last_model = candidate.provider, candidate.model_name
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
                parsed = json.loads(_extract_json_block(text))
                self.router.last_provider, self.router.last_model = candidate.provider, candidate.model_name
                return schema(**parsed)
            except Exception as exc2:  # noqa: BLE001
                last_exc = exc2
                continue
        raise AllProvidersExhaustedError(f"Judge generation failed on all providers: {last_exc}")

    # -- DeepEvalBaseLLM required interface --
    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> Any:
        return self._structured_sync(prompt, schema)

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> Any:
        return await self._structured_async(prompt, schema)
