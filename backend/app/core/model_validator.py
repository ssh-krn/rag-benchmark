"""
app/core/model_validator.py

Pre-flight verification run at startup (and callable from /api/health) that
checks each configured provider's API to confirm models actually exist and
are currently accessible, before any inference/grading/judging is attempted.

Builds an ordered `active_judge_chain` of verified (provider, model_name)
pairs across Groq -> Gemini -> Ollama(:cloud).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from app.config import get_settings
from app.core.resilient_llm import classify_429

logger = logging.getLogger("model_validator")


@dataclass
class ProviderReport:
    provider: str
    reachable: bool
    verified_models: List[str] = field(default_factory=list)
    rejected_models: Dict[str, str] = field(default_factory=dict)  # model -> reason
    error: Optional[str] = None


@dataclass
class VerificationReport:
    groq: ProviderReport
    gemini: ProviderReport
    ollama_cloud: ProviderReport
    active_judge_chain: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "groq": self.groq.__dict__,
            "gemini": self.gemini.__dict__,
            "ollama_cloud": self.ollama_cloud.__dict__,
            "active_judge_chain": self.active_judge_chain,
        }


async def _check_groq() -> ProviderReport:
    settings = get_settings()
    candidates = [settings.GROQ_REASONING_MODEL, *settings.GROQ_REASONING_MODEL_FALLBACKS,
                  settings.GROQ_FAST_GRADER_MODEL]
    if not settings.GROQ_API_KEY:
        return ProviderReport("groq", reachable=False, error="GROQ_API_KEY not set")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
            )
        if resp.status_code != 200:
            return ProviderReport("groq", reachable=False, error=f"HTTP {resp.status_code}: {resp.text[:200]}")
        available_ids = {m["id"] for m in resp.json().get("data", [])}
        verified, rejected = [], {}
        for m in candidates:
            if m in available_ids:
                verified.append(m)
            else:
                rejected[m] = "not present in /models listing (deprecated or unauthenticated)"
        return ProviderReport("groq", reachable=True, verified_models=verified, rejected_models=rejected)
    except Exception as exc:  # noqa: BLE001
        return ProviderReport("groq", reachable=False, error=str(exc))


async def _check_gemini() -> ProviderReport:
    settings = get_settings()
    candidates = settings.GEMINI_FALLBACK_MODEL_CHAIN
    if not settings.GEMINI_API_KEY:
        return ProviderReport("gemini", reachable=False, error="GEMINI_API_KEY not set")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": settings.GEMINI_API_KEY},
            )
        if resp.status_code != 200:
            return ProviderReport("gemini", reachable=False, error=f"HTTP {resp.status_code}: {resp.text[:200]}")
        available_names = {m["name"].split("/")[-1] for m in resp.json().get("models", [])}
        verified, rejected = [], {}
        for m in candidates:
            if m in available_names:
                verified.append(m)
            else:
                rejected[m] = "not present in models.list (deprecated, paywalled, or unauthenticated)"
        return ProviderReport("gemini", reachable=True, verified_models=verified, rejected_models=rejected)
    except Exception as exc:  # noqa: BLE001
        return ProviderReport("gemini", reachable=False, error=str(exc))


async def _probe_ollama_cloud_model(client: httpx.AsyncClient, base_url: str, headers: dict, model: str) -> Optional[str]:
    """Returns None if OK, or a rejection reason string."""
    try:
        resp = await client.post(
            f"{base_url}/api/chat",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {"num_predict": 1},
            },
            timeout=15.0,
        )
        if resp.status_code == 200:
            return None
        if resp.status_code == 429:
            kind = classify_429(resp.text)
            return f"429 ({kind}): {resp.text[:200]}"
        if resp.status_code in (401, 403):
            return f"auth failed ({resp.status_code}): {resp.text[:200]}"
        return f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as exc:  # noqa: BLE001
        return f"probe error: {exc}"


async def _check_ollama_cloud() -> ProviderReport:
    settings = get_settings()
    base_url = settings.OLLAMA_BASE_URL
    headers: Dict[str, str] = {}
    if settings.OLLAMA_API_KEY:
        base_url = "https://ollama.com"
        headers["Authorization"] = f"Bearer {settings.OLLAMA_API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            tags_resp = await client.get(f"{base_url}/api/tags", headers=headers)
        reachable = tags_resp.status_code == 200
    except Exception as exc:  # noqa: BLE001
        return ProviderReport("ollama_cloud", reachable=False, error=f"tags check failed: {exc}")

    if not reachable:
        return ProviderReport(
            "ollama_cloud", reachable=False,
            error=f"/api/tags returned HTTP {tags_resp.status_code}. Run `ollama signin` or set OLLAMA_API_KEY.",
        )

    verified, rejected = [], {}
    async with httpx.AsyncClient() as client:
        for model in settings.OLLAMA_CLOUD_MODEL_CHAIN:
            reason = await _probe_ollama_cloud_model(client, base_url, headers, model)
            if reason is None:
                verified.append(model)
            else:
                rejected[model] = reason
    return ProviderReport("ollama_cloud", reachable=True, verified_models=verified, rejected_models=rejected)


async def run_preflight_verification() -> VerificationReport:
    groq, gemini, ollama_cloud = await asyncio.gather(_check_groq(), _check_gemini(), _check_ollama_cloud())

    chain: List[Dict[str, str]] = []
    for m in groq.verified_models:
        chain.append({"provider": "groq", "model": m})
    for m in gemini.verified_models:
        chain.append({"provider": "gemini", "model": m})
    for m in ollama_cloud.verified_models:
        chain.append({"provider": "ollama_cloud", "model": m})

    report = VerificationReport(groq=groq, gemini=gemini, ollama_cloud=ollama_cloud, active_judge_chain=chain)
    if not chain:
        logger.error("Pre-flight verification found NO usable providers/models. Check API keys and `ollama signin`.")
    else:
        logger.info("Pre-flight verification succeeded. active_judge_chain=%s", chain)
    return report


_cached_report: Optional[VerificationReport] = None


async def get_cached_or_refresh_report(force_refresh: bool = False) -> VerificationReport:
    global _cached_report
    if _cached_report is None or force_refresh:
        _cached_report = await run_preflight_verification()
    return _cached_report
