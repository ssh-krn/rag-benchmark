"""
app/main.py
FastAPI application entrypoint. Runs the pre-flight free-tier model
verification once at startup (lifespan), wires CORS for the React frontend,
and mounts all API routes.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.core.model_validator import run_preflight_verification

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Running pre-flight free-tier & Ollama :cloud model verification...")
    report = await run_preflight_verification()
    app.state.preflight_report = report
    if not report.active_judge_chain:
        logger.warning(
            "No verified LLM providers available at startup. /api/chat and /api/evals/run "
            "will fail until at least one of GROQ_API_KEY, GEMINI_API_KEY, or `ollama signin` "
            "/ OLLAMA_API_KEY is configured correctly."
        )
    yield
    logger.info("Shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Advanced Multi-Strategy RAG Platform",
        description="Backend for 5 selectable RAG modes with an integrated, rate-limit-resilient "
                     "DeepEval evaluation suite.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
