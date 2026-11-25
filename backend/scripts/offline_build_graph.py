"""
scripts/offline_build_graph.py

STANDALONE CLI SCRIPT (not an API endpoint). Run this yourself, offline, to
build the LightRAG NetworkX knowledge graph from an existing ChromaDB
collection.

By DEFAULT this hits a genuinely LOCAL Ollama model (no `:cloud` suffix, no
API key, no Groq/Gemini involved at all) — e.g. `ollama pull qwen2.5:3b`
first, then just run this script. Pass --allow-cloud-fallback if you
specifically want the Groq -> Gemini -> Ollama(:cloud) waterfall instead.

Usage:
    # Local Ollama (default) — uses OLLAMA_LOCAL_MODEL from .env (qwen2.5:3b)
    python scripts/offline_build_graph.py --collection machine-learning \
        --output ./lightrag_graphs/machine-learning_graph.graphml

    # Pick a different local model / Ollama host explicitly
    python scripts/offline_build_graph.py --collection machine-learning \
        --output ./lightrag_graphs/machine-learning_graph.graphml \
        --ollama-model qwen2.5:7b --ollama-base-url http://localhost:11434

    # Opt into the Groq -> Gemini -> Ollama(:cloud) fallback chain instead
    python scripts/offline_build_graph.py --collection machine-learning \
        --output ./lightrag_graphs/machine-learning_graph.graphml \
        --allow-cloud-fallback

This script intentionally lives outside app/ and is never imported by the
FastAPI app — the backend only ever READS the .graphml file this produces.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
import networkx as nx  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.core.resilient_llm import (  # noqa: E402
    ResilientChatRouter,
    build_candidate_chain,
    build_local_ollama_router,
)
from app.core.vector_store import get_vector_store_manager  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("offline_build_graph")

EXTRACTION_PROMPT = """Extract entities and relationships from the following text chunk for a
knowledge graph. Return ONLY valid JSON of the form:
{{
  "entities": [{{"name": "...", "type": "...", "description": "..."}}],
  "relationships": [{{"source": "...", "target": "...", "relation": "...", "description": "..."}}]
}}

Text chunk:
{chunk}
"""


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"entities": [], "relationships": []}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"entities": [], "relationships": []}

def _checkpoint_path_for(output_path: str, explicit: str | None) -> str:
    return explicit or f"{output_path}.checkpoint.jsonl"


def _load_checkpoint(path: str) -> dict[int, dict]:
    """Returns {chunk_index: {"entities": [...], "relationships": [...]}} for
    every chunk that completed successfully in a prior run."""
    done: dict[int, dict] = {}
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done[row["chunk_index"]] = row
            except (json.JSONDecodeError, KeyError):
                logger.warning("Skipping malformed checkpoint line %d in %s", line_no, path)
    return done


def _append_checkpoint(path: str, chunk_index: int, entities: list, relationships: list) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"chunk_index": chunk_index, "entities": entities, "relationships": relationships}) + "\n")
        f.flush()
        os.fsync(f.fileno())  # survive a hard crash, not just a clean exit


def _apply_payload_to_graph(graph: "nx.Graph", payload: dict) -> None:
    for entity in payload.get("entities", []):
        name = entity.get("name")
        if not name:
            continue
        graph.add_node(name, entity_type=entity.get("type", ""), description=entity.get("description", ""))

    for rel in payload.get("relationships", []):
        src, tgt = rel.get("source"), rel.get("target")
        if not src or not tgt:
            continue
        graph.add_edge(src, tgt, relation=rel.get("relation", "related_to"), description=rel.get("description", ""))


def _write_graph(graph: "nx.Graph", output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    nx.write_graphml(graph, output_path)

def _check_local_ollama(base_url: str, model: str) -> None:
    """Fails fast with an actionable message instead of burning through every
    chunk only to error 103 times in a row (as happens when the fallback
    chain silently cascades through dead/rate-limited cloud providers)."""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5.0)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not reach Ollama at {base_url} ({exc}). Is the Ollama daemon running? "
            f"Start it with `ollama serve` (or just open the Ollama app)."
        ) from exc

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama at {base_url} returned HTTP {resp.status_code} for /api/tags.")

    available = {m["name"] for m in resp.json().get("models", [])}
    if model not in available:
        raise RuntimeError(
            f"Model '{model}' is not pulled locally. Available models: {sorted(available) or '(none)'}. "
            f"Run `ollama pull {model}` first, or pass --ollama-model <name> to use one you already have."
        )
    logger.info("Local Ollama check OK: %s is reachable and '%s' is pulled.", base_url, model)

async def build_graph(
    collection_name: str,
    output_path: str,
    limit: int | None,
    allow_cloud_fallback: bool = False,
    ollama_model: str | None = None,
    ollama_base_url: str | None = None,
    checkpoint_path: str | None = None,
    fresh: bool = False,
    checkpoint_every: int = 5,
    max_consecutive_failures: int = 5,
) -> None:
    settings = get_settings()
    manager = get_vector_store_manager()
    handle = manager.get_collection(collection_name)

    if allow_cloud_fallback:
        logger.info("Using Groq -> Gemini -> Ollama(:cloud) fallback chain (--allow-cloud-fallback was set).")
        chain = build_candidate_chain(role="reasoning")
        ordered = [c for c in chain if c.provider == "ollama_cloud"] + [c for c in chain if c.provider != "ollama_cloud"]
        router = ResilientChatRouter(role="reasoning", candidates=ordered)
    else:
        model = ollama_model or settings.OLLAMA_LOCAL_MODEL
        base_url = ollama_base_url or settings.OLLAMA_BASE_URL
        _check_local_ollama(base_url, model)
        logger.info("Using LOCAL Ollama only: %s @ %s (no Groq/Gemini/cloud involved)", model, base_url)
        router = build_local_ollama_router(models=[model], base_url=base_url)

    ckpt_path = _checkpoint_path_for(output_path, checkpoint_path)
    if fresh and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        logger.info("Removed existing checkpoint (--fresh): %s", ckpt_path)

    done = _load_checkpoint(ckpt_path)
    graph = nx.Graph()
    for row in done.values():
        _apply_payload_to_graph(graph, row)
    if done:
        logger.info("Resuming from checkpoint: %d chunk(s) already extracted, loaded from %s. "
                     "Skipping straight to the first unprocessed chunk.", len(done), ckpt_path)

    docs = handle.raw_docs[:limit] if limit else handle.raw_docs
    total = len(docs)
    logger.info("%d/%d chunks remaining to process.", total - len(done), total)

    consecutive_failures = 0

    try:
        for i, chunk_text in enumerate(docs):
            if i in done:
                continue
            logger.info("Processing chunk %d/%d", i + 1, total)
            try:
                response = await router.ainvoke(EXTRACTION_PROMPT.format(chunk=chunk_text[:2000]))
                payload = _extract_json(getattr(response, "content", str(response)))
            except Exception as exc:  # noqa: BLE001
                consecutive_failures = 1
                logger.warning("Extraction failed for chunk %d (%d consecutive failure(s)): %s", i, consecutive_failures, exc)
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(
                        "%d consecutive failures — aborting early rather than burning through the rest of the "
                        "collection against what looks like a dead/unstable model. Progress IS saved: %d chunks "
                        "in %s and a graph snapshot at %s. Fix the issue (restart Ollama, check VRAM, try a "
                        "smaller model) and re-run the exact same command — it will resume from chunk %d.",
                        consecutive_failures, len(done), ckpt_path, output_path, i,
                    )
                    _write_graph(graph, output_path)
                    return
                continue

            consecutive_failures = 0
            entities = payload.get("entities", [])
            relationships = payload.get("relationships", [])
            _apply_payload_to_graph(graph, payload)
            _append_checkpoint(ckpt_path, i, entities, relationships)
            done[i] = {"chunk_index": i, "entities": entities, "relationships": relationships}

            if (i + 1) % checkpoint_every == 0:
                _write_graph(graph, output_path)
                logger.info("Checkpoint snapshot: %d nodes / %d edges written to %s so far.",
                            graph.number_of_nodes(), graph.number_of_edges(), output_path)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user — progress for every completed chunk is already in %s.", ckpt_path)
        _write_graph(graph, output_path)
        logger.info("Saved partial graph (%d nodes / %d edges) to %s. Re-run the same command to resume.", ...)
        return

    _write_graph(graph, output_path)
    logger.info(
    "Done. Wrote graph with %d nodes / %d edges to %s",
    graph.number_of_nodes(),
    graph.number_of_edges(),
    output_path,
)
    logger.info("Checkpoint at %s is safe to delete now, or keep it if you plan to re-run with --fresh later.", ckpt_path)


def main():
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Offline LightRAG graph builder")
    parser.add_argument("--collection", required=True, help="ChromaDB collection name to build the graph from")
    parser.add_argument("--output", required=True, help="Output .graphml path")
    parser.add_argument("--limit", type=int, default=None, help="Optional: limit number of chunks processed")
    parser.add_argument(
        "--ollama-model", default=None,
        help=f"Local Ollama model to use (default: OLLAMA_LOCAL_MODEL={settings.OLLAMA_LOCAL_MODEL!r} from .env). "
             f"Must already be pulled: `ollama pull <model>`.",
    )
    parser.add_argument(
        "--ollama-base-url", default=None,
        help=f"Local Ollama daemon URL (default: OLLAMA_BASE_URL={settings.OLLAMA_BASE_URL!r} from .env).",
    )
    parser.add_argument(
        "--allow-cloud-fallback", action="store_true",
        help="Use the Groq -> Gemini -> Ollama(:cloud) chain instead of a local-only model.",
    )
    parser.add_argument("--checkpoint-path", default=None,
        help="Where to write the per-chunk resume checkpoint (default: <output>.checkpoint.jsonl). "
             "If this file exists when the script starts, already-completed chunks are skipped automatically.")
    parser.add_argument("--fresh", action="store_true",
        help="Ignore/delete any existing checkpoint and start this run from chunk 0.")
    parser.add_argument("--checkpoint-every", type=int, default=5,
        help="Write a .graphml snapshot to --output every N successfully-processed chunks (default: 5).")
    parser.add_argument("--max-consecutive-failures", type=int, default=5,
        help="Abort early (saving progress first) after this many extraction failures in a row.")
    args = parser.parse_args()
    asyncio.run(build_graph(
        args.collection,
        args.output,
        args.limit,
        allow_cloud_fallback=args.allow_cloud_fallback,
        ollama_model=args.ollama_model,
        ollama_base_url=args.ollama_base_url,
        checkpoint_path=args.checkpoint_path,
        fresh=args.fresh,
        checkpoint_every=args.checkpoint_every,
        max_consecutive_failures=args.max_consecutive_failures,
    ))

if __name__ == "__main__":
    main()
