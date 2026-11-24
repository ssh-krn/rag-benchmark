"""
scripts/offline_build_graph.py

STANDALONE CLI SCRIPT (not an API endpoint). Run this yourself, offline, to
build the LightRAG NetworkX knowledge graph from an existing ChromaDB
collection using an Ollama :cloud model for entity/relationship extraction.

Usage:
    python scripts/offline_build_graph.py --collection ml-foundation \
        --output ./lightrag_graphs/ml-foundation_graph.graphml

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

import networkx as nx  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.core.resilient_llm import ResilientChatRouter, build_candidate_chain  # noqa: E402
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


async def build_graph(collection_name: str, output_path: str, limit: int | None) -> None:
    settings = get_settings()
    manager = get_vector_store_manager()
    handle = manager.get_collection(collection_name)

    chain = build_candidate_chain(role="reasoning")
    ollama_first = [c for c in chain if c.provider == "ollama_cloud"] + [c for c in chain if c.provider != "ollama_cloud"]
    router = ResilientChatRouter(role="reasoning", candidates=ollama_first)

    graph = nx.Graph()
    docs = handle.raw_docs[:limit] if limit else handle.raw_docs

    for i, chunk_text in enumerate(docs):
        logger.info("Processing chunk %d/%d", i + 1, len(docs))
        try:
            response = await router.ainvoke(EXTRACTION_PROMPT.format(chunk=chunk_text[:2000]))
            payload = _extract_json(getattr(response, "content", str(response)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Extraction failed for chunk %d: %s", i, exc)
            continue

        for entity in payload.get("entities", []):
            name = entity.get("name")
            if not name:
                continue
            graph.add_node(name, entity_type=entity.get("type", ""), description=entity.get("description", ""))

        for rel in payload.get("relationships", []):
            src, tgt = rel.get("source"), rel.get("target")
            if not src or not tgt:
                continue
            graph.add_edge(src, tgt, relation=rel.get("relation", "related_to"),
                            description=rel.get("description", ""))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    nx.write_graphml(graph, output_path)
    logger.info("Wrote graph with %d nodes / %d edges to %s", graph.number_of_nodes(), graph.number_of_edges(), output_path)


def main():
    parser = argparse.ArgumentParser(description="Offline LightRAG graph builder")
    parser.add_argument("--collection", required=True, help="ChromaDB collection name to build the graph from")
    parser.add_argument("--output", required=True, help="Output .graphml path")
    parser.add_argument("--limit", type=int, default=None, help="Optional: limit number of chunks processed")
    args = parser.parse_args()
    asyncio.run(build_graph(args.collection, args.output, args.limit))


if __name__ == "__main__":
    main()
