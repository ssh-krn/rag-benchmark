"""
app/core/graph_store.py

Read-only loader for pre-built NetworkX LightRAG knowledge graphs
(.graphml files produced offline by scripts/offline_build_graph.py), plus
1-hop/2-hop traversal utilities and ChromaDB entity/relation-store lookups.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import get_settings

logger = logging.getLogger("graph_store")


class LightRAGGraphHandle:
    def __init__(self, name: str, graph):
        self.name = name
        self.graph = graph

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def find_matching_entities(self, keywords: List[str], limit: int = 10) -> List[str]:
        keywords_lower = [k.lower() for k in keywords]
        matches = []
        for node, data in self.graph.nodes(data=True):
            label = str(node).lower()
            entity_type = str(data.get("entity_type", "")).lower()
            description = str(data.get("description", "")).lower()
            if any(kw in label or kw in entity_type or kw in description for kw in keywords_lower):
                matches.append(node)
            if len(matches) >= limit:
                break
        return matches

    def traverse(self, seed_nodes: List[str], hops: int = 2) -> Tuple[Set[str], List[Dict[str, Any]]]:
        """Return (subgraph_nodes, subgraph_edges) reachable within `hops` of any seed node."""
        visited: Set[str] = set(n for n in seed_nodes if n in self.graph)
        frontier = set(visited)
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(self.graph.neighbors(node))
            next_frontier -= visited
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

        edges = []
        for u, v, data in self.graph.edges(data=True):
            if u in visited and v in visited:
                edges.append({
                    "source": u,
                    "target": v,
                    "relation": data.get("relation") or data.get("description"),
                    "weight": data.get("weight"),
                })
        return visited, edges

    def node_payload(self, node: str) -> Dict[str, Any]:
        data = dict(self.graph.nodes[node]) if node in self.graph else {}
        return {"id": node, **data}


class GraphStoreManager:
    def __init__(self):
        self.settings = get_settings()
        self._handles: Dict[str, LightRAGGraphHandle] = {}

    def _load(self, name: str) -> LightRAGGraphHandle:
        import networkx as nx

        path = os.path.join(self.settings.GRAPH_DIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"LightRAG graph file not found at {path}. Build it offline first "
                                     f"with scripts/offline_build_graph.py")
        graph = nx.read_graphml(path)
        handle = LightRAGGraphHandle(name=name, graph=graph)
        logger.info("Loaded LightRAG graph '%s': %d nodes, %d edges", name, handle.node_count, handle.edge_count)
        return handle

    def get_graph(self, name: Optional[str] = None) -> LightRAGGraphHandle:
        name = name or self.settings.DEFAULT_GRAPH_NAME
        if name not in self._handles:
            self._handles[name] = self._load(name)
        return self._handles[name]

    def list_graphs(self) -> List[Dict[str, Any]]:
        out = []
        if not os.path.isdir(self.settings.GRAPH_DIR):
            return out
        for fname in os.listdir(self.settings.GRAPH_DIR):
            if fname.endswith(".graphml"):
                try:
                    handle = self.get_graph(fname)
                    out.append({"name": fname, "nodes": handle.node_count, "edges": handle.edge_count})
                except Exception as exc:  # noqa: BLE001
                    out.append({"name": fname, "error": str(exc)})
        return out

    def health(self) -> Dict[str, Any]:
        return {
            "graph_dir": self.settings.GRAPH_DIR,
            "available_graphs": self.list_graphs(),
        }


@lru_cache
def get_graph_store_manager() -> GraphStoreManager:
    return GraphStoreManager()
