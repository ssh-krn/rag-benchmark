// src/components/LightRagGraphView.tsx
import { useMemo, useState } from "react";
import type { GraphEdge, GraphNode } from "@/types/rag";

interface LightRagGraphViewProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  matchedEntities?: string[];
  height?: number;
}

interface LaidOutNode extends GraphNode {
  x: number;
  y: number;
}

/** Deterministic circular layout — no physics engine dependency needed for
 * subgraphs of this size (typically a few dozen nodes). */
function layoutNodes(nodes: GraphNode[], width: number, height: number): LaidOutNode[] {
  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.min(width, height) / 2 - 36;
  return nodes.map((n, i) => {
    const angle = (2 * Math.PI * i) / Math.max(nodes.length, 1);
    return { ...n, x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
  });
}

export function LightRagGraphView({ nodes, edges, matchedEntities = [], height = 340 }: LightRagGraphViewProps) {
  const width = 560;
  const [hovered, setHovered] = useState<string | null>(null);
  const [selected, setSelected] = useState<GraphNode | null>(null);

  const laidOut = useMemo(() => layoutNodes(nodes, width, height), [nodes, height]);
  const posById = useMemo(() => new Map(laidOut.map((n) => [n.id, n])), [laidOut]);
  const matchedSet = useMemo(() => new Set(matchedEntities), [matchedEntities]);

  if (nodes.length === 0) {
    return <p className="p-3 text-xs text-[color:var(--color-ink-faint)]">No subgraph data for this query.</p>;
  }

  return (
    <div className="space-y-2">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full rounded-lg border border-[color:var(--color-border)] bg-[color:var(--color-canvas)]">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="rgba(255,255,255,0.25)" />
          </marker>
        </defs>

        {edges.map((e, i) => {
          const s = posById.get(e.source);
          const t = posById.get(e.target);
          if (!s || !t) return null;
          const active = hovered === e.source || hovered === e.target;
          return (
            <g key={i}>
              <line
                x1={s.x}
                y1={s.y}
                x2={t.x}
                y2={t.y}
                stroke={active ? "rgba(94, 234, 212, 0.55)" : "rgba(255,255,255,0.14)"}
                strokeWidth={active ? 1.6 : 1}
                markerEnd="url(#arrow)"
              />
              {active && (
                <text
                  x={(s.x + t.x) / 2}
                  y={(s.y + t.y) / 2 - 4}
                  fill="var(--color-signal)"
                  fontSize="9"
                  textAnchor="middle"
                  className="font-mono"
                >
                  {e.relation}
                </text>
              )}
            </g>
          );
        })}

        {laidOut.map((n) => {
          const isMatched = matchedSet.has(n.id);
          const isHovered = hovered === n.id;
          return (
            <g
              key={n.id}
              transform={`translate(${n.x}, ${n.y})`}
              onMouseEnter={() => setHovered(n.id)}
              onMouseLeave={() => setHovered((h) => (h === n.id ? null : h))}
              onClick={() => setSelected(n)}
              className="cursor-pointer"
            >
              <circle
                r={isMatched ? 7 : 5}
                fill={isMatched ? "var(--color-signal)" : "#3f3f46"}
                stroke={isHovered ? "var(--color-signal)" : "rgba(255,255,255,0.2)"}
                strokeWidth={isHovered ? 2 : 1}
              />
              <text x={0} y={-10} textAnchor="middle" fontSize="9" fill="#c4c4cc" className="select-none">
                {n.id.length > 18 ? n.id.slice(0, 16) + "…" : n.id}
              </text>
            </g>
          );
        })}
      </svg>

      {selected && (
        <div className="rounded-lg border border-[color:var(--color-border)] bg-[color:var(--color-panel)] p-2.5 text-xs">
          <p className="font-semibold text-[color:var(--color-ink)]">{selected.id}</p>
          {selected.entity_type ? <p className="mt-0.5 text-[color:var(--color-ink-faint)]">type: {String(selected.entity_type)}</p> : null}
          {selected.description ? <p className="mt-1 text-[color:var(--color-ink-dim)]">{String(selected.description)}</p> : null}
        </div>
      )}
    </div>
  );
}
