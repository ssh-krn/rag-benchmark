// src/components/SourceInspector.tsx
import { useMemo, useRef, useState } from "react";
import {
  ChevronsRight,
  FileText,
  Globe,
  Share2,
  Copy,
  ExternalLink,
  Search,
  GripVertical,
} from "lucide-react";
import type { ChatMessage, SourceMetadata } from "@/types/rag";
import { LightRagGraphView } from "./LightRagGraphView";

interface SourceInspectorProps {
  message: ChatMessage | null;
  open: boolean;
  onToggle: () => void;
  focusedRef: string | null;
}

function ScoreBar({ value, max = 1 }: { value: number | null | undefined; max?: number }) {
  if (value == null) return null;
  const pct = Math.max(0, Math.min(1, value / max)) * 100;
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-[color:var(--color-card)]">
      <div className="h-full bg-[color:var(--color-signal)]/70" style={{ width: `${pct}%` }} />
    </div>
  );
}

function Badge({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <span className={`rounded border border-[color:var(--color-border)] bg-[color:var(--color-card)] px-1.5 py-0.5 font-mono text-[10px] tabular ${color ?? "text-[color:var(--color-ink-dim)]"}`}>
      {label}={value}
    </span>
  );
}

function ChunkCard({ source, index, isFocused }: { source: SourceMetadata; index: number; isFocused: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const copyText = async () => {
    if (source.text) await navigator.clipboard.writeText(source.text);
  };

  return (
    <div
      ref={ref}
      id={`cite-passage-${index + 1}`}
      className={`rounded-lg border p-2.5 transition ${
        isFocused ? "border-[color:var(--color-signal)]/60 bg-[color:var(--color-signal)]/[0.06]" : "border-[color:var(--color-border)] bg-[color:var(--color-panel)]"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-1.5 text-xs">
          <FileText size={12} className="text-[color:var(--color-ink-faint)]" />
          <span className="font-medium">Passage {index + 1}</span>
          {source.page != null && <span className="text-[color:var(--color-ink-faint)]">· p.{source.page}</span>}
        </div>
        <button onClick={copyText} title="Copy chunk text" className="rounded p-1 text-[color:var(--color-ink-faint)] hover:bg-[color:var(--color-card-hover)] hover:text-[color:var(--color-ink)]">
          <Copy size={12} />
        </button>
      </div>
      <p className="mt-1 truncate text-[11px] text-[color:var(--color-ink-faint)]">{source.source ?? "unknown source"}</p>

      <div className="mt-1.5 flex flex-wrap gap-1">
        {source.collection_name && <Badge label="collection" value={source.collection_name} />}
        {source.chunk_id && <Badge label="chunk_id" value={source.chunk_id} />}
        {source.retrieval_method && <Badge label="method" value={source.retrieval_method} />}
        {source.dense_rank != null && <Badge label="dense_rank" value={source.dense_rank} />}
        {source.bm25_rank != null && <Badge label="bm25_rank" value={source.bm25_rank} />}
        {source.rrf_score != null && <Badge label="rrf" value={source.rrf_score.toFixed(3)} color="text-[color:var(--color-violet)]" />}
        {source.rerank_score != null && <Badge label="rerank" value={source.rerank_score.toFixed(3)} color="text-[color:var(--color-signal)]" />}
      </div>

      {source.score != null && (
        <div className="mt-1.5">
          <ScoreBar value={source.score} />
        </div>
      )}

      <button onClick={() => setExpanded((e) => !e)} className="mt-1.5 text-[11px] text-[color:var(--color-signal)] hover:underline">
        {expanded ? "Collapse" : "Expand full text"}
      </button>
      {expanded && <p className="mt-1.5 whitespace-pre-wrap text-[11.5px] leading-relaxed text-[color:var(--color-ink-dim)]">{source.text}</p>}
    </div>
  );
}

function WebCard({ source, index, isFocused }: { source: SourceMetadata; index: number; isFocused: boolean }) {
  let domain = "";
  try {
    domain = source.url ? new URL(source.url).hostname.replace("www.", "") : "";
  } catch {
    domain = source.url ?? "";
  }
  return (
    <div
      id={`cite-web-${index + 1}`}
      className={`rounded-lg border p-2.5 transition ${
        isFocused ? "border-[color:var(--color-signal)]/60 bg-[color:var(--color-signal)]/[0.06]" : "border-[color:var(--color-border)] bg-[color:var(--color-panel)]"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 text-xs">
          <Globe size={12} className="text-[color:var(--color-ink-faint)]" />
          <span className="font-medium">Web Source {index + 1}</span>
        </div>
        {source.url && (
          <a href={source.url} target="_blank" rel="noreferrer" className="rounded p-1 text-[color:var(--color-ink-faint)] hover:bg-[color:var(--color-card-hover)] hover:text-[color:var(--color-signal)]">
            <ExternalLink size={12} />
          </a>
        )}
      </div>
      <p className="mt-1 text-[11.5px] font-medium text-[color:var(--color-ink)]">{source.source}</p>
      <p className="text-[10.5px] text-[color:var(--color-ink-faint)]">{domain}</p>
      {source.score != null && (
        <div className="mt-1.5 flex items-center gap-2">
          <ScoreBar value={source.score} />
          <span className="tabular text-[10px] text-[color:var(--color-ink-faint)]">{source.score.toFixed(2)}</span>
        </div>
      )}
      <p className="mt-1.5 line-clamp-4 text-[11.5px] leading-relaxed text-[color:var(--color-ink-dim)]">{source.text}</p>
    </div>
  );
}

export function SourceInspector({ message, open, onToggle, focusedRef }: SourceInspectorProps) {
  const [width, setWidth] = useState(360);
  const [entitySearch, setEntitySearch] = useState("");
  const draggingRef = useRef(false);

  const handleDragStart = () => {
    draggingRef.current = true;
    const onMove = (e: MouseEvent) => {
      if (!draggingRef.current) return;
      const next = Math.min(560, Math.max(280, window.innerWidth - e.clientX));
      setWidth(next);
    };
    const onUp = () => {
      draggingRef.current = false;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  const chunkSources = useMemo(
    () => (message?.retrievedContext ?? []).filter((s) => s.retrieval_method !== "web_search" && s.retrieval_method !== "graph_traversal"),
    [message],
  );
  const webSources = useMemo(() => (message?.retrievedContext ?? []).filter((s) => s.retrieval_method === "web_search"), [message]);
  const isLightRag = message?.mode === "lightrag";

  const filteredEntities = useMemo(() => {
    const all = message?.matchedEntities ?? [];
    if (!entitySearch) return all;
    return all.filter((e) => e.toLowerCase().includes(entitySearch.toLowerCase()));
  }, [message, entitySearch]);

  if (!open) {
    return (
      <button
        onClick={onToggle}
        title="Open metadata inspector (Cmd/Ctrl+I)"
        className="fixed right-0 top-1/2 z-20 -translate-y-1/2 rounded-l-lg border border-r-0 border-[color:var(--color-border)] bg-[color:var(--color-panel)] p-2 text-[color:var(--color-ink-faint)] hover:text-[color:var(--color-ink)]"
      >
        <ChevronsRight size={16} className="rotate-180" />
      </button>
    );
  }

  return (
    <aside style={{ width }} className="relative flex h-full shrink-0 flex-col border-l border-[color:var(--color-border)] bg-[color:var(--color-panel)]">
      <div
        onMouseDown={handleDragStart}
        className="absolute left-0 top-0 z-10 flex h-full w-1.5 -translate-x-1/2 cursor-col-resize items-center justify-center"
      >
        <GripVertical size={12} className="text-[color:var(--color-ink-faint)] opacity-0 hover:opacity-100" />
      </div>

      <div className="flex items-center justify-between border-b border-[color:var(--color-border)] px-3 py-2.5">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-[color:var(--color-ink-faint)]">Retrieved Context &amp; Metadata</h3>
        <button onClick={onToggle} className="rounded p-1 text-[color:var(--color-ink-faint)] hover:bg-[color:var(--color-card-hover)] hover:text-[color:var(--color-ink)]">
          <ChevronsRight size={14} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        {!message ? (
          <p className="text-xs text-[color:var(--color-ink-faint)]">Send a message to populate retrieved context.</p>
        ) : (
          <div className="space-y-4">
            {isLightRag && (
              <>
                <section>
                  <h4 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-[color:var(--color-ink-faint)]">
                    Dual-level query breakdown
                  </h4>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="rounded-lg border border-[color:var(--color-border)] bg-[color:var(--color-canvas)]/60 p-2">
                      <p className="mb-1 text-[10px] font-medium text-[color:var(--color-ink-faint)]">Low-level entities</p>
                      <div className="flex flex-wrap gap-1">
                        {(message.matchedEntities ?? []).slice(0, 8).map((e) => (
                          <span key={e} className="rounded-full bg-[color:var(--color-card)] px-1.5 py-0.5 text-[10px]">{e}</span>
                        ))}
                      </div>
                    </div>
                    <div className="rounded-lg border border-[color:var(--color-border)] bg-[color:var(--color-canvas)]/60 p-2">
                      <p className="mb-1 text-[10px] font-medium text-[color:var(--color-ink-faint)]">High-level themes</p>
                      <div className="flex flex-wrap gap-1">
                        {(message.subgraphNodes ?? []).slice(0, 8).map((n) => (
                          <span key={n.id} className="rounded-full bg-[color:var(--color-card)] px-1.5 py-0.5 text-[10px]">{String(n.entity_type || n.id)}</span>
                        ))}
                      </div>
                    </div>
                  </div>
                </section>

                <section>
                  <h4 className="mb-1.5 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-[color:var(--color-ink-faint)]">
                    <Share2 size={12} /> Subgraph visualizer
                  </h4>
                  <LightRagGraphView
                    nodes={message.subgraphNodes ?? []}
                    edges={message.subgraphEdges ?? []}
                    matchedEntities={message.matchedEntities}
                  />
                </section>

                <section>
                  <div className="mb-1.5 flex items-center justify-between">
                    <h4 className="text-[11px] font-semibold uppercase tracking-wide text-[color:var(--color-ink-faint)]">Matched entities</h4>
                  </div>
                  <div className="mb-2 flex items-center gap-1.5 rounded-md border border-[color:var(--color-border)] bg-[color:var(--color-canvas)]/60 px-2 py-1">
                    <Search size={12} className="text-[color:var(--color-ink-faint)]" />
                    <input
                      value={entitySearch}
                      onChange={(e) => setEntitySearch(e.target.value)}
                      placeholder="Search entities…"
                      className="w-full bg-transparent text-[11px] outline-none placeholder:text-[color:var(--color-ink-faint)]"
                    />
                  </div>
                  <div className="space-y-1">
                    {filteredEntities.map((e) => (
                      <div key={e} className="rounded-md border border-[color:var(--color-border)] bg-[color:var(--color-panel)] px-2 py-1 text-[11px]">
                        {e}
                      </div>
                    ))}
                  </div>
                </section>

                <section>
                  <h4 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-[color:var(--color-ink-faint)]">Relationships</h4>
                  <div className="space-y-1">
                    {(message.relationships ?? []).slice(0, 25).map((r, i) => (
                      <div key={i} className="rounded-md border border-[color:var(--color-border)] bg-[color:var(--color-panel)] px-2 py-1 font-mono text-[10.5px] text-[color:var(--color-ink-dim)]">
                        {r.source} <span className="text-[color:var(--color-violet)]">--[{r.relation}]--&gt;</span> {r.target}
                      </div>
                    ))}
                  </div>
                </section>
              </>
            )}

            {chunkSources.length > 0 && (
              <section>
                <h4 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-[color:var(--color-ink-faint)]">ChromaDB chunks</h4>
                <div className="space-y-2">
                  {chunkSources.map((s, i) => (
                    <ChunkCard key={s.chunk_id ?? i} source={s} index={i} isFocused={focusedRef === `passage-${i + 1}`} />
                  ))}
                </div>
              </section>
            )}

            {webSources.length > 0 && (
              <section>
                <h4 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-[color:var(--color-ink-faint)]">Tavily web sources</h4>
                <div className="space-y-2">
                  {webSources.map((s, i) => (
                    <WebCard key={s.url ?? i} source={s} index={i} isFocused={focusedRef === `web-${i + 1}`} />
                  ))}
                </div>
              </section>
            )}

            {chunkSources.length === 0 && webSources.length === 0 && !isLightRag && (
              <p className="text-xs text-[color:var(--color-ink-faint)]">No retrieved context for this message yet.</p>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}
