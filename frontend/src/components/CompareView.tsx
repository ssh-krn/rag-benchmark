// src/components/CompareView.tsx
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { Loader2, Send, Zap, Clock, Boxes, GitCompareArrows } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { ChatResponse, CollectionInfo, GraphInfo, RagMode } from "@/types/rag";
import { RAG_MODES, RAG_MODE_LABELS } from "@/types/rag";
import { ExecutionTraceBar } from "./ExecutionTraceBar";

interface CompareViewProps {
  collections: CollectionInfo[];
  graphs: GraphInfo[];
}

interface Column {
  mode: RagMode;
  loading: boolean;
  totalMs: number | null;
  response: ChatResponse | null;
  error: string | null;
}

function emptyColumn(mode: RagMode): Column {
  return { mode, loading: false, totalMs: null, response: null, error: null };
}

function ColumnPanel({ column, collectionName, graphName, onModeChange }: {
  column: Column;
  collectionName?: string;
  graphName?: string;
  onModeChange: (m: RagMode) => void;
}) {
  const { mode, loading, totalMs, response, error } = column;
  return (
    <div className="flex-1 space-y-3 rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-card)] p-4">
      <div className="flex items-center justify-between">
        <select
          value={mode}
          onChange={(e) => onModeChange(e.target.value as RagMode)}
          className="rounded-lg border border-[color:var(--color-border)] bg-[color:var(--color-panel)] px-2.5 py-1.5 text-xs font-medium outline-none"
        >
          {RAG_MODES.map((m) => (
            <option key={m} value={m}>
              {RAG_MODE_LABELS[m]}
            </option>
          ))}
        </select>
        {loading && <Loader2 size={14} className="animate-spin text-[color:var(--color-signal)]" />}
      </div>

      {error && (
        <div className="rounded-md border border-[color:var(--color-coral)]/30 bg-[color:var(--color-coral)]/[0.08] px-2.5 py-1.5 text-xs text-[color:var(--color-coral)]">
          {error}
        </div>
      )}

      {response && (
        <>
          <div className="flex flex-wrap items-center gap-3 text-[11px] text-[color:var(--color-ink-dim)]">
            <span className="flex items-center gap-1">
              <Clock size={12} /> <span className="tabular">{totalMs}ms total</span>
            </span>
            <span className="flex items-center gap-1">
              <Boxes size={12} /> <span className="tabular">{response.retrieved_context.length} sources</span>
            </span>
            {response.retrieved_context[0]?.score != null && (
              <span className="flex items-center gap-1">
                <Zap size={12} /> <span className="tabular">top score {response.retrieved_context[0].score?.toFixed(3)}</span>
              </span>
            )}
            <span className="ml-auto rounded-full border border-[color:var(--color-border)] px-2 py-0.5 font-mono tabular">
              {response.provider_used ?? "?"} / {response.model_used ?? "?"}
            </span>
          </div>

          <ExecutionTraceBar steps={response.execution_trace} toolCalls={[]} grades={[]} />

          <div className="msg-prose max-h-80 overflow-y-auto rounded-lg border border-[color:var(--color-border)] bg-[color:var(--color-panel)] p-3 text-[13px] leading-relaxed">
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
              {response.answer}
            </ReactMarkdown>
          </div>
        </>
      )}

      {!response && !loading && !error && (
        <p className="text-xs text-[color:var(--color-ink-faint)]">Run a prompt to see this strategy's answer.</p>
      )}
      <p className="hidden text-xs">{collectionName}{graphName}</p>
    </div>
  );
}

export function CompareView({ collections, graphs }: CompareViewProps) {
  const [prompt, setPrompt] = useState("");
  const [collectionName, setCollectionName] = useState<string | undefined>(collections[0]?.name);
  const [graphName, setGraphName] = useState<string | undefined>(graphs[0]?.name);
  const [left, setLeft] = useState<Column>(emptyColumn("traditional_rag"));
  const [right, setRight] = useState<Column>(emptyColumn("lightrag"));

  const runBoth = async () => {
    const q = prompt.trim();
    if (!q) return;

    setLeft((c) => ({ ...c, loading: true, error: null }));
    setRight((c) => ({ ...c, loading: true, error: null }));

    const run = async (col: Column, setter: typeof setLeft) => {
      const start = performance.now();
      try {
        const res = await api.chat({
          query: q,
          mode: col.mode,
          collection_name: collectionName,
          graph_name: graphName,
        });
        setter({ mode: col.mode, loading: false, totalMs: Math.round(performance.now() - start), response: res, error: null });
      } catch (err) {
        setter({
          mode: col.mode,
          loading: false,
          totalMs: null,
          response: null,
          error: err instanceof ApiError ? err.message : "Request failed",
        });
      }
    };

    await Promise.all([run(left, setLeft), run(right, setRight)]);
  };

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-5">
      <div className="flex items-center gap-2">
        <GitCompareArrows size={18} className="text-[color:var(--color-signal)]" />
        <h2 className="text-sm font-semibold">Side-by-Side Mode Comparator</h2>
      </div>

      <div className="flex items-center gap-2 rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-panel)] p-2">
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && runBoth()}
          placeholder="Run the same prompt against two strategies at once…"
          className="flex-1 bg-transparent px-2 py-1.5 text-sm outline-none placeholder:text-[color:var(--color-ink-faint)]"
        />
        <select
          value={collectionName}
          onChange={(e) => setCollectionName(e.target.value)}
          className="rounded-md border border-[color:var(--color-border)] bg-[color:var(--color-card)] px-2 py-1 text-xs"
        >
          {collections.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name}
            </option>
          ))}
        </select>
        <select
          value={graphName}
          onChange={(e) => setGraphName(e.target.value)}
          className="rounded-md border border-[color:var(--color-border)] bg-[color:var(--color-card)] px-2 py-1 text-xs"
        >
          {graphs.map((g) => (
            <option key={g.name} value={g.name}>
              {g.name}
            </option>
          ))}
        </select>
        <button
          onClick={runBoth}
          disabled={!prompt.trim() || left.loading || right.loading}
          className="flex items-center gap-1.5 rounded-lg bg-[color:var(--color-signal)]/90 px-3.5 py-2 text-sm font-medium text-[#04201c] disabled:opacity-30"
        >
          <Send size={14} /> Run both
        </button>
      </div>

      <div className="flex flex-1 gap-4">
        <ColumnPanel column={left} collectionName={collectionName} graphName={graphName} onModeChange={(m) => setLeft(emptyColumn(m))} />
        <ColumnPanel column={right} collectionName={collectionName} graphName={graphName} onModeChange={(m) => setRight(emptyColumn(m))} />
      </div>
    </div>
  );
}
