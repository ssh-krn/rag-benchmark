// src/components/EvalDashboard.tsx
import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChevronDown, ChevronRight, Database, History, Play, RotateCw, Timer } from "lucide-react";
import { useEvalRunner } from "@/hooks/useEvalRunner";
import type { RagMode, TestCaseResult } from "@/types/rag";
import { RAG_MODES, RAG_MODE_LABELS } from "@/types/rag";

const METRIC_COLORS: Record<string, string> = {
  faithfulness: "#5eead4",
  answer_relevancy: "#a78bfa",
  contextual_precision: "#fbbf24",
  contextual_recall: "#fb7163",
};

function TelemetryCard({ label, value, color, icon: Icon }: { label: string; value: string | number; color?: string; icon?: React.ComponentType<{ size?: number; className?: string }> }) {
  return (
    <div className="flex-1 rounded-lg border border-[color:var(--color-border)] bg-[color:var(--color-card)] p-3">
      <div className="flex items-center gap-1.5 text-[10.5px] uppercase tracking-wide text-[color:var(--color-ink-faint)]">
        {Icon && <Icon size={11} />}
        {label}
      </div>
      <p className={`mt-1 tabular text-lg font-semibold ${color ?? "text-[color:var(--color-ink)]"}`}>{value}</p>
    </div>
  );
}

function DrilldownRow({ result }: { result: TestCaseResult }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <tr className="border-t border-[color:var(--color-border)]">
        <td className="py-1.5 pl-2">
          <button onClick={() => setOpen((o) => !o)} className="flex items-center gap-1 text-left">
            {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            <span className="text-[11.5px]">{result.question}</span>
          </button>
        </td>
        <td className="py-1.5 text-center text-[11px]">{result.mode}</td>
        <td className="py-1.5 text-center text-[11px]">{result.cache_hit ? "cached" : "fresh"}</td>
        <td className="py-1.5 pr-2 text-right text-[11px]">
          {result.error ? <span className="text-[color:var(--color-coral)]">error</span> : <span className="text-[color:var(--color-emerald)]">ok</span>}
        </td>
      </tr>
      {open && (
        <tr>
          <td colSpan={4} className="bg-[color:var(--color-panel)] px-3 py-3">
            {result.error ? (
              <p className="text-xs text-[color:var(--color-coral)]">{result.error}</p>
            ) : (
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-semibold text-[color:var(--color-ink-faint)]">Actual output</p>
                  <p className="mt-0.5 whitespace-pre-wrap text-[color:var(--color-ink-dim)]">{result.rag_output}</p>
                </div>
                {result.retrieved_contexts && result.retrieved_contexts.length > 0 && (
                  <div>
                    <p className="font-semibold text-[color:var(--color-ink-faint)]">Retrieved contexts ({result.retrieved_contexts.length})</p>
                    <ul className="mt-0.5 list-disc space-y-1 pl-4 text-[color:var(--color-ink-dim)]">
                      {result.retrieved_contexts.slice(0, 4).map((c, i) => (
                        <li key={i} className="line-clamp-2">
                          {c}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  {result.metrics.map((m) => (
                    <div key={m.metric_name} title={m.reason ?? undefined} className="cursor-help rounded-md border border-[color:var(--color-border)] bg-[color:var(--color-card)] p-2">
                      <p className="text-[10px] text-[color:var(--color-ink-faint)]">{m.metric_name}</p>
                      <p className="tabular text-sm font-semibold">{m.score != null ? m.score.toFixed(2) : "—"}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  );
}

export function EvalDashboard() {
  const { runId, report, starting, error, pastRunIds, startRun, resumeRun } = useEvalRunner();
  const [datasetPath, setDatasetPath] = useState("data/golden_eval_dataset.json");
  const [selectedModes, setSelectedModes] = useState<RagMode[]>([...RAG_MODES]);
  const [batchSize, setBatchSize] = useState(3);
  const [cooldown, setCooldown] = useState(15);
  const [resumeId, setResumeId] = useState("");

  const toggleMode = (m: RagMode) => {
    setSelectedModes((prev) => (prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]));
  };

  const chartData = useMemo(() => {
    if (!report) return [];
    return report.metrics_summary.map((s) => ({
      mode: RAG_MODE_LABELS[s.mode as RagMode] ?? s.mode,
      faithfulness: s.avg_faithfulness ?? undefined,
      answer_relevancy: s.avg_answer_relevancy ?? undefined,
      contextual_precision: s.avg_contextual_precision ?? undefined,
      contextual_recall: s.avg_contextual_recall ?? undefined,
    }));
  }, [report]);

  return (
    <div className="h-full overflow-y-auto p-5">
      <div className="mb-4 flex items-center gap-2">
        <Database size={18} className="text-[color:var(--color-signal)]" />
        <h2 className="text-sm font-semibold">DeepEval Benchmark Lab</h2>
      </div>

      <div className="mb-5 grid grid-cols-1 gap-4 lg:grid-cols-[1.1fr_1fr]">
        <div className="space-y-3 rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-card)] p-4">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-[color:var(--color-ink-faint)]">Start a run</h3>

          <label className="block text-xs">
            <span className="mb-1 block text-[color:var(--color-ink-dim)]">Dataset path</span>
            <input
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
              className="w-full rounded-md border border-[color:var(--color-border)] bg-[color:var(--color-panel)] px-2.5 py-1.5 font-mono text-[11px] outline-none"
            />
          </label>

          <div>
            <span className="mb-1 block text-xs text-[color:var(--color-ink-dim)]">RAG modes to evaluate</span>
            <div className="flex flex-wrap gap-1.5">
              {RAG_MODES.map((m) => (
                <label
                  key={m}
                  className={`flex cursor-pointer items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] ${
                    selectedModes.includes(m)
                      ? "border-[color:var(--color-signal-dim)]/50 bg-[color:var(--color-signal)]/[0.1] text-[color:var(--color-signal)]"
                      : "border-[color:var(--color-border)] text-[color:var(--color-ink-dim)]"
                  }`}
                >
                  <input type="checkbox" checked={selectedModes.includes(m)} onChange={() => toggleMode(m)} className="hidden" />
                  {RAG_MODE_LABELS[m]}
                </label>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <label className="block text-xs">
              <span className="mb-1 block text-[color:var(--color-ink-dim)]">Batch size</span>
              <input
                type="number"
                min={1}
                max={10}
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                className="w-full rounded-md border border-[color:var(--color-border)] bg-[color:var(--color-panel)] px-2.5 py-1.5 text-xs tabular outline-none"
              />
            </label>
            <label className="block text-xs">
              <span className="mb-1 block text-[color:var(--color-ink-dim)]">Inter-batch cooldown (s)</span>
              <input
                type="number"
                min={0}
                value={cooldown}
                onChange={(e) => setCooldown(Number(e.target.value))}
                className="w-full rounded-md border border-[color:var(--color-border)] bg-[color:var(--color-panel)] px-2.5 py-1.5 text-xs tabular outline-none"
              />
            </label>
          </div>

          <button
            onClick={() => startRun({ datasetPath, targetModes: selectedModes, batchSize, sleepBetweenBatches: cooldown })}
            disabled={starting || selectedModes.length === 0}
            className="flex w-full items-center justify-center gap-1.5 rounded-lg bg-[color:var(--color-signal)]/90 px-3.5 py-2 text-sm font-medium text-[#04201c] disabled:opacity-30"
          >
            <Play size={14} /> {starting ? "Starting…" : "Start evaluation run"}
          </button>
          {error && <p className="text-xs text-[color:var(--color-coral)]">{error}</p>}
        </div>

        <div className="space-y-3 rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-card)] p-4">
          <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-[color:var(--color-ink-faint)]">
            <History size={13} /> Resume an interrupted run
          </h3>
          <label className="block text-xs">
            <span className="mb-1 block text-[color:var(--color-ink-dim)]">resume_run_id</span>
            <div className="flex gap-2">
              <input
                value={resumeId}
                onChange={(e) => setResumeId(e.target.value)}
                placeholder="20261124-091200-ab12cd34"
                className="flex-1 rounded-md border border-[color:var(--color-border)] bg-[color:var(--color-panel)] px-2.5 py-1.5 font-mono text-[11px] outline-none"
              />
              <button
                onClick={() => resumeId && resumeRun(resumeId)}
                disabled={!resumeId}
                className="flex items-center gap-1.5 rounded-md border border-[color:var(--color-border)] px-3 py-1.5 text-xs hover:bg-[color:var(--color-card-hover)] disabled:opacity-30"
              >
                <RotateCw size={12} /> Resume
              </button>
            </div>
          </label>

          {pastRunIds.length > 0 && (
            <div>
              <span className="mb-1 block text-xs text-[color:var(--color-ink-dim)]">Quick-pick past runs</span>
              <div className="flex flex-wrap gap-1.5">
                {pastRunIds.map((id) => (
                  <button
                    key={id}
                    onClick={() => resumeRun(id)}
                    className={`rounded-full border px-2.5 py-1 font-mono text-[10.5px] tabular ${
                      id === runId ? "border-[color:var(--color-signal-dim)]/50 text-[color:var(--color-signal)]" : "border-[color:var(--color-border)] text-[color:var(--color-ink-dim)]"
                    }`}
                  >
                    {id}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {report && (
        <>
          <div className="mb-5 flex flex-wrap gap-3">
            <TelemetryCard label="Completed / Total" value={`${report.completed} / ${report.total}`} />
            <TelemetryCard label="SHA-256 Cache Hits" value={report.cached_hits} color="text-[color:var(--color-signal)]" />
            <TelemetryCard label="Failed / Retrying" value={report.failed} color={report.failed > 0 ? "text-[color:var(--color-coral)]" : undefined} />
            <TelemetryCard label="Active Judge Provider" value={report.active_provider ?? "—"} icon={Timer} />
            <TelemetryCard label="Status" value={report.status} color={report.status === "completed" ? "text-[color:var(--color-emerald)]" : "text-[color:var(--color-amber)]"} />
          </div>

          <div className="mb-5 rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-card)] p-4">
            <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-[color:var(--color-ink-faint)]">
              Multi-RAG metric comparison
            </h3>
            {chartData.length === 0 ? (
              <p className="text-xs text-[color:var(--color-ink-faint)]">No results yet — metrics will appear as cases complete.</p>
            ) : (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="mode" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
                  <YAxis domain={[0, 1]} tick={{ fill: "#a1a1aa", fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: "#18181b", border: "1px solid rgba(255,255,255,0.1)", fontSize: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="faithfulness" fill={METRIC_COLORS.faithfulness} name="Faithfulness" />
                  <Bar dataKey="answer_relevancy" fill={METRIC_COLORS.answer_relevancy} name="Answer Relevancy" />
                  <Bar dataKey="contextual_precision" fill={METRIC_COLORS.contextual_precision} name="Contextual Precision" />
                  <Bar dataKey="contextual_recall" fill={METRIC_COLORS.contextual_recall} name="Contextual Recall" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>

          <div className="rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-card)] p-4">
            <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-[color:var(--color-ink-faint)]">Detailed metrics matrix</h3>
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-[color:var(--color-ink-faint)]">
                  <th className="pb-2">Mode</th>
                  <th className="pb-2 text-center">Count</th>
                  <th className="pb-2 text-center">Faithfulness</th>
                  <th className="pb-2 text-center">Answer Relevancy</th>
                  <th className="pb-2 text-center">Contextual Precision</th>
                  <th className="pb-2 text-center">Contextual Recall</th>
                </tr>
              </thead>
              <tbody>
                {report.metrics_summary
                  .slice()
                  .sort((a, b) => (b.avg_faithfulness ?? 0) - (a.avg_faithfulness ?? 0))
                  .map((s) => (
                    <tr key={s.mode} className="border-t border-[color:var(--color-border)]">
                      <td className="py-1.5 font-medium">{RAG_MODE_LABELS[s.mode as RagMode] ?? s.mode}</td>
                      <td className="py-1.5 text-center tabular">{s.count}</td>
                      <td className="py-1.5 text-center tabular">{s.avg_faithfulness?.toFixed(3) ?? "—"}</td>
                      <td className="py-1.5 text-center tabular">{s.avg_answer_relevancy?.toFixed(3) ?? "—"}</td>
                      <td className="py-1.5 text-center tabular">{s.avg_contextual_precision?.toFixed(3) ?? "—"}</td>
                      <td className="py-1.5 text-center tabular">{s.avg_contextual_recall?.toFixed(3) ?? "—"}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>

          {report.results && report.results.length > 0 && (
            <div className="mt-5 rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-card)] p-4">
              <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-[color:var(--color-ink-faint)]">Per-question drilldown</h3>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left text-[color:var(--color-ink-faint)]">
                    <th className="pb-2 pl-2">Question</th>
                    <th className="pb-2 text-center">Mode</th>
                    <th className="pb-2 text-center">Cache</th>
                    <th className="pb-2 pr-2 text-right">Result</th>
                  </tr>
                </thead>
                <tbody>
                  {report.results.map((r) => (
                    <DrilldownRow key={`${r.golden_id}-${r.mode}`} result={r} />
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
