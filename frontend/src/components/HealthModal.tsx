// src/components/HealthModal.tsx
import { AnimatePresence, motion } from "framer-motion";
import { X, CircleCheck, CircleX, Timer, Database, Share2 } from "lucide-react";
import type { HealthResponse, ProviderReport } from "@/types/rag";

interface HealthModalProps {
  open: boolean;
  onClose: () => void;
  health: HealthResponse | null;
  onRefresh: () => void;
}

function ProviderTable({ name, report }: { name: string; report: ProviderReport }) {
  const allModels = [
    ...report.verified_models.map((m) => ({ model: m, ok: true, reason: undefined as string | undefined })),
    ...Object.entries(report.rejected_models).map(([m, reason]) => ({ model: m, ok: false, reason })),
  ];

  return (
    <div className="rounded-lg border border-[color:var(--color-border)] bg-[color:var(--color-panel)] p-3">
      <div className="mb-2 flex items-center justify-between">
        <h4 className="text-sm font-semibold capitalize">{name.replace("_", " ")}</h4>
        {report.reachable ? (
          <span className="flex items-center gap-1 text-[11px] text-[color:var(--color-emerald)]">
            <CircleCheck size={13} /> reachable
          </span>
        ) : (
          <span className="flex items-center gap-1 text-[11px] text-[color:var(--color-coral)]">
            <CircleX size={13} /> unreachable
          </span>
        )}
      </div>
      {report.error && <p className="mb-2 text-xs text-[color:var(--color-coral)]">{report.error}</p>}
      <table className="w-full text-xs">
        <tbody>
          {allModels.length === 0 && (
            <tr>
              <td className="py-1 text-[color:var(--color-ink-faint)]">No models checked.</td>
            </tr>
          )}
          {allModels.map((m) => (
            <tr key={m.model} className="border-t border-[color:var(--color-border)]">
              <td className="py-1.5 pr-2 font-mono tabular">{m.model}</td>
              <td className="py-1.5 text-right">
                {m.ok ? (
                  <span className="text-[color:var(--color-emerald)]">verified</span>
                ) : (
                  <span title={m.reason} className="cursor-help text-[color:var(--color-coral)]">
                    failed
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function HealthModal({ open, onClose, health, onRefresh }: HealthModalProps) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 p-4 pt-16 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            className="max-h-[80vh] w-full max-w-3xl overflow-y-auto rounded-xl border border-[color:var(--color-border-strong)] bg-[color:var(--color-card)] shadow-2xl"
            initial={{ opacity: 0, y: -12, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -12, scale: 0.98 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-[color:var(--color-border)] px-5 py-3.5">
              <h2 className="text-sm font-semibold">System Health &amp; Pre-Flight Model Inspector</h2>
              <div className="flex items-center gap-2">
                <button onClick={onRefresh} className="rounded-md border border-[color:var(--color-border)] px-2.5 py-1 text-xs hover:bg-[color:var(--color-card-hover)]">
                  Refresh
                </button>
                <button onClick={onClose} className="rounded-md p-1 hover:bg-[color:var(--color-card-hover)]">
                  <X size={16} />
                </button>
              </div>
            </div>

            <div className="space-y-4 p-5">
              {!health ? (
                <p className="text-sm text-[color:var(--color-ink-dim)]">Loading health report…</p>
              ) : (
                <>
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <div className="rounded-lg border border-[color:var(--color-border)] bg-[color:var(--color-panel)] p-3">
                      <div className="mb-1.5 flex items-center gap-2 text-xs font-semibold">
                        <Database size={14} className="text-[color:var(--color-signal)]" /> ChromaDB
                      </div>
                      <p className="text-xs text-[color:var(--color-ink-dim)]">status: {health.chromadb.status}</p>
                      <p className="text-xs text-[color:var(--color-ink-dim)]">path: {health.chromadb.path}</p>
                      <p className="mt-1 text-xs text-[color:var(--color-ink-faint)]">
                        collections: {(health.chromadb.collections ?? []).join(", ") || "none"}
                      </p>
                      <p className="mt-1 text-xs text-[color:var(--color-ink-faint)]">
                        embeddings: all-MiniLM-L6-v2 · reranker: ms-marco-MiniLM-L-12-v2
                      </p>
                    </div>
                    <div className="rounded-lg border border-[color:var(--color-border)] bg-[color:var(--color-panel)] p-3">
                      <div className="mb-1.5 flex items-center gap-2 text-xs font-semibold">
                        <Share2 size={14} className="text-[color:var(--color-violet)]" /> LightRAG Graphs
                      </div>
                      {health.lightrag_graph.available_graphs.length === 0 ? (
                        <p className="text-xs text-[color:var(--color-ink-faint)]">No graphs found in {health.lightrag_graph.graph_dir}</p>
                      ) : (
                        <ul className="space-y-1 text-xs text-[color:var(--color-ink-dim)]">
                          {health.lightrag_graph.available_graphs.map((g) => (
                            <li key={g.name} className="tabular">
                              {g.name} — {g.nodes ?? "?"} nodes / {g.edges ?? "?"} edges
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  </div>

                  <div>
                    <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-[color:var(--color-ink-faint)]">
                      Live free-tier &amp; Ollama :cloud model verification
                    </h3>
                    <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
                      <ProviderTable name="groq" report={health.preflight_verification.groq} />
                      <ProviderTable name="gemini" report={health.preflight_verification.gemini} />
                      <ProviderTable name="ollama_cloud" report={health.preflight_verification.ollama_cloud} />
                    </div>
                  </div>

                  <div>
                    <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-[color:var(--color-ink-faint)]">
                      Active judge chain (ordered failover)
                    </h3>
                    <div className="flex flex-wrap items-center gap-2">
                      {health.preflight_verification.active_judge_chain.length === 0 && (
                        <span className="text-xs text-[color:var(--color-coral)]">No verified providers — chat and evals will fail.</span>
                      )}
                      {health.preflight_verification.active_judge_chain.map((entry, i) => (
                        <div key={`${entry.provider}-${entry.model}-${i}`} className="flex items-center gap-2">
                          <span className="rounded-full border border-[color:var(--color-border)] bg-[color:var(--color-panel)] px-2.5 py-1 font-mono text-[11px] tabular">
                            {entry.provider} / {entry.model}
                          </span>
                          {i < health.preflight_verification.active_judge_chain.length - 1 && (
                            <span className="text-[color:var(--color-ink-faint)]">→</span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-[color:var(--color-ink-faint)]">
                      <Timer size={13} /> Active 429 cooldowns / circuit breakers
                    </h3>
                    {health.active_cooldowns.length === 0 ? (
                      <p className="text-xs text-[color:var(--color-ink-dim)]">No models are currently cooling down.</p>
                    ) : (
                      <ul className="space-y-1.5">
                        {health.active_cooldowns.map((cd, i) => (
                          <li
                            key={i}
                            className="flex items-center justify-between rounded-md border border-[color:var(--color-amber)]/30 bg-[color:var(--color-amber)]/[0.08] px-3 py-1.5 text-xs"
                          >
                            <span className="font-mono tabular">
                              {cd.provider} / {cd.model}
                            </span>
                            <span className="tabular text-[color:var(--color-amber)]">{cd.seconds_remaining}s remaining</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </>
              )}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
