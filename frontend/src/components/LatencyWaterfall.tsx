// src/components/LatencyWaterfall.tsx
import { Zap, Clock, Boxes, RotateCcw } from "lucide-react";
import type { ChatMessage } from "@/types/rag";

interface LatencyWaterfallProps {
  message: ChatMessage;
}

function Stat({ icon: Icon, label, value, color }: { icon: React.ComponentType<{ size?: number; className?: string }>; label: string; value: string; color?: string }) {
  return (
    <div className="flex items-center gap-1.5 text-[11px] text-[color:var(--color-ink-dim)]">
      <Icon size={12} className={color ?? "text-[color:var(--color-ink-faint)]"} />
      <span>{label}</span>
      <span className="tabular text-[color:var(--color-ink)]">{value}</span>
    </div>
  );
}

export function LatencyWaterfall({ message }: LatencyWaterfallProps) {
  const { totalLatencyMs, ttftMs, retrievalMs, generationMs, modelUsed, providerUsed, fallbackOccurred, retryCount, retrievedContext } = message;

  if (totalLatencyMs == null && ttftMs == null && !modelUsed) return null;

  const retrieval = retrievalMs ?? 0;
  const generation = generationMs ?? Math.max(0, (totalLatencyMs ?? 0) - retrieval);
  const totalForBar = Math.max(retrieval + generation, 1);
  const retrievalPct = (retrieval / totalForBar) * 100;
  const generationPct = 100 - retrievalPct;

  return (
    <div className="space-y-2 rounded-lg border border-[color:var(--color-border)] bg-[color:var(--color-panel)] p-2.5">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
        {totalLatencyMs != null && <Stat icon={Clock} label="total" value={`${totalLatencyMs}ms`} />}
        {ttftMs != null && <Stat icon={Zap} label="ttft" value={`${ttftMs}ms`} color="text-[color:var(--color-signal)]" />}
        {retrievedContext && <Stat icon={Boxes} label="sources" value={`${retrievedContext.length}`} />}
        {retryCount != null && retryCount > 0 && <Stat icon={RotateCcw} label="retries" value={`${retryCount}`} color="text-[color:var(--color-amber)]" />}
        {modelUsed && (
          <span
            className={`ml-auto rounded-full border px-2.5 py-1 font-mono text-[11px] tabular ${
              fallbackOccurred
                ? "border-[color:var(--color-amber)]/40 bg-[color:var(--color-amber)]/[0.08] text-[color:var(--color-amber)]"
                : "border-[color:var(--color-border)] bg-[color:var(--color-card)] text-[color:var(--color-ink-dim)]"
            }`}
          >
            {providerUsed ?? "provider"} / {modelUsed}
            {fallbackOccurred ? " (fallback)" : ""}
          </span>
        )}
      </div>

      {(retrievalMs != null || generationMs != null) && (
        <div className="flex h-2 overflow-hidden rounded-full bg-[color:var(--color-card)]">
          <div
            className="bg-[color:var(--color-violet)]/70"
            style={{ width: `${retrievalPct}%` }}
            title={`Retrieval: ${retrieval}ms`}
          />
          <div
            className="bg-[color:var(--color-signal)]/70"
            style={{ width: `${generationPct}%` }}
            title={`Generation: ${generation}ms`}
          />
        </div>
      )}
    </div>
  );
}
