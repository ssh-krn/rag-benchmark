// src/components/ExecutionTraceBar.tsx
import { useState } from "react";
import { ChevronRight, Wrench, ShieldCheck, ShieldAlert } from "lucide-react";
import type { ExecutionStep, GradeRecord, ToolCallRecord } from "@/types/rag";

function NodeStepper({ steps }: { steps: ExecutionStep[] }) {
  if (steps.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {steps.map((step, i) => (
        <div key={`${step.node}-${i}`} className="flex items-center gap-1.5">
          <div
            title={step.description}
            className="flex cursor-help items-center gap-1.5 rounded-full border border-[color:var(--color-border)] bg-[color:var(--color-panel)] px-2.5 py-1 text-[11px] text-[color:var(--color-ink-dim)]"
          >
            <span className="text-[color:var(--color-ink)]">{step.node}</span>
            {step.durationMs != null && <span className="tabular text-[color:var(--color-ink-faint)]">{step.durationMs}ms</span>}
          </div>
          {i < steps.length - 1 && <ChevronRight size={12} className="text-[color:var(--color-ink-faint)]" />}
        </div>
      ))}
    </div>
  );
}

function ToolCallPill({ call }: { call: ToolCallRecord }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-lg border border-[color:var(--color-border)] bg-[color:var(--color-panel)] text-xs">
      <button onClick={() => setOpen((o) => !o)} className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left">
        <Wrench size={12} className="text-[color:var(--color-signal)]" />
        <span className="font-mono">{call.tool}</span>
        <span className="ml-auto tabular text-[color:var(--color-ink-faint)]">
          {call.result_chunks != null ? `${call.result_chunks} chunks` : ""}
          {call.result_chars != null ? ` · ${call.result_chars} chars` : ""}
        </span>
      </button>
      {open && (
        <div className="border-t border-[color:var(--color-border)] px-2.5 py-2 font-mono text-[11px] text-[color:var(--color-ink-dim)]">
          <pre className="whitespace-pre-wrap">{JSON.stringify(call.args, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

function GradeBadge({ grade }: { grade: GradeRecord }) {
  // Verdicts are derived client-side from execution_trace descriptions (see
  // useRagStream's deriveGrades) and normalized to lowercase: "relevant",
  // "not relevant", "grounded", "hallucinated", or "unknown".
  const positive = grade.verdict === "relevant" || grade.verdict === "grounded";
  const Icon = positive ? ShieldCheck : ShieldAlert;
  const color = positive ? "text-[color:var(--color-emerald)] border-[color:var(--color-emerald)]/30 bg-[color:var(--color-emerald)]/[0.08]"
    : "text-[color:var(--color-coral)] border-[color:var(--color-coral)]/30 bg-[color:var(--color-coral)]/[0.08]";
  return (
    <div title={grade.reasoning} className={`flex cursor-help items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] ${color}`}>
      <Icon size={12} />
      <span className="capitalize">{grade.grader}</span>
      <span className="font-mono">{grade.verdict}</span>
      {grade.retry_attempt > 0 && <span className="tabular text-[color:var(--color-ink-faint)]">retry {grade.retry_attempt}</span>}
    </div>
  );
}

interface ExecutionTraceBarProps {
  steps: ExecutionStep[];
  toolCalls: ToolCallRecord[];
  grades: GradeRecord[];
}

export function ExecutionTraceBar({ steps, toolCalls, grades }: ExecutionTraceBarProps) {
  if (steps.length === 0 && toolCalls.length === 0 && grades.length === 0) return null;
  return (
    <div className="space-y-2 rounded-lg border border-[color:var(--color-border)] bg-[color:var(--color-canvas)]/60 p-2.5">
      <NodeStepper steps={steps} />
      {toolCalls.length > 0 && (
        <div className="grid gap-1.5 sm:grid-cols-2">
          {toolCalls.map((c) => (
            <ToolCallPill key={c.id} call={c} />
          ))}
        </div>
      )}
      {grades.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {grades.map((g) => (
            <GradeBadge key={g.id} grade={g} />
          ))}
        </div>
      )}
    </div>
  );
}
