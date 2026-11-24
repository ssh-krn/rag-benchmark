// src/components/HeaderBar.tsx
import { useState } from "react";
import {
  Activity,
  Database,
  GitMerge,
  RefreshCcw,
  Bot,
  Share2,
  ChevronDown,
  CircleCheck,
  CircleAlert,
  CircleX,
} from "lucide-react";
import type { CollectionInfo, GraphInfo, HealthResponse, RagMode } from "@/types/rag";
import { RAG_MODES, RAG_MODE_LABELS, RAG_MODE_SUBTITLES } from "@/types/rag";

export type ViewKey = "workbench" | "compare" | "evals" | "about";

const MODE_ICONS: Record<RagMode, React.ComponentType<{ size?: number; className?: string }>> = {
  traditional_rag: Database,
  hybrid_rag: GitMerge,
  corrective_rag: RefreshCcw,
  agentic_rag: Bot,
  lightrag: Share2,
};

const VIEW_TABS: { key: ViewKey; label: string }[] = [
  { key: "workbench", label: "Workbench" },
  { key: "compare", label: "Side-by-Side Compare" },
  { key: "evals", label: "DeepEval Lab" },
  { key: "about", label: "About & Architecture" },
];

interface HeaderBarProps {
  mode: RagMode;
  onModeChange: (m: RagMode) => void;
  collections: CollectionInfo[];
  graphs: GraphInfo[];
  collectionName?: string;
  graphName?: string;
  onCollectionChange: (name: string) => void;
  onGraphChange: (name: string) => void;
  health: HealthResponse | null;
  healthLoading: boolean;
  onOpenHealth: () => void;
  view: ViewKey;
  onViewChange: (v: ViewKey) => void;
}

function HealthPill({ health, loading, onClick }: { health: HealthResponse | null; loading: boolean; onClick: () => void }) {
  let Icon = CircleAlert;
  let color = "text-amber";
  let label = "Checking…";

  if (!loading && health) {
    const chainLen = health.preflight_verification?.active_judge_chain?.length ?? 0;
    const cooldowns = health.active_cooldowns?.length ?? 0;
    if (chainLen === 0) {
      Icon = CircleX;
      color = "text-[color:var(--color-coral)]";
      label = "No providers verified";
    } else if (cooldowns > 0) {
      Icon = CircleAlert;
      color = "text-[color:var(--color-amber)]";
      label = `${chainLen} models active · ${cooldowns} cooling down`;
    } else {
      Icon = CircleCheck;
      color = "text-[color:var(--color-emerald)]";
      label = `${chainLen} models verified`;
    }
  }

  return (
    <button
      onClick={onClick}
      className="flex items-center gap-2 rounded-full border border-[color:var(--color-border)] bg-[color:var(--color-card)] px-3 py-1.5 text-xs text-[color:var(--color-ink-dim)] transition hover:border-[color:var(--color-border-strong)] hover:bg-[color:var(--color-card-hover)]"
    >
      <Icon size={14} className={color} />
      <span className="tabular">{label}</span>
    </button>
  );
}

function ReadOnlySelect({
  label,
  value,
  options,
  onChange,
  highlighted,
}: {
  label: string;
  value?: string;
  options: { value: string; badge?: string }[];
  onChange: (v: string) => void;
  highlighted?: boolean;
}) {
  return (
    <label
      className={`flex items-center gap-2 rounded-lg border px-2.5 py-1.5 text-xs transition ${
        highlighted
          ? "border-[color:var(--color-signal-dim)]/50 bg-[color:var(--color-signal)]/[0.07]"
          : "border-[color:var(--color-border)] bg-[color:var(--color-card)]"
      }`}
    >
      <span className="text-[color:var(--color-ink-faint)]">{label}</span>
      <div className="relative">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="appearance-none bg-transparent pr-5 text-[color:var(--color-ink)] outline-none"
        >
          {options.length === 0 && <option value="">none available</option>}
          {options.map((o) => (
            <option key={o.value} value={o.value} className="bg-[color:var(--color-card)]">
              {o.value} {o.badge ? `(${o.badge})` : ""}
            </option>
          ))}
        </select>
        <ChevronDown size={12} className="pointer-events-none absolute right-0 top-1/2 -translate-y-1/2 text-[color:var(--color-ink-faint)]" />
      </div>
    </label>
  );
}

export function HeaderBar(props: HeaderBarProps) {
  const {
    mode,
    onModeChange,
    collections,
    graphs,
    collectionName,
    graphName,
    onCollectionChange,
    onGraphChange,
    health,
    healthLoading,
    onOpenHealth,
    view,
    onViewChange,
  } = props;
  const [hoveredMode, setHoveredMode] = useState<RagMode | null>(null);

  return (
    <header className="sticky top-0 z-30 border-b border-[color:var(--color-border)] bg-[color:var(--color-canvas)]/95 backdrop-blur">
      <div className="flex flex-wrap items-center gap-3 px-4 py-3">
        <div className="flex items-center gap-2 pr-2">
          <Activity size={18} className="text-[color:var(--color-signal)]" />
          <span className="text-sm font-semibold tracking-tight">RAG Research Workbench</span>
        </div>

        <HealthPill health={health} loading={healthLoading} onClick={onOpenHealth} />

        <nav className="ml-1 flex items-center gap-1 rounded-full border border-[color:var(--color-border)] bg-[color:var(--color-panel)] p-1">
          {RAG_MODES.map((m) => {
            const Icon = MODE_ICONS[m];
            const active = m === mode;
            return (
              <div key={m} className="relative">
                <button
                  onClick={() => onModeChange(m)}
                  onMouseEnter={() => setHoveredMode(m)}
                  onMouseLeave={() => setHoveredMode((h) => (h === m ? null : h))}
                  className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium transition ${
                    active
                      ? "bg-[color:var(--color-signal)]/15 text-[color:var(--color-signal)]"
                      : "text-[color:var(--color-ink-dim)] hover:bg-[color:var(--color-card-hover)] hover:text-[color:var(--color-ink)]"
                  }`}
                >
                  <Icon size={14} />
                  <span className="hidden lg:inline">{RAG_MODE_LABELS[m]}</span>
                </button>
                {hoveredMode === m && (
                  <div className="absolute left-1/2 top-full z-40 mt-2 w-56 -translate-x-1/2 rounded-lg border border-[color:var(--color-border-strong)] bg-[color:var(--color-card)] p-2.5 text-xs shadow-xl">
                    <p className="font-semibold text-[color:var(--color-ink)]">{RAG_MODE_LABELS[m]}</p>
                    <p className="mt-0.5 text-[color:var(--color-ink-dim)]">{RAG_MODE_SUBTITLES[m]}</p>
                  </div>
                )}
              </div>
            );
          })}
        </nav>

        <div className="flex items-center gap-2">
          <ReadOnlySelect
            label="collection"
            value={collectionName}
            options={collections.map((c) => ({ value: c.name, badge: c.chunk_count != null ? `${c.chunk_count} chunks` : undefined }))}
            onChange={onCollectionChange}
          />
          <ReadOnlySelect
            label="graph"
            value={graphName}
            options={graphs.map((g) => ({ value: g.name, badge: g.nodes != null ? `${g.nodes}n/${g.edges}e` : undefined }))}
            onChange={onGraphChange}
            highlighted={mode === "lightrag"}
          />
        </div>

        <div className="ml-auto flex items-center gap-1 rounded-full border border-[color:var(--color-border)] bg-[color:var(--color-panel)] p-1">
          {VIEW_TABS.map((tab) => (
            <button
              key={tab.key}
              onClick={() => onViewChange(tab.key)}
              className={`rounded-full px-3 py-1.5 text-xs font-medium transition ${
                view === tab.key
                  ? "bg-[color:var(--color-card)] text-[color:var(--color-ink)]"
                  : "text-[color:var(--color-ink-dim)] hover:text-[color:var(--color-ink)]"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>
    </header>
  );
}
