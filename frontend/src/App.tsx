// src/App.tsx
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { HeaderBar, type ViewKey } from "@/components/HeaderBar";
import { HealthModal } from "@/components/HealthModal";
import { ChatPanel } from "@/components/ChatPanel";
import { SourceInspector } from "@/components/SourceInspector";
import { CompareView } from "@/components/CompareView";
import { EvalDashboard } from "@/components/EvalDashboard";
import { AboutView } from "@/components/AboutView";
import { useSystemHealth } from "@/hooks/useSystemHealth";
import { useRagStream } from "@/hooks/useRagStream";
import type { RagMode } from "@/types/rag";

export default function App() {
  const { health, collections, loading: healthLoading, refresh } = useSystemHealth();
  const { messages, isStreaming, sendMessage, retryLast, clear } = useRagStream();

  const [mode, setMode] = useState<RagMode>("traditional_rag");
  const [collectionName, setCollectionName] = useState<string | undefined>(undefined);
  const [graphName, setGraphName] = useState<string | undefined>(undefined);
  const [view, setView] = useState<ViewKey>("workbench");
  const [healthModalOpen, setHealthModalOpen] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [focusedRef, setFocusedRef] = useState<string | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const collectionList = collections?.collections ?? [];
  const graphList = collections?.lightrag_graphs ?? [];

  // Default collection/graph selection once options load.
  useEffect(() => {
    if (!collectionName && collectionList.length > 0) setCollectionName(collectionList[0].name);
  }, [collectionList, collectionName]);
  useEffect(() => {
    if (!graphName && graphList.length > 0) setGraphName(graphList[0].name);
  }, [graphList, graphName]);

  const lastAssistant = useMemo(() => [...messages].reverse().find((m) => m.role === "assistant") ?? null, [messages]);

  const handleSend = useCallback(
    (question: string) => {
      sendMessage({ question, mode, collectionName, graphName });
    },
    [sendMessage, mode, collectionName, graphName],
  );

  const handleSelectStarter = useCallback(
    (m: RagMode, prompt: string) => {
      setMode(m);
      sendMessage({ question: prompt, mode: m, collectionName, graphName });
    },
    [sendMessage, collectionName, graphName],
  );

  const handleCiteClick = useCallback((ref: string) => {
    setInspectorOpen(true);
    setFocusedRef(ref);
    requestAnimationFrame(() => {
      document.getElementById(`cite-${ref}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
    window.setTimeout(() => setFocusedRef((r) => (r === ref ? null : r)), 2500);
  }, []);

  // Global keyboard shortcuts: Cmd/Ctrl+K clear chat, Cmd/Ctrl+I toggle inspector.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (!mod) return;
      if (e.key.toLowerCase() === "k") {
        e.preventDefault();
        clear();
      } else if (e.key.toLowerCase() === "i") {
        e.preventDefault();
        setInspectorOpen((o) => !o);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [clear]);

  return (
    <div className="flex h-screen flex-col bg-[color:var(--color-canvas)] text-[color:var(--color-ink)]">
      <HeaderBar
        mode={mode}
        onModeChange={setMode}
        collections={collectionList}
        graphs={graphList}
        collectionName={collectionName}
        graphName={graphName}
        onCollectionChange={setCollectionName}
        onGraphChange={setGraphName}
        health={health}
        healthLoading={healthLoading}
        onOpenHealth={() => setHealthModalOpen(true)}
        view={view}
        onViewChange={setView}
      />

      <main className="flex flex-1 overflow-hidden">
        {view === "workbench" && (
          <>
            <div className="flex-1 overflow-hidden">
              <ChatPanel
                messages={messages}
                isStreaming={isStreaming}
                onSend={handleSend}
                onRetry={() => retryLast()}
                onClear={clear}
                mode={mode}
                onSelectStarter={handleSelectStarter}
                onCiteClick={handleCiteClick}
                inputRef={inputRef}
              />
            </div>
            <SourceInspector message={lastAssistant} open={inspectorOpen} onToggle={() => setInspectorOpen((o) => !o)} focusedRef={focusedRef} />
          </>
        )}

        {view === "compare" && <CompareView collections={collectionList} graphs={graphList} />}
        {view === "evals" && <EvalDashboard />}
        {view === "about" && <AboutView />}
      </main>

      <HealthModal open={healthModalOpen} onClose={() => setHealthModalOpen(false)} health={health} onRefresh={refresh} />
    </div>
  );
}
