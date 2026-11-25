// src/components/ChatPanel.tsx
import type { ChatMessage, RagMode } from "@/types/rag";
import { RAG_MODE_LABELS } from "@/types/rag";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  Copy,
  FileJson,
  Loader2,
  RotateCcw,
  Send,
  Sparkles,
  Terminal,
  Trash2,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { ExecutionTraceBar } from "./ExecutionTraceBar";
import { LatencyWaterfall } from "./LatencyWaterfall";

const STARTER_PROMPTS: { mode: RagMode; label: string; prompt: string }[] = [
  {
    mode: "traditional_rag",
    label: "L1 vs L2 regularization",
    prompt: "What is the mathematical difference between L1 and L2 regularization penalties, and how does L1 induce sparsity in model weights?",
  },
  {
    mode: "hybrid_rag",
    label: "RRF formula lookup",
    prompt: "How does Reciprocal Rank Fusion (RRF) combine dense vector similarity scores with BM25 lexical rankings without requiring score normalization?",
  },
  {
    mode: "corrective_rag",
    label: "Self-reflective loop",
    prompt: "Explain the hallucination and relevance grading loop used to correct an insufficient retrieval before generation.",
  },
  {
    mode: "agentic_rag",
    label: "AdamW vs Adam",
    prompt: "How does PyTorch's AdamW optimizer decouple weight decay from the adaptive gradient update compared to standard L2 regularization in Adam?",
  },
  {
    mode: "lightrag",
    label: "Transformer block multi-hop",
    prompt: "Trace the architectural relationship between attention heads, layer normalization, residual connections, and feed-forward networks across a multi-layer Transformer encoder block.",
  },
];

/** Converts [Passage N] / [Web Source N] citation markers into markdown links
 *  with a `cite:` pseudo-scheme so they can be intercepted and rendered as
 *  interactive pills by the custom `a` component below. */
function preprocessCitations(markdown: string): string {
  return markdown
    .replace(/\[Passage\s+(\d+)\]/gi, (m, n) => `[${m}](cite:passage-${n})`)
    .replace(/\[Web Source\s+(\d+)\]/gi, (m, n) => `[${m}](cite:web-${n})`)
    .replace(/\[Web\s+(\d+)\]/gi, (m, n) => `[${m}](cite:web-${n})`);
}

interface ChatPanelProps {
  messages: ChatMessage[];
  isStreaming: boolean;
  onSend: (question: string) => void;
  onRetry: () => void;
  onClear: () => void;
  mode: RagMode;
  onSelectStarter: (mode: RagMode, prompt: string) => void;
  onCiteClick: (ref: string) => void;
  inputRef: React.RefObject<HTMLTextAreaElement | null>;
}

function MessageToolbar({ message }: { message: ChatMessage }) {
  const [showRaw, setShowRaw] = useState(false);
  const [copied, setCopied] = useState(false);

  const copyMarkdown = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const exportJson = () => {
    const payload = {
      content: message.content,
      executionTrace: message.executionTrace,
      retrievedContext: message.retrievedContext,
      toolCalls: message.toolCalls,
      grades: message.grades,
      totalLatencyMs: message.totalLatencyMs,
      ttftMs: message.ttftMs,
      modelUsed: message.modelUsed,
      providerUsed: message.providerUsed,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trace-${message.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <div className="mt-2 flex items-center gap-1 text-[color:var(--color-ink-faint)]">
        <button onClick={copyMarkdown} title="Copy markdown" className="rounded p-1.5 hover:bg-[color:var(--color-card-hover)] hover:text-[color:var(--color-ink)]">
          <Copy size={13} />
        </button>
        <button onClick={exportJson} title="Export trace + metadata as JSON" className="rounded p-1.5 hover:bg-[color:var(--color-card-hover)] hover:text-[color:var(--color-ink)]">
          <FileJson size={13} />
        </button>
        <button
          onClick={() => setShowRaw((s) => !s)}
          title="Toggle raw SSE event log"
          className={`rounded p-1.5 hover:bg-[color:var(--color-card-hover)] hover:text-[color:var(--color-ink)] ${showRaw ? "text-[color:var(--color-signal)]" : ""}`}
        >
          <Terminal size={13} />
        </button>
        {copied && <span className="text-[11px] text-[color:var(--color-emerald)]">copied</span>}
      </div>
      {showRaw && (
        <div className="mt-2 max-h-56 overflow-y-auto rounded-lg border border-[color:var(--color-border)] bg-black/40 p-2 font-mono text-[11px] text-[color:var(--color-ink-dim)]">
          {(message.rawFrames ?? []).length === 0 && <p>No raw frames captured (synchronous fallback was used).</p>}
          {(message.rawFrames ?? []).map((f, i) => (
            <div key={i} className="border-b border-white/5 py-1">
              <span className="text-[color:var(--color-signal)]">event: {f.event}</span>
              <pre className="whitespace-pre-wrap break-all">{f.raw}</pre>
            </div>
          ))}
        </div>
      )}
    </>
  );
}

function AssistantBubble({ message, onCiteClick }: { message: ChatMessage; onCiteClick: (ref: string) => void }) {
  return (
    <div className="max-w-[46rem] space-y-2.5">
      {(message.executionTrace?.length || message.toolCalls?.length || message.grades?.length) ? (
        <ExecutionTraceBar steps={message.executionTrace ?? []} toolCalls={message.toolCalls ?? []} grades={message.grades ?? []} />
      ) : null}

      <div className="rounded-2xl rounded-tl-sm border border-[color:var(--color-border)] bg-[color:var(--color-card)] px-4 py-3">
        {message.error && (
          <div className="mb-2 flex items-center gap-2 rounded-md border border-[color:var(--color-coral)]/30 bg-[color:var(--color-coral)]/[0.08] px-2.5 py-1.5 text-xs text-[color:var(--color-coral)]">
            <AlertTriangle size={13} />
            <span>{message.error}</span>
          </div>
        )}
        <div className="msg-prose text-[13.5px] leading-relaxed text-[color:var(--color-ink)]">
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
            components={{
              a: ({ href, children }) => {
                if (href?.startsWith("cite:")) {
                  const ref = href.slice("cite:".length);
                  return (
                    <button
                      onClick={() => onCiteClick(ref)}
                      className="mx-0.5 rounded-full border border-[color:var(--color-signal-dim)]/40 bg-[color:var(--color-signal)]/[0.12] px-1.5 py-0.5 text-[11px] font-medium text-[color:var(--color-signal)] transition hover:bg-[color:var(--color-signal)]/[0.22]"
                    >
                      {children}
                    </button>
                  );
                }
                return (
                  <a href={href} target="_blank" rel="noreferrer">
                    {children}
                  </a>
                );
              },
            }}
          >
            {preprocessCitations(message.content || (message.isStreaming ? "" : "_No content returned._"))}
          </ReactMarkdown>
          {message.isStreaming && <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse bg-[color:var(--color-signal)] align-middle" />}
        </div>
        {!message.isStreaming && message.content && <MessageToolbar message={message} />}
      </div>

      <LatencyWaterfall message={message} />
    </div>
  );
}

export function ChatPanel({ messages, isStreaming, onSend, onRetry, onClear, mode, onSelectStarter, onCiteClick, inputRef }: ChatPanelProps) {
  const [draft, setDraft] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages.length, messages[messages.length - 1]?.content]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const submit = () => {
    const q = draft.trim();
    if (!q || isStreaming) return;
    onSend(q);
    setDraft("");
  };

  return (
    <div className="flex h-full flex-col">
      <div className="mx-auto mt-4 max-w-xl rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-panel)]/70 px-3.5 py-2.5 text-left text-xs text-[color:var(--color-ink-dim)]">
        <span className="font-medium text-[color:var(--color-signal)]">Note:</span>{" "}
        Uploading new PDFs is disabled due to local compute (CPU/GPU) and free-tier API limits. 
        Retrieval is currently scoped to the pre-indexed{" "}
        <span className="text-[color:var(--color-ink)]">Hundred-Page Machine Learning Book</span>{" "}
        corpus.
      </div>
      <div ref={scrollRef} className="flex-1 space-y-5 overflow-y-auto px-5 py-5">
        {messages.length === 0 && (
          <div className="mx-auto max-w-2xl pt-10 text-center">
            <Sparkles className="mx-auto mb-3 text-[color:var(--color-signal)]" size={26} />
            <h2 className="text-lg font-semibold">Ask a question, pick a retrieval strategy</h2>
            <p className="mt-1.5 text-sm text-[color:var(--color-ink-dim)]">
              Starter prompts below auto-select the mode they showcase best.
            </p>
            <div className="mt-5 flex flex-wrap justify-center gap-2">
              {STARTER_PROMPTS.map((s) => (
                <button
                  key={s.label}
                  onClick={() => onSelectStarter(s.mode, s.prompt)}
                  className="rounded-full border border-[color:var(--color-border)] bg-[color:var(--color-card)] px-3 py-1.5 text-xs text-[color:var(--color-ink-dim)] transition hover:border-[color:var(--color-signal-dim)]/50 hover:text-[color:var(--color-ink)]"
                >
                  <span className="mr-1.5 text-[color:var(--color-ink-faint)]">{RAG_MODE_LABELS[s.mode]}:</span>
                  {s.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m) =>
          m.role === "user" ? (
            <div key={m.id} className="flex justify-end">
              <div className="max-w-[32rem] rounded-2xl rounded-tr-sm border border-[color:var(--color-border)] bg-[color:var(--color-panel)] px-4 py-2.5 text-[13.5px] text-[color:var(--color-ink)]">
                {m.content}
              </div>
            </div>
          ) : (
            <motion.div key={m.id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
              <AssistantBubble message={m} onCiteClick={onCiteClick} />
            </motion.div>
          ),
        )}

        {isStreaming && (
          <div className="flex items-center gap-2 text-xs text-[color:var(--color-ink-faint)]">
            <Loader2 size={13} className="animate-spin" />
            Running {RAG_MODE_LABELS[mode]}…
          </div>
        )}
      </div>

      <div className="border-t border-[color:var(--color-border)] p-3">
        <div className="flex items-end gap-2 rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-panel)] p-2">
          <textarea
            ref={inputRef}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={`Ask something in ${RAG_MODE_LABELS[mode]} mode… (Enter to send, Shift+Enter for newline)`}
            rows={1}
            className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm outline-none placeholder:text-[color:var(--color-ink-faint)]"
          />
          <button
            onClick={() => onRetry()}
            title="Retry last question"
            disabled={messages.length === 0 || isStreaming}
            className="rounded-lg p-2 text-[color:var(--color-ink-faint)] transition hover:bg-[color:var(--color-card-hover)] hover:text-[color:var(--color-ink)] disabled:opacity-30"
          >
            <RotateCcw size={16} />
          </button>
          <button
            onClick={onClear}
            title="Clear conversation (Cmd/Ctrl+K)"
            disabled={messages.length === 0}
            className="rounded-lg p-2 text-[color:var(--color-ink-faint)] transition hover:bg-[color:var(--color-card-hover)] hover:text-[color:var(--color-ink)] disabled:opacity-30"
          >
            <Trash2 size={16} />
          </button>
          <button
            onClick={submit}
            disabled={!draft.trim() || isStreaming}
            className="flex items-center gap-1.5 rounded-lg bg-[color:var(--color-signal)]/90 px-3.5 py-2 text-sm font-medium text-[#04201c] transition hover:bg-[color:var(--color-signal)] disabled:cursor-not-allowed disabled:opacity-30"
          >
            <Send size={14} />
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
