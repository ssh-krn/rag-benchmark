// src/components/AboutView.tsx
import { Database, GitMerge, RefreshCcw, Bot, Share2, BookOpen } from "lucide-react";

interface ArchCardProps {
  icon: React.ComponentType<{ size?: number; className?: string }>;
  title: string;
  subtitle: string;
  flow: string[];
  strengths: string;
  weaknesses: string;
  accent: string;
}

function ArchCard({ icon: Icon, title, subtitle, flow, strengths, weaknesses, accent }: ArchCardProps) {
  return (
    <div className="rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-card)] p-5">
      <div className="mb-3 flex items-center gap-2.5">
        <div className={`flex h-9 w-9 items-center justify-center rounded-lg ${accent}`}>
          <Icon size={17} />
        </div>
        <div>
          <h3 className="text-sm font-semibold">{title}</h3>
          <p className="text-xs text-[color:var(--color-ink-faint)]">{subtitle}</p>
        </div>
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-1.5 rounded-lg border border-[color:var(--color-border)] bg-[color:var(--color-panel)] p-2.5">
        {flow.map((step, i) => (
          <div key={i} className="flex items-center gap-1.5">
            <span className="rounded-full bg-[color:var(--color-card)] px-2 py-1 font-mono text-[10.5px] text-[color:var(--color-ink-dim)]">{step}</span>
            {i < flow.length - 1 && <span className="text-[color:var(--color-ink-faint)]">→</span>}
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        <div className="rounded-lg border border-[color:var(--color-emerald)]/20 bg-[color:var(--color-emerald)]/[0.05] p-2.5">
          <p className="mb-1 text-[10.5px] font-semibold uppercase tracking-wide text-[color:var(--color-emerald)]">Works best when</p>
          <p className="text-xs text-[color:var(--color-ink-dim)]">{strengths}</p>
        </div>
        <div className="rounded-lg border border-[color:var(--color-coral)]/20 bg-[color:var(--color-coral)]/[0.05] p-2.5">
          <p className="mb-1 text-[10.5px] font-semibold uppercase tracking-wide text-[color:var(--color-coral)]">Weaknesses</p>
          <p className="text-xs text-[color:var(--color-ink-dim)]">{weaknesses}</p>
        </div>
      </div>
    </div>
  );
}

const METRICS = [
  {
    name: "Faithfulness",
    desc: "Measures whether every claim in the generated answer is actually supported by the retrieved context — the core defense against hallucination.",
  },
  {
    name: "Answer Relevancy",
    desc: "Measures whether the answer actually addresses the question asked, independent of whether it's grounded — penalizes off-topic or evasive responses.",
  },
  {
    name: "Contextual Precision",
    desc: "Measures whether the most relevant retrieved chunks are ranked near the top of the context passed to the generator, rewarding good retrieval ordering.",
  },
  {
    name: "Contextual Recall",
    desc: "Measures whether the retrieved context contains everything from the expected/ground-truth answer — penalizes retrieval that misses necessary information.",
  },
];

export function AboutView() {
  return (
    <div className="h-full overflow-y-auto p-5">
      <div className="mx-auto max-w-4xl space-y-8 pb-10">
        <section>
          <div className="mb-2 flex items-center gap-2">
            <BookOpen size={18} className="text-[color:var(--color-signal)]" />
            <h2 className="text-sm font-semibold">System Overview &amp; Data Isolation Philosophy</h2>
          </div>
          <p className="text-sm leading-relaxed text-[color:var(--color-ink-dim)]">
            This workbench benchmarks five distinct retrieval paradigms against the same question, all reading from
            a <span className="text-[color:var(--color-ink)]">read-only local ChromaDB vector store</span> (embedded
            with <code className="rounded bg-[color:var(--color-card)] px-1 py-0.5 text-[12px]">all-MiniLM-L6-v2</code>{" "}
            and reranked with FlashRank's <code className="rounded bg-[color:var(--color-card)] px-1 py-0.5 text-[12px]">ms-marco-MiniLM-L-12-v2</code>)
            and an offline-built <span className="text-[color:var(--color-ink)]">LightRAG knowledge graph</span> (NetworkX,
            constructed via Ollama). Only LightRAG ever touches the knowledge graph — every other mode is confined to
            ChromaDB chunks, plus live Tavily web search for Agentic RAG alone. Nothing in this app can ingest, create,
            or delete a collection or graph; every selector here is read-only by design.
          </p>
          <p className="mt-2 text-sm leading-relaxed text-[color:var(--color-ink-dim)]">
            Behind the scenes, a rate-limit-hardened <span className="text-[color:var(--color-ink)]">DeepEval</span> harness
            grades every mode on the same questions, with SHA-256 response caching to avoid repeat API spend and an
            ordered <span className="text-[color:var(--color-ink)]">Groq → Gemini → Ollama&nbsp;:cloud</span> failover
            chain so a rate-limited or quota-exhausted judge model never stalls a benchmark run.
          </p>
        </section>

        <section>
          <h2 className="mb-3 text-sm font-semibold">The Five RAG Strategies</h2>
          <div className="space-y-4">
            <ArchCard
              icon={Database}
              title="Traditional Vector RAG"
              subtitle="Pure dense similarity search"
              flow={["query", "dense search (k=5)", "synthesis", "[Passage X] citations"]}
              strengths="Direct semantic lookups where the answer paraphrases a single well-matched passage — fast, simple, predictable."
              weaknesses="Misses exact keyword/acronym matches dense embeddings don't capture well, and has no self-correction if retrieval quality is poor."
              accent="bg-[color:var(--color-signal)]/15 text-[color:var(--color-signal)]"
            />
            <ArchCard
              icon={GitMerge}
              title="Hybrid Search RAG"
              subtitle="Dense + BM25 + RRF + FlashRank"
              flow={["dense (k=10)", "BM25 (k=10)", "RRF fusion", "FlashRank rerank (k=4)", "synthesis"]}
              strengths="Sparse BM25Okapi lexical matching complements dense embeddings for exact terms, formulas and acronyms. Reciprocal Rank Fusion — RRF(d) = Σ 1/(k+rank) — merges the two incompatible score scales using only rank position, so no min-max or z-score calibration is needed. FlashRank's cross-encoder then refines the fused candidates into a tight top-k."
              weaknesses="More moving parts than traditional RAG; still has no self-correction loop if the fused-and-reranked context is genuinely insufficient."
              accent="bg-[color:var(--color-violet)]/15 text-[color:var(--color-violet)]"
            />
            <ArchCard
              icon={RefreshCcw}
              title="Self-Reflective / Corrective RAG"
              subtitle="LangGraph closed-loop verification"
              flow={["retrieve+rerank", "grade_relevance", "rewrite_query?", "generate", "grade_groundedness", "pop+retry?"]}
              strengths="Two-stage closed-loop verification: a pre-generation Relevance Grader can trigger an automated query rewrite (max 2 retries) when retrieval looks insufficient, and a post-generation Hallucination/Groundedness Auditor can force a regeneration — rejected drafts are popped from message history before each corrective retry so the model never anchors on its own bad answer."
              weaknesses="More LLM calls per question (grading + rewriting), so higher latency and token cost than a single-pass mode; bounded retries mean it can still return an imperfect answer if nothing in the collection is relevant."
              accent="bg-[color:var(--color-amber)]/15 text-[color:var(--color-amber)]"
            />
            <ArchCard
              icon={Bot}
              title="Autonomous Agentic RAG"
              subtitle="ReAct tool-use + web + audit"
              flow={["agent (reason)", "tool_call", "tool_node", "…loop (≤4 calls)…", "groundedness_audit", "pop+retry?"]}
              strengths="Dynamically routes between local textbook chunks (search_textbook_chunks) and live web search (search_web_tavily) — ideal for questions bridging classical theory with fast-moving modern libraries (e.g. how a specific framework implements a decades-old algorithm), capped by a final groundedness audit."
              weaknesses="Least predictable latency (tool-call budget, web search variance) and the most providers/tokens in play, so it's the mode most exposed to rate limits across a long session."
              accent="bg-[color:var(--color-emerald)]/15 text-[color:var(--color-emerald)]"
            />
            <ArchCard
              icon={Share2}
              title="LightRAG Knowledge Graph"
              subtitle="Dual-level entity/relation traversal"
              flow={["dual-level keyword extraction", "entity matching", "1-2 hop traversal", "synthesis"]}
              strengths="Entity-relation graph extraction lets low-level entity-neighborhood traversal and high-level thematic synthesis work together, which is why graph traversal excels at multi-hop architectural reasoning (e.g. tracing how attention, residuals, and normalization interrelate across a Transformer block) that chunk-based retrieval alone tends to flatten."
              weaknesses="Entirely dependent on how well the offline graph-building pass extracted entities/relations from the source material; if a concept was never captured as a node, graph traversal can't surface it, and it can't resolve arbitrary numeric/formula lookups the way BM25 can."
              accent="bg-[color:var(--color-coral)]/15 text-[color:var(--color-coral)]"
            />
          </div>
        </section>

        <section>
          <h2 className="mb-3 text-sm font-semibold">DeepEval Metrics Glossary</h2>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {METRICS.map((m) => (
              <div key={m.name} className="rounded-lg border border-[color:var(--color-border)] bg-[color:var(--color-card)] p-3.5">
                <p className="mb-1 text-xs font-semibold text-[color:var(--color-ink)]">{m.name}</p>
                <p className="text-xs leading-relaxed text-[color:var(--color-ink-dim)]">{m.desc}</p>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
