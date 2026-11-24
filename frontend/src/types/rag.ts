// src/types/rag.ts
// Strict TypeScript interfaces mirroring the FastAPI backend's request/response
// schemas and the /api/chat/stream SSE event protocol.

export type RagMode =
  | "traditional_rag"
  | "hybrid_rag"
  | "corrective_rag"
  | "agentic_rag"
  | "lightrag";

export const RAG_MODES: RagMode[] = [
  "traditional_rag",
  "hybrid_rag",
  "corrective_rag",
  "agentic_rag",
  "lightrag",
];

export const RAG_MODE_LABELS: Record<RagMode, string> = {
  traditional_rag: "Traditional RAG",
  hybrid_rag: "Hybrid Search",
  corrective_rag: "Corrective RAG",
  agentic_rag: "Agentic RAG",
  lightrag: "LightRAG",
};

export const RAG_MODE_SUBTITLES: Record<RagMode, string> = {
  traditional_rag: "Pure Dense Vector",
  hybrid_rag: "Dense + BM25 + RRF + FlashRank",
  corrective_rag: "Self-Reflective Relevance & Hallucination Loop",
  agentic_rag: "Autonomous ChromaDB + Tavily Web Tools + Audit",
  lightrag: "Ollama Knowledge Graph Dual-Level Traversal",
};

export type RetrievalMethod =
  | "dense"
  | "bm25"
  | "hybrid_rrf"
  | "reranked"
  | "web_search"
  | "graph_traversal";

// ---------------------------------------------------------------------------
// /api/collections
// ---------------------------------------------------------------------------

export interface CollectionInfo {
  name: string;
  chunk_count: number | null;
  error?: string;
}

export interface GraphInfo {
  name: string;
  nodes?: number;
  edges?: number;
  error?: string;
}

export interface CollectionsResponse {
  collections: CollectionInfo[];
  lightrag_graphs: GraphInfo[];
}

// ---------------------------------------------------------------------------
// /api/health
// ---------------------------------------------------------------------------

export interface ProviderReport {
  provider: string;
  reachable: boolean;
  verified_models: string[];
  rejected_models: Record<string, string>;
  error?: string | null;
}

export interface ActiveJudgeChainEntry {
  provider: string;
  model: string;
}

export interface PreflightVerification {
  groq: ProviderReport;
  gemini: ProviderReport;
  ollama_cloud: ProviderReport;
  active_judge_chain: ActiveJudgeChainEntry[];
}

export interface CooldownEntry {
  provider: string;
  model: string;
  seconds_remaining: number;
  reason: string;
}

export interface ChromaHealth {
  status: string;
  collections?: string[];
  path?: string;
  error?: string;
}

export interface LightRagGraphHealth {
  graph_dir: string;
  available_graphs: GraphInfo[];
}

export interface HealthResponse {
  status: string;
  chromadb: ChromaHealth;
  lightrag_graph: LightRagGraphHealth;
  active_cooldowns: CooldownEntry[];
  preflight_verification: PreflightVerification;
}

// ---------------------------------------------------------------------------
// Retrieved context / source metadata
// ---------------------------------------------------------------------------

export interface SourceMetadata {
  chunk_id?: string | null;
  source?: string | null;
  page?: number | null;
  collection_name?: string | null;
  score?: number | null;
  retrieval_method?: RetrievalMethod | null;
  text?: string | null;

  // Hybrid-specific
  dense_rank?: number | null;
  bm25_rank?: number | null;
  rrf_score?: number | null;
  rerank_score?: number | null;

  // Web-specific
  url?: string | null;

  // Graph-specific
  entities?: string[] | null;
  relations?: Record<string, unknown>[] | null;
}

export interface GraphNode {
  id: string;
  entity_type?: string;
  description?: string;
  [key: string]: unknown;
}

export interface GraphEdge {
  source: string;
  target: string;
  relation?: string;
  weight?: number;
}

// ---------------------------------------------------------------------------
// Execution trace
// ---------------------------------------------------------------------------

export interface ExecutionStep {
  node: string;
  description: string;
  detail?: Record<string, unknown> | null;
  /** Client-side enrichment: wall-clock ms this node took, computed from SSE timestamps. */
  durationMs?: number;
  timestamp?: number;
}

// ---------------------------------------------------------------------------
// /api/chat request/response
// ---------------------------------------------------------------------------

// NOTE: the backend's Pydantic schema is `{ query, mode, collection_name?,
// graph_name?, conversation_id?, chat_history? }` — there is no `max_retries`
// field (corrective/agentic retry caps are server-side settings, not
// per-request). `query` is the field name actually read by FastAPI.
export interface ChatRequest {
  query: string;
  mode: RagMode;
  collection_name?: string;
  graph_name?: string;
}

export interface ChatResponse {
  answer: string;
  mode: RagMode;
  retrieved_context: SourceMetadata[];
  execution_trace: ExecutionStep[];
  matched_entities?: string[] | null;
  relationships?: GraphEdge[] | null;
  subgraph_nodes?: GraphNode[] | null;
  subgraph_edges?: GraphEdge[] | null;
  model_used?: string | null;
  provider_used?: string | null;
}

// ---------------------------------------------------------------------------
// SSE event payloads (/api/chat/stream)
//
// The backend (app/api/routes.py) emits exactly five named frames:
//   event: trace   -> one ExecutionStep at a time, as each LangGraph node completes
//   event: context -> a bare SourceMetadata[] array, once retrieval finishes
//   event: token   -> { text } — currently the FULL answer in one frame, since
//                      the backend generates synchronously and streams the
//                      result rather than token-by-token (see README for the
//                      follow-up needed for true incremental streaming)
//   event: final   -> the complete ChatResponse (same shape as POST /api/chat)
//   event: error   -> { error }
//
// There is no per-tool-call or per-grader-verdict event on the wire. Tool
// execution and grader verdicts are only visible as entries inside
// `execution_trace` (e.g. a step named "relevance_grader" whose description
// reads "Relevance grade: RELEVANT"). The client derives GradeRecord entries
// from those trace steps (see hooks/useRagStream.ts `deriveGrades`); there is
// not enough structured data on the wire to safely derive per-tool-call
// argument/result cards, so ToolCallRecord stays defined (for forward
// compatibility / manual testing) but is not populated from live traffic.
// ---------------------------------------------------------------------------

export type TraceFrame = ExecutionStep;
export type ContextFrame = SourceMetadata[];
export type FinalFrame = ChatResponse;

export interface TokenEvent {
  text: string;
}

export interface ErrorEvent {
  error: string;
  node?: string;
}

export type SSEEventName = "trace" | "context" | "token" | "final" | "error";

export interface ParsedSSEFrame {
  event: SSEEventName;
  data: unknown;
  raw: string;
  receivedAt: number;
}

// ---------------------------------------------------------------------------
// Chat UI message model (client-side only)
// ---------------------------------------------------------------------------

/** Not emitted by the current backend — kept for forward compatibility if a
 * future backend revision adds structured per-tool-call frames. */
export interface ToolCallRecord {
  id: string;
  tool: string;
  args: Record<string, unknown>;
  result_chars?: number;
  result_chunks?: number;
  timestamp: number;
}

/** Derived client-side from `execution_trace` steps whose node name matches a
 * known grader/auditor pattern — see `deriveGrades` in useRagStream.ts. */
export interface GradeRecord {
  id: string;
  grader: string;
  verdict: string; // "relevant" | "not relevant" | "grounded" | "hallucinated" | "unknown"
  retry_attempt: number;
  reasoning?: string;
  timestamp: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  mode?: RagMode;
  collectionName?: string;
  graphName?: string;
  isStreaming?: boolean;
  error?: string;

  retrievedContext?: SourceMetadata[];
  executionTrace?: ExecutionStep[];
  toolCalls?: ToolCallRecord[];
  grades?: GradeRecord[];
  rawFrames?: ParsedSSEFrame[];

  matchedEntities?: string[];
  relationships?: GraphEdge[];
  subgraphNodes?: GraphNode[];
  subgraphEdges?: GraphEdge[];

  totalLatencyMs?: number;
  ttftMs?: number;
  retrievalMs?: number;
  generationMs?: number;
  modelUsed?: string;
  providerUsed?: string;
  fallbackOccurred?: boolean;
  retryCount?: number;
}

// ---------------------------------------------------------------------------
// DeepEval runs
// ---------------------------------------------------------------------------

export type EvalStatus = "pending" | "running" | "completed" | "failed";

export interface EvalRunRequest {
  dataset_path: string;
  target_modes?: RagMode[];
  resume_run_id?: string;
  batch_size?: number;
  sleep_between_batches?: number;
}

export interface EvalRunStartResponse {
  run_id: string;
  status: string;
}

export interface MetricResult {
  metric_name: string;
  score?: number | null;
  reason?: string | null;
  success?: boolean | null;
  error?: string | null;
}

export interface ModeMetricsSummary {
  mode: string;
  count: number;
  avg_faithfulness?: number | null;
  avg_answer_relevancy?: number | null;
  avg_contextual_precision?: number | null;
  avg_contextual_recall?: number | null;
}

export interface TestCaseResult {
  golden_id: string;
  mode: string;
  question: string;
  cache_hit: boolean;
  rag_output?: string | null;
  retrieved_contexts?: string[] | null;
  metrics: MetricResult[];
  provider_used?: string | null;
  model_used?: string | null;
  error?: string | null;
}

export interface CheckpointReport {
  run_id: string;
  status: EvalStatus;
  total: number;
  completed: number;
  failed: number;
  cached_hits: number;
  active_provider?: string | null;
  metrics_summary: ModeMetricsSummary[];
  started_at?: string | null;
  updated_at?: string | null;
  error?: string | null;
  /** Optional: per-test-case results for drilldown drawers, if the backend includes them. */
  results?: TestCaseResult[] | null;
}
