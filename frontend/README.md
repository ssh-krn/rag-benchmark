# RAG Research Workbench — Frontend

React 19 + TypeScript + Vite + Tailwind CSS v4 frontend for the Advanced
Multi-Strategy RAG & DeepEval Benchmarking Platform. Connects to an existing
FastAPI backend — **no backend/server code lives here**, and there is no
upload/delete UI anywhere (collections and graphs are read-only selectors
sourced from `GET /api/collections`).

## Setup

```bash
cd frontend
npm install
cp .env.example .env   # set VITE_API_BASE_URL if your backend isn't on :8000
npm run dev
```

Verified: `npx tsc -b` and `npx vite build` both pass cleanly.

## Structure

```
src/
├── types/rag.ts        # Strict interfaces for REST responses + SSE events
├── lib/api.ts           # Typed REST client (health, collections, chat, evals)
├── lib/sse.ts            # Chunk-safe fetch+ReadableStream SSE parser
├── hooks/useRagStream.ts        # Streaming chat state, TTFT/latency, traces
├── hooks/useSystemHealth.ts     # /health + /collections polling
├── hooks/useEvalRunner.ts       # DeepEval run trigger/resume + 3s polling
└── components/           # HeaderBar, HealthModal, ChatPanel, SourceInspector,
                           # LightRagGraphView, CompareView, EvalDashboard, AboutView
```

## API contract this frontend expects

This has been reconciled against the actual FastAPI backend (`app/api/routes.py`,
`app/schemas/chat.py`, `app/schemas/evals.py`) rather than the original spec
draft. Three real gaps between "what a fully-featured frontend spec asks for"
and "what this backend currently sends" are called out below rather than
papered over with fabricated data.

- `GET /api/health`, `GET /api/collections` — match the backend exactly, field
  for field (see `src/types/rag.ts`).
- `POST /api/chat` — `{ query, mode, collection_name?, graph_name? }` →
  `ChatResponse` (`answer`, `mode`, `retrieved_context`, `execution_trace`,
  `matched_entities?`, `relationships?`, `subgraph_nodes?`, `subgraph_edges?`,
  `model_used?`, `provider_used?`). Used as the synchronous fallback if the
  SSE transport fails to open.
- `POST /api/chat/stream` — same request body, `text/event-stream` response
  with exactly five named frames (see `src/lib/sse.ts`'s `VALID_EVENTS` and
  the comment block above the SSE types in `src/types/rag.ts`):
  - `trace` — one `ExecutionStep` `{ node, description, detail? }` per
    LangGraph node, as it completes.
  - `context` — a bare `SourceMetadata[]` array (not wrapped in an object).
  - `token` — `{ text }`. **Known gap:** the backend currently generates the
    full answer synchronously and sends it as a single `token` frame rather
    than incremental deltas, so the chat bubble doesn't type character-by-
    character yet and TTFT ≈ total latency. `useRagStream` replaces rather
    than appends on this frame so it's forward-compatible with true
    incremental streaming if the backend adds it later.
  - `final` — the complete `ChatResponse`, identical shape to `POST /api/chat`.
  - `error` — `{ error, node? }`.
  - **Known gap:** there are no separate `tool_call` or `grade_result` frames.
    Tool execution and grader verdicts only exist as entries inside
    `execution_trace` (e.g. a step named `relevance_grader` whose description
    reads `"Relevance grade: RELEVANT"`). `useRagStream`'s `deriveGrades()`
    parses those descriptions into `GradeRecord`s for the grader badges; there
    isn't enough structured data on the wire to safely reconstruct per-tool
    call arguments, so `ToolCallRecord`/`ToolCallPill` stay defined for
    forward compatibility but render nothing against live traffic today.
  - **Known gap:** `total_latency_ms`/`ttft_ms`/`retrieval_ms`/`generation_ms`/
    `fallback_occurred` aren't sent by the backend. Total latency and TTFT are
    computed client-side from wall-clock time instead; `retry_count` is
    derived by counting `rewrite_query`/`pop_rejected_draft`/`pop_and_retry`
    steps in the trace; retrieval-vs-generation split and fallback detection
    are left `undefined` (the latency waterfall and fallback badge hide
    themselves gracefully when the fields are absent) rather than guessed at.
- `POST /api/evals/run`, `GET /api/evals/runs/{run_id}` — match the backend's
  `EvalRunRequest`/`CheckpointReport` schemas exactly. The dashboard polls the
  latter every 3s while a run is `pending`/`running`. **Known gap:** the
  backend's `CheckpointReport` doesn't currently include per-test-case
  results, so the per-question drilldown table only renders if a future
  backend revision adds the optional `results: TestCaseResult[]` field (the
  type is already defined and wired up, just unused while the field is
  absent). `active_provider` is likewise always `null` today since
  `build_report()` is never called with it — the telemetry card shows `—`.

If the backend's SSE event names or field names change, `src/types/rag.ts`
and `src/lib/sse.ts`'s `VALID_EVENTS` are the two files to update — they are
the single source of truth for the wire contract.
