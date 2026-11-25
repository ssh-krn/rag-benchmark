# Advanced Multi-Strategy RAG Platform — Backend

Python + FastAPI backend only (no frontend code). Implements 5 selectable RAG
modes plus a rate-limit-resilient DeepEval evaluation suite, per the spec.

## Setup

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in GROQ_API_KEY / GEMINI_API_KEY / TAVILY_API_KEY
```

Pre-populate `./chroma_db` (ChromaDB) and `./lightrag_graphs/*.graphml` yourself
offline — this backend never ingests data via its API. Two standalone CLI
scripts handle this (neither is imported by `app/`, so they don't violate the
no-ingestion-endpoint constraint):

```bash
# 1. Chunk PDFs/txt/md and load them into ChromaDB. Reuses the backend's own
#    embedding singleton (app/core/embeddings.py) so ingestion-time and
#    query-time vectors are produced by the exact same model/config.
python scripts/offline_ingest_chroma.py --input ./docs/machine-learning \
    --collection machine-learning
# add --dry-run to preview chunking first, --reset to rebuild from scratch

# 2. Build a LightRAG knowledge graph from an existing ChromaDB collection
#    (run this after step 1).
python scripts/offline_build_graph.py --collection machine-learning \
    --output ./lightrag_graphs/machine-learning_graph.graphml
```

For Ollama `:cloud` models, either run `ollama signin` once (local daemon
proxy, leave `OLLAMA_API_KEY` blank) or set `OLLAMA_BASE_URL=https://ollama.com`
with `OLLAMA_API_KEY` set.

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

On startup, a pre-flight check hits Groq's `/models`, Gemini's `models.list`,
and Ollama's `/api/tags` + a 1-token `:cloud` probe, building the verified
`active_judge_chain`. See `GET /api/health` for the live report and any active
cooldowns.

## Key endpoints

- `GET /api/health` — provider/model verification report, ChromaDB + graph status, cooldowns
- `GET /api/collections` — pre-populated Chroma collections + LightRAG graphs
- `POST /api/chat` / `POST /api/chat/stream` — run any of the 5 RAG modes
- `POST /api/evals/run` / `GET /api/evals/runs/{run_id}` — DeepEval benchmark, resumable

## Notes on scope

- No `/ingest` or upload endpoints exist by design — bring your own pre-built
  ChromaDB store and `.graphml` graph.
- `lightrag` is the only mode touching the NetworkX graph; all others
  (`traditional_rag`, `hybrid_rag`, `corrective_rag`, `agentic_rag`) read only
  from ChromaDB (+ Tavily, `agentic_rag` only).
- `app/core/resilient_llm.py` is the single place that classifies 429s as
  transient vs. quota-exhausted and drives Groq → Gemini → Ollama(:cloud)
  failover for both inference and DeepEval judging.
