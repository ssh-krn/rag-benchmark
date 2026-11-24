// src/lib/api.ts
// Typed REST client for the read-only FastAPI backend. No upload/delete calls
// exist anywhere in this module by design.

import type {
  ChatRequest,
  ChatResponse,
  CheckpointReport,
  CollectionsResponse,
  EvalRunRequest,
  EvalRunStartResponse,
  HealthResponse,
} from "@/types/rag";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new ApiError(body || `Request to ${path} failed with ${res.status}`, res.status);
  }
  return (await res.json()) as T;
}

export const api = {
  getHealth: () => request<HealthResponse>("/api/health"),

  getCollections: () => request<CollectionsResponse>("/api/collections"),

  chat: (payload: ChatRequest) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  startEvalRun: (payload: EvalRunRequest) =>
    request<EvalRunStartResponse>("/api/evals/run", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  getEvalRun: (runId: string) => request<CheckpointReport>(`/api/evals/runs/${runId}`),
};

export { ApiError };
