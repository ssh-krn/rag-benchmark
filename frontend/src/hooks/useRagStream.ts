// src/hooks/useRagStream.ts
// Manages streaming tokens, trace steps, derived grader verdicts and
// retrieved-context metadata for the chat workbench.
//
// Reconciled against the backend's actual /api/chat/stream contract
// (app/api/routes.py): frames are `trace` (one ExecutionStep), `context`
// (a bare SourceMetadata[] array), `token` ({ text }, currently the full
// answer in one frame), `final` (the complete ChatResponse) and `error`
// ({ error }). See the comment block above the SSE types in types/rag.ts
// for the full rationale. Falls back to POST /api/chat if the SSE
// transport fails to open at all.

import { useCallback, useRef, useState } from "react";
import { API_BASE_URL, api } from "@/lib/api";
import { streamSSE } from "@/lib/sse";
import type {
  ChatMessage,
  ChatResponse,
  ContextFrame,
  ErrorEvent,
  ExecutionStep,
  GradeRecord,
  ParsedSSEFrame,
  RagMode,
  TokenEvent,
  TraceFrame,
} from "@/types/rag";

function uid(prefix = "msg"): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

/** Trace step node names that represent a self-correction retry, used to
 * compute a client-side retry count since the backend doesn't send one. */
const RETRY_NODE_PATTERN = /rewrite_query|pop_rejected_draft|pop_and_retry/i;

/** Node names that represent a relevance/groundedness grader or auditor. */
const GRADER_NODE_PATTERNS: { match: RegExp; grader: string }[] = [
  { match: /relevance_grader/i, grader: "relevance" },
  { match: /groundedness_grader/i, grader: "groundedness" },
  { match: /groundedness_audit/i, grader: "groundedness" },
];

const VERDICT_PATTERN = /\b(GROUNDED|HALLUCINATED|NOT RELEVANT|RELEVANT)\b/i;

/** Derives GradeRecord[] from whatever execution_trace steps are available so
 * far. Recomputed (not appended) each time the trace grows, since backend
 * step descriptions are the only source of grader verdicts on the wire. */
function deriveGrades(steps: ExecutionStep[]): GradeRecord[] {
  const counters: Record<string, number> = {};
  const records: GradeRecord[] = [];

  steps.forEach((step, i) => {
    const found = GRADER_NODE_PATTERNS.find((p) => p.match.test(step.node));
    if (!found) return;
    const verdictMatch = step.description.match(VERDICT_PATTERN);
    const verdict = verdictMatch ? verdictMatch[1].toLowerCase() : "unknown";
    const attempt = counters[found.grader] ?? 0;
    records.push({
      id: `grade_${i}_${found.grader}`,
      grader: found.grader,
      verdict,
      retry_attempt: attempt,
      reasoning: step.description,
      timestamp: step.timestamp ?? Date.now(),
    });
    counters[found.grader] = attempt + 1;
  });

  return records;
}

function countRetries(steps: ExecutionStep[]): number {
  return steps.filter((s) => RETRY_NODE_PATTERN.test(s.node)).length;
}

interface SendParams {
  question: string;
  mode: RagMode;
  collectionName?: string;
  graphName?: string;
}

export function useRagStream() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const startRef = useRef<number>(0);
  const firstTokenRef = useRef<number | null>(null);

  const updateAssistant = useCallback((id: string, patch: Partial<ChatMessage> | ((m: ChatMessage) => Partial<ChatMessage>)) => {
    setMessages((prev) =>
      prev.map((m) => {
        if (m.id !== id) return m;
        const delta = typeof patch === "function" ? patch(m) : patch;
        return { ...m, ...delta };
      }),
    );
  }, []);

  const clear = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setIsStreaming(false);
  }, []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setIsStreaming(false);
  }, []);

  const applyFinalResponse = useCallback(
    (assistantId: string, res: ChatResponse, totalLatencyMs: number) => {
      updateAssistant(assistantId, (m) => ({
        content: res.answer,
        isStreaming: false,
        error: undefined,
        retrievedContext: res.retrieved_context,
        executionTrace: res.execution_trace,
        grades: deriveGrades(res.execution_trace ?? []),
        matchedEntities: res.matched_entities ?? undefined,
        relationships: res.relationships ?? undefined,
        subgraphNodes: res.subgraph_nodes ?? undefined,
        subgraphEdges: res.subgraph_edges ?? undefined,
        modelUsed: res.model_used ?? undefined,
        providerUsed: res.provider_used ?? undefined,
        totalLatencyMs,
        ttftMs: m.ttftMs,
        retryCount: countRetries(res.execution_trace ?? []),
      }));
    },
    [updateAssistant],
  );

  const sendMessage = useCallback(
    async ({ question, mode, collectionName, graphName }: SendParams) => {
      const userMsg: ChatMessage = {
        id: uid("user"),
        role: "user",
        content: question,
        mode,
        collectionName,
        graphName,
      };
      const assistantId = uid("assistant");
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        mode,
        collectionName,
        graphName,
        isStreaming: true,
        executionTrace: [],
        toolCalls: [],
        grades: [],
        rawFrames: [],
        retrievedContext: [],
      };
      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setIsStreaming(true);

      startRef.current = performance.now();
      firstTokenRef.current = null;
      const controller = new AbortController();
      abortRef.current = controller;

      let streamFailedHard = false;
      let gotFinalFrame = false;

      const requestBody = {
        query: question,
        mode,
        collection_name: collectionName,
        graph_name: graphName,
      };

      await streamSSE({
        url: `${API_BASE_URL}/api/chat/stream`,
        body: requestBody,
        signal: controller.signal,
        onFrame: (frame: ParsedSSEFrame) => {
          updateAssistant(assistantId, (m) => ({ rawFrames: [...(m.rawFrames ?? []), frame] }));

          switch (frame.event) {
            case "trace": {
              const step = frame.data as TraceFrame;
              updateAssistant(assistantId, (m) => {
                const nextTrace = [...(m.executionTrace ?? []), { ...step, timestamp: step.timestamp ?? Date.now() }];
                return { executionTrace: nextTrace, grades: deriveGrades(nextTrace) };
              });
              break;
            }
            case "context": {
              const sources = frame.data as ContextFrame;
              updateAssistant(assistantId, { retrievedContext: Array.isArray(sources) ? sources : [] });
              break;
            }
            case "token": {
              if (firstTokenRef.current === null) {
                firstTokenRef.current = performance.now();
                updateAssistant(assistantId, { ttftMs: Math.round(firstTokenRef.current - startRef.current) });
              }
              const text = (frame.data as TokenEvent)?.text ?? "";
              // The backend currently sends the complete answer as a single
              // `token` frame rather than incremental deltas, so we replace
              // rather than append to avoid duplicating it if more than one
              // token frame is ever introduced server-side.
              updateAssistant(assistantId, (m) => ({ content: text.length > (m.content?.length ?? 0) ? text : m.content }));
              break;
            }
            case "final": {
              gotFinalFrame = true;
              const res = frame.data as ChatResponse;
              const totalLatencyMs = Math.round(performance.now() - startRef.current);
              applyFinalResponse(assistantId, res, totalLatencyMs);
              break;
            }
            case "error": {
              const d = frame.data as ErrorEvent;
              updateAssistant(assistantId, { error: d.error, isStreaming: false });
              break;
            }
          }
        },
        onNetworkError: () => {
          streamFailedHard = true;
        },
      });

      // Fallback: if the SSE transport itself failed to open/parse (not an
      // in-band `error` frame, and no `final` frame ever arrived), retry once
      // via the synchronous /api/chat endpoint.
      if (streamFailedHard && !gotFinalFrame) {
        try {
          const res = await api.chat(requestBody);
          applyFinalResponse(assistantId, res, Math.round(performance.now() - startRef.current));
        } catch (err) {
          updateAssistant(assistantId, {
            isStreaming: false,
            error: err instanceof Error ? err.message : "Both streaming and fallback requests failed.",
          });
        }
      }

      setIsStreaming(false);
    },
    [updateAssistant, applyFinalResponse],
  );

  const retryLast = useCallback(
    async (overrides?: Partial<SendParams>) => {
      const lastUser = [...messages].reverse().find((m) => m.role === "user");
      if (!lastUser) return;
      await sendMessage({
        question: lastUser.content,
        mode: overrides?.mode ?? lastUser.mode ?? "traditional_rag",
        collectionName: overrides?.collectionName ?? lastUser.collectionName,
        graphName: overrides?.graphName ?? lastUser.graphName,
      });
    },
    [messages, sendMessage],
  );

  return { messages, isStreaming, sendMessage, retryLast, clear, stop };
}
