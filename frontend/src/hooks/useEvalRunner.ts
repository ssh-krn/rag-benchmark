// src/hooks/useEvalRunner.ts
import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { CheckpointReport, EvalRunRequest, RagMode } from "@/types/rag";

const POLL_MS = 3000;

export function useEvalRunner() {
  const [runId, setRunId] = useState<string | null>(null);
  const [report, setReport] = useState<CheckpointReport | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pastRunIds, setPastRunIds] = useState<string[]>(() => {
    try {
      return JSON.parse(localStorage.getItem("rag.eval.pastRuns") || "[]");
    } catch {
      return [];
    }
  });
  const pollRef = useRef<number | null>(null);

  const persistRunId = useCallback((id: string) => {
    setPastRunIds((prev) => {
      const next = [id, ...prev.filter((r) => r !== id)].slice(0, 20);
      localStorage.setItem("rag.eval.pastRuns", JSON.stringify(next));
      return next;
    });
  }, []);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const poll = useCallback(async (id: string) => {
    try {
      const r = await api.getEvalRun(id);
      setReport(r);
      if (r.status === "completed" || r.status === "failed") {
        stopPolling();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to poll run status");
    }
  }, [stopPolling]);

  const startPolling = useCallback(
    (id: string) => {
      stopPolling();
      poll(id);
      pollRef.current = window.setInterval(() => poll(id), POLL_MS);
    },
    [poll, stopPolling],
  );

  const startRun = useCallback(
    async (opts: {
      datasetPath: string;
      targetModes: RagMode[];
      batchSize: number;
      sleepBetweenBatches: number;
      resumeRunId?: string;
    }) => {
      setStarting(true);
      setError(null);
      try {
        const payload: EvalRunRequest = {
          dataset_path: opts.datasetPath,
          target_modes: opts.targetModes,
          batch_size: opts.batchSize,
          sleep_between_batches: opts.sleepBetweenBatches,
          resume_run_id: opts.resumeRunId,
        };
        const res = await api.startEvalRun(payload);
        setRunId(res.run_id);
        persistRunId(res.run_id);
        startPolling(res.run_id);
      } catch (err) {
        setError(err instanceof ApiError ? err.message : "Failed to start evaluation run");
      } finally {
        setStarting(false);
      }
    },
    [persistRunId, startPolling],
  );

  const resumeRun = useCallback(
    (id: string) => {
      setRunId(id);
      persistRunId(id);
      startPolling(id);
    },
    [persistRunId, startPolling],
  );

  useEffect(() => () => stopPolling(), [stopPolling]);

  return { runId, report, starting, error, pastRunIds, startRun, resumeRun, stopPolling };
}
