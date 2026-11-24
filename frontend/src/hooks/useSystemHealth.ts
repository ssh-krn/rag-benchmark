// src/hooks/useSystemHealth.ts
import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { CollectionsResponse, HealthResponse } from "@/types/rag";

export function useSystemHealth(pollIntervalMs = 30_000) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [collections, setCollections] = useState<CollectionsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [h, c] = await Promise.all([api.getHealth(), api.getCollections()]);
      setHealth(h);
      setCollections(c);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reach backend");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, pollIntervalMs);
    return () => window.clearInterval(id);
  }, [refresh, pollIntervalMs]);

  return { health, collections, loading, error, refresh };
}
