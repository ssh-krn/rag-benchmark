// src/lib/sse.ts
// Chunk-safe SSE parser for POST-initiated streams (the EventSource API only
// supports GET, so /api/chat/stream is consumed via fetch + ReadableStream).

import type { ParsedSSEFrame, SSEEventName } from "@/types/rag";

export interface SSEStreamOptions {
  url: string;
  body: unknown;
  signal?: AbortSignal;
  onFrame: (frame: ParsedSSEFrame) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onNetworkError?: (err: unknown) => void;
}

// Must match the five named frames the backend actually emits
// (app/api/routes.py's /api/chat/stream event_generator).
const VALID_EVENTS: SSEEventName[] = ["trace", "context", "token", "final", "error"];

/**
 * Parses a raw `text/event-stream` buffer, splitting on blank-line-delimited
 * frames and tolerating partial chunks across multiple reads.
 */
function parseBuffer(buffer: string): { frames: ParsedSSEFrame[]; remainder: string } {
  const frames: ParsedSSEFrame[] = [];
  const blocks = buffer.split("\n\n");
  // The final block may be incomplete (no trailing blank line yet) — keep it as remainder.
  const remainder = blocks.pop() ?? "";

  for (const block of blocks) {
    if (!block.trim()) continue;
    let eventName: SSEEventName = "token";
    const dataLines: string[] = [];

    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) {
        const name = line.slice(6).trim() as SSEEventName;
        if (VALID_EVENTS.includes(name)) eventName = name;
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      }
    }

    const raw = dataLines.join("\n");
    let data: unknown = raw;
    try {
      data = JSON.parse(raw);
    } catch {
      // keep raw string if it isn't JSON
    }

    frames.push({ event: eventName, data, raw, receivedAt: Date.now() });
  }

  return { frames, remainder };
}

export async function streamSSE(options: SSEStreamOptions): Promise<void> {
  const { url, body, signal, onFrame, onOpen, onClose, onNetworkError } = options;

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal,
    });

    if (!res.ok || !res.body) {
      throw new Error(`Stream request failed with status ${res.status}`);
    }

    onOpen?.();

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const { frames, remainder } = parseBuffer(buffer);
      buffer = remainder;
      for (const frame of frames) onFrame(frame);
    }

    // Flush anything left in the buffer (server closed without trailing \n\n)
    if (buffer.trim()) {
      const { frames } = parseBuffer(buffer + "\n\n");
      for (const frame of frames) onFrame(frame);
    }

    onClose?.();
  } catch (err) {
    if ((err as Error)?.name === "AbortError") {
      onClose?.();
      return;
    }
    onNetworkError?.(err);
  }
}
