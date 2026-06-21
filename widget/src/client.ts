import type {
  ChatRequest,
  ChatResponse,
  FeedbackRequest,
  HealthResponse,
  LeadRequest,
} from "./types";

export interface ChatClientOptions {
  baseUrl: string;
}

export interface ChatCallbacks {
  onToken?: (token: string) => void;
  onSources?: (sources: string[]) => void;
  onLeadSuggestion?: (data: unknown) => void;
}

/** Parse SSE lines from a text buffer. Returns parsed events and unprocessed rest. */
function parseSSEBuffer(buffer: string): {
  events: Array<{ event: string; data: unknown }>;
  rest: string;
} {
  const events: Array<{ event: string; data: unknown }> = [];
  let currentEvent = "";
  let currentData = "";

  const lines = buffer.split("\n");
  let lastConsumed = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;

    if (line.startsWith("event: ")) {
      currentEvent = line.slice(7).trim();
    } else if (line.startsWith("data: ")) {
      currentData += line.slice(6);
    } else if (line === "") {
      // Empty line = end of event
      if (currentEvent && currentData) {
        try {
          events.push({ event: currentEvent, data: JSON.parse(currentData) });
        } catch {
          events.push({ event: currentEvent, data: currentData });
        }
      }
      currentEvent = "";
      currentData = "";
      lastConsumed = i + 1;
    }
  }

  // If we still have state after the last consumed line, keep it as rest
  const restLines = lines.slice(lastConsumed);
  const rest = restLines.join("\n");

  return { events, rest };
}

export function createChatClient(options: ChatClientOptions) {
  const { baseUrl } = options;
  const headers = { "Content-Type": "application/json" };

  async function health(): Promise<HealthResponse> {
    const res = await fetch(`${baseUrl}/health`);
    if (!res.ok) {
      throw new Error(`Health check failed: ${res.status}`);
    }
    return res.json() as Promise<HealthResponse>;
  }

  async function chat(
    req: ChatRequest & Partial<ChatCallbacks>
  ): Promise<ChatResponse> {
    const { onToken, onSources, onLeadSuggestion, ...cleanReq } = req;
    const res = await fetch(`${baseUrl}/chat`, {
      method: "POST",
      headers,
      body: JSON.stringify(cleanReq),
    });

    if (!res.ok) {
      throw new Error(`Chat request failed: ${res.status}`);
    }

    const reader = res.body?.getReader();
    if (!reader) {
      throw new Error("Response body is not readable");
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let answerTokens: string[] = [];
    let sources: string[] = [];
    let model = "";
    let chunksUsed = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const { events, rest } = parseSSEBuffer(buffer);
      buffer = rest;

      for (const evt of events) {
        switch (evt.event) {
          case "answer": {
            const data = evt.data as { token: string };
            answerTokens.push(data.token);
            onToken?.(data.token);
            break;
          }
          case "sources": {
            const data = evt.data as { sources: string[] };
            sources = data.sources;
            onSources?.(data.sources);
            break;
          }
          case "lead_suggestion": {
            onLeadSuggestion?.(evt.data);
            break;
          }
          case "done": {
            const data = evt.data as {
              model: string;
              chunks_used: number;
            };
            model = data.model;
            chunksUsed = data.chunks_used;
            break;
          }
          case "error": {
            const data = evt.data as { detail: string };
            throw new Error(data.detail);
          }
        }
      }
    }

    return {
      answer: answerTokens.join(""),
      sources,
      model,
      chunksUsed,
    };
  }

  async function feedback(req: FeedbackRequest): Promise<{ status: string }> {
    const res = await fetch(`${baseUrl}/feedback`, {
      method: "POST",
      headers,
      body: JSON.stringify(req),
    });
    if (!res.ok) {
      throw new Error(`Feedback request failed: ${res.status}`);
    }
    return res.json() as Promise<{ status: string }>;
  }

  async function lead(req: LeadRequest): Promise<{ status: string }> {
    const res = await fetch(`${baseUrl}/lead`, {
      method: "POST",
      headers,
      body: JSON.stringify(req),
    });
    if (!res.ok) {
      throw new Error(`Lead request failed: ${res.status}`);
    }
    return res.json() as Promise<{ status: string }>;
  }

  return { health, chat, feedback, lead };
}
