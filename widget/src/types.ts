/** Shared types for the Appther chat widget API. */

export interface Turn {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  question: string;
  history?: Turn[];
}

export interface ChatResponse {
  answer: string;
  sources: string[];
  model: string;
  chunksUsed: number;
}

export interface SSEChatEvent {
  event: "answer" | "sources" | "done" | "lead_suggestion" | "error";
  data: unknown;
}

export interface FeedbackRequest {
  question: string;
  answer: string;
  thumbs_up: boolean;
  chunks: Array<{ chunk_id: string; url: string; score: number }>;
  reason?: string | null;
}

export interface LeadRequest {
  name: string;
  email: string;
  question: string;
  phone?: string | null;
  message?: string | null;
}

export interface HealthResponse {
  status: string;
  service: string;
}
