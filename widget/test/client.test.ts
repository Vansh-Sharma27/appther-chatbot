import { describe, it, expect, beforeAll, afterEach, afterAll } from "vitest";
import { createChatClient } from "../src/client";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";

const API_BASE = "https://chat.appther.com";

const server = setupServer();

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe("createChatClient", () => {
  let client: ReturnType<typeof createChatClient>;

  beforeEach(() => {
    client = createChatClient({ baseUrl: API_BASE });
  });

  describe("health", () => {
    it("returns ok status from /health", async () => {
      server.use(
        http.get(`${API_BASE}/health`, () =>
          HttpResponse.json({ status: "ok", service: "appther-chatbot" })
        )
      );

      const result = await client.health();
      expect(result.status).toBe("ok");
      expect(result.service).toBe("appther-chatbot");
    });

    it("throws on network error", async () => {
      server.use(
        http.get(`${API_BASE}/health`, () => HttpResponse.error())
      );

      await expect(client.health()).rejects.toThrow();
    });
  });

  describe("chat (SSE stream)", () => {
    it("streams answer event and returns ChatResponse", async () => {
      server.use(
        http.post(`${API_BASE}/chat`, () => {
          const stream = [
            `event: answer\ndata: ${JSON.stringify({ token: "Appther offers ERP solutions." })}\n\n`,
            `event: sources\ndata: ${JSON.stringify({ sources: ["https://www.appther.com/services/erp"] })}\n\n`,
            `event: done\ndata: ${JSON.stringify({ model: "gemini-2.5-flash-lite", chunks_used: 4 })}\n\n`,
          ].join("");
          return new HttpResponse(stream, {
            headers: { "Content-Type": "text/event-stream" },
          });
        })
      );

      const result = await client.chat({ question: "What does Appther do?" });
      expect(result.answer).toBe("Appther offers ERP solutions.");
      expect(result.sources).toEqual(["https://www.appther.com/services/erp"]);
      expect(result.model).toBe("gemini-2.5-flash-lite");
      expect(result.chunksUsed).toBe(4);
    });

    it("streams answer tokens progressively via onToken callback", async () => {
      const tokens: string[] = [];

      server.use(
        http.post(`${API_BASE}/chat`, () => {
          const stream = [
            `event: answer\ndata: ${JSON.stringify({ token: "Appther " })}\n\n`,
            `event: answer\ndata: ${JSON.stringify({ token: "offers " })}\n\n`,
            `event: answer\ndata: ${JSON.stringify({ token: "ERP solutions." })}\n\n`,
            `event: sources\ndata: ${JSON.stringify({ sources: ["https://www.appther.com/services/erp"] })}\n\n`,
            `event: done\ndata: ${JSON.stringify({ model: "gemini-2.5-flash-lite", chunks_used: 4 })}\n\n`,
          ].join("");
          return new HttpResponse(stream, {
            headers: { "Content-Type": "text/event-stream" },
          });
        })
      );

      const result = await client.chat({
        question: "What does Appther do?",
        onToken: (token) => tokens.push(token),
      });

      expect(tokens).toEqual(["Appther ", "offers ", "ERP solutions."]);
      expect(result.answer).toBe("Appther offers ERP solutions.");
    });

    it("returns lead_suggestion event for no-answer", async () => {
      let leadSuggestion: unknown = null;

      server.use(
        http.post(`${API_BASE}/chat`, () => {
          const stream = [
            `event: answer\ndata: ${JSON.stringify({ token: "I don't have information about that." })}\n\n`,
            `event: sources\ndata: ${JSON.stringify({ sources: [] })}\n\n`,
            `event: lead_suggestion\ndata: ${JSON.stringify({ message: "Would you like us to reach out?" })}\n\n`,
            `event: done\ndata: ${JSON.stringify({ model: "gemini-2.5-flash-lite", chunks_used: 0 })}\n\n`,
          ].join("");
          return new HttpResponse(stream, {
            headers: { "Content-Type": "text/event-stream" },
          });
        })
      );

      const result = await client.chat({
        question: "Obscure question",
        onLeadSuggestion: (data) => {
          leadSuggestion = data;
        },
      });

      expect(leadSuggestion).toEqual({
        message: "Would you like us to reach out?",
      });
      expect(result.answer).toBe("I don't have information about that.");
    });

    it("rejects with error event", async () => {
      server.use(
        http.post(`${API_BASE}/chat`, () => {
          const stream = [
            `event: error\ndata: ${JSON.stringify({ detail: "Failed to generate answer" })}\n\n`,
          ].join("");
          return new HttpResponse(stream, {
            headers: { "Content-Type": "text/event-stream" },
          });
        })
      );

      await expect(
        client.chat({ question: "What?" })
      ).rejects.toThrow("Failed to generate answer");
    });

    it("sends history with the request", async () => {
      let sentBody: unknown = null;

      server.use(
        http.post(`${API_BASE}/chat`, async ({ request }) => {
          sentBody = await request.json();
          const stream = [
            `event: answer\ndata: ${JSON.stringify({ token: "Yes." })}\n\n`,
            `event: sources\ndata: ${JSON.stringify({ sources: [] })}\n\n`,
            `event: done\ndata: ${JSON.stringify({ model: "gemini-2.5-flash-lite", chunks_used: 1 })}\n\n`,
          ].join("");
          return new HttpResponse(stream, {
            headers: { "Content-Type": "text/event-stream" },
          });
        })
      );

      await client.chat({
        question: "Can they do CRM?",
        history: [
          { role: "user", content: "Tell me about Appther" },
          { role: "assistant", content: "Appther is an ERP company" },
        ],
      });

      expect(sentBody).toMatchObject({
        question: "Can they do CRM?",
        history: [
          { role: "user", content: "Tell me about Appther" },
          { role: "assistant", content: "Appther is an ERP company" },
        ],
      });
    });
  });

  describe("feedback", () => {
    it("sends thumbs up feedback", async () => {
      let sentBody: unknown = null;
      server.use(
        http.post(`${API_BASE}/feedback`, async ({ request }) => {
          sentBody = await request.json();
          return HttpResponse.json({ status: "ok" });
        })
      );

      const result = await client.feedback({
        question: "What is ERP?",
        answer: "ERP stands for...",
        thumbs_up: true,
        chunks: [{ chunk_id: "c1", url: "https://www.appther.com/faq", score: 0.95 }],
      });

      expect(result.status).toBe("ok");
      expect(sentBody).toMatchObject({
        question: "What is ERP?",
        thumbs_up: true,
      });
    });

    it("sends feedback with reason", async () => {
      server.use(
        http.post(`${API_BASE}/feedback`, () =>
          HttpResponse.json({ status: "ok" })
        )
      );

      const result = await client.feedback({
        question: "Pricing?",
        answer: "$5000",
        thumbs_up: false,
        reason: "Too vague",
        chunks: [],
      });

      expect(result.status).toBe("ok");
    });
  });

  describe("lead", () => {
    it("captures a lead", async () => {
      let sentBody: unknown = null;
      server.use(
        http.post(`${API_BASE}/lead`, async ({ request }) => {
          sentBody = await request.json();
          return HttpResponse.json({ status: "ok" });
        })
      );

      const result = await client.lead({
        name: "John Doe",
        email: "john@example.com",
        question: "I need ERP",
      });

      expect(result.status).toBe("ok");
      expect(sentBody).toMatchObject({
        name: "John Doe",
        email: "john@example.com",
        question: "I need ERP",
      });
    });

    it("captures lead with optional fields", async () => {
      server.use(
        http.post(`${API_BASE}/lead`, () =>
          HttpResponse.json({ status: "ok" })
        )
      );

      const result = await client.lead({
        name: "Jane",
        email: "jane@example.com",
        question: "Need CRM",
        phone: "+1234567890",
        message: "Please contact me",
      });

      expect(result.status).toBe("ok");
    });
  });
});
