import { describe, it, expect, beforeAll, afterEach, afterAll } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { ChatWidget } from "../src/components/ChatWidget";

const API_BASE = "https://chat.appther.com";

const server = setupServer();

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

function renderWidget() {
  const user = userEvent.setup();
  const utils = render(<ChatWidget apiBaseUrl={API_BASE} />);
  return { user, ...utils };
}

function makeStreamResponse(events: string[]): HttpResponse<string> {
  const body = events.join("");
  return new HttpResponse(body, {
    headers: { "Content-Type": "text/event-stream" },
  });
}

describe("ChatWidget", () => {
  describe("rendering and open/close", () => {
    it("renders a closed widget with a toggle button", () => {
      renderWidget();
      const toggle = screen.getByRole("button");
      expect(toggle).toBeInTheDocument();
    });

    it("opens the chat panel when toggle button is clicked", async () => {
      const { user } = renderWidget();
      const toggle = screen.getByRole("button");
      await user.click(toggle);

      expect(
        screen.getByPlaceholderText(/type.*message/i)
      ).toBeInTheDocument();
    });

    it("closes the chat panel when toggle button is clicked again", async () => {
      const { user } = renderWidget();
      const toggle = screen.getByRole("button");
      await user.click(toggle);
      await user.click(toggle);

      expect(
        screen.queryByPlaceholderText(/type.*message/i)
      ).not.toBeInTheDocument();
    });
  });

  describe("sending messages", () => {
    beforeEach(() => {
      server.use(
        http.post(`${API_BASE}/chat`, () =>
          makeStreamResponse([
            `event: answer\ndata: ${JSON.stringify({ token: "Appther offers ERP solutions for businesses." })}\n\n`,
            `event: sources\ndata: ${JSON.stringify({ sources: ["https://www.appther.com/services/erp"] })}\n\n`,
            `event: done\ndata: ${JSON.stringify({ model: "gemini-2.5-flash-lite", chunks_used: 4 })}\n\n`,
          ])
        )
      );
    });

    it("sends a message and displays the user message", async () => {
      const { user } = renderWidget();
      await user.click(screen.getByRole("button"));
      const input = screen.getByPlaceholderText(/type.*message/i);

      await user.type(input, "What does Appther do?");
      await user.keyboard("{Enter}");

      expect(screen.getByText("What does Appther do?")).toBeInTheDocument();
    });

    it("displays the bot response after sending", async () => {
      const { user } = renderWidget();
      await user.click(screen.getByRole("button"));
      const input = screen.getByPlaceholderText(/type.*message/i);

      await user.type(input, "What does Appther do?");
      await user.keyboard("{Enter}");

      expect(
        await screen.findByText(
          "Appther offers ERP solutions for businesses."
        )
      ).toBeInTheDocument();
    });

    it("shows source links in the bot response", async () => {
      const { user } = renderWidget();
      await user.click(screen.getByRole("button"));
      const input = screen.getByPlaceholderText(/type.*message/i);

      await user.type(input, "What does Appther do?");
      await user.keyboard("{Enter}");

      const link = await screen.findByRole("link", {
        name: /appther\.com\/services\/erp/i,
      });
      expect(link).toBeInTheDocument();
      expect(link).toHaveAttribute(
        "href",
        "https://www.appther.com/services/erp"
      );
    });
  });

  describe("feedback", () => {
    beforeEach(() => {
      server.use(
        http.post(`${API_BASE}/chat`, () =>
          makeStreamResponse([
            `event: answer\ndata: ${JSON.stringify({ token: "Appther offers ERP solutions." })}\n\n`,
            `event: sources\ndata: ${JSON.stringify({ sources: ["https://www.appther.com/faq"] })}\n\n`,
            `event: done\ndata: ${JSON.stringify({ model: "gemini-2.5-flash-lite", chunks_used: 4 })}\n\n`,
          ])
        ),
        http.post(`${API_BASE}/feedback`, () =>
          HttpResponse.json({ status: "ok" })
        )
      );
    });

    it("shows feedback buttons after receiving an answer", async () => {
      const { user } = renderWidget();
      await user.click(screen.getByRole("button"));
      const input = screen.getByPlaceholderText(/type.*message/i);

      await user.type(input, "What does Appther do?");
      await user.keyboard("{Enter}");

      const thumbsUp = await screen.findByRole("button", {
        name: /thumbs.?up|like|helpful/i,
      });
      const thumbsDown = screen.getByRole("button", {
        name: /thumbs.?down|dislike|not helpful/i,
      });

      expect(thumbsUp).toBeInTheDocument();
      expect(thumbsDown).toBeInTheDocument();
    });
  });

  describe("error handling", () => {
    it("shows an error message when the API call fails", async () => {
      server.use(
        http.post(`${API_BASE}/chat`, () =>
          makeStreamResponse([
            `event: error\ndata: ${JSON.stringify({ detail: "Failed to generate answer" })}\n\n`,
          ])
        )
      );

      const { user } = renderWidget();
      await user.click(screen.getByRole("button"));
      const input = screen.getByPlaceholderText(/type.*message/i);

      await user.type(input, "What?");
      await user.keyboard("{Enter}");

      expect(
        await screen.findByText(/failed to generate answer/i)
      ).toBeInTheDocument();
    });
  });

  describe("lead suggestion", () => {
    it("shows a lead capture form when no answer is found", async () => {
      server.use(
        http.post(`${API_BASE}/chat`, () =>
          makeStreamResponse([
            `event: answer\ndata: ${JSON.stringify({ token: "I don't have information about that." })}\n\n`,
            `event: sources\ndata: ${JSON.stringify({ sources: [] })}\n\n`,
            `event: lead_suggestion\ndata: ${JSON.stringify({ message: "Would you like us to reach out?" })}\n\n`,
            `event: done\ndata: ${JSON.stringify({ model: "gemini-2.5-flash-lite", chunks_used: 0 })}\n\n`,
          ])
        ),
        http.post(`${API_BASE}/lead`, () =>
          HttpResponse.json({ status: "ok" })
        )
      );

      const { user } = renderWidget();
      await user.click(screen.getByRole("button"));
      const input = screen.getByPlaceholderText(/type.*message/i);

      await user.type(input, "Obscure question");
      await user.keyboard("{Enter}");

      expect(
        await screen.findByText(/would you like us to reach out/i)
      ).toBeInTheDocument();

      const nameInput = screen.getByLabelText(/name/i);
      expect(nameInput).toBeInTheDocument();
    });

    it("submits the lead form successfully", async () => {
      server.use(
        http.post(`${API_BASE}/chat`, () =>
          makeStreamResponse([
            `event: answer\ndata: ${JSON.stringify({ token: "I don't know." })}\n\n`,
            `event: sources\ndata: ${JSON.stringify({ sources: [] })}\n\n`,
            `event: lead_suggestion\ndata: ${JSON.stringify({ message: "Would you like us to reach out?" })}\n\n`,
            `event: done\ndata: ${JSON.stringify({ model: "gemini-2.5-flash-lite", chunks_used: 0 })}\n\n`,
          ])
        ),
        http.post(`${API_BASE}/lead`, () =>
          HttpResponse.json({ status: "ok" })
        )
      );

      const { user } = renderWidget();
      await user.click(screen.getByRole("button"));
      const input = screen.getByPlaceholderText(/type.*message/i);

      await user.type(input, "Random question");
      await user.keyboard("{Enter}");

      await screen.findByText(/would you like us to reach out/i);

      const nameInput = screen.getByLabelText(/name/i);
      const emailInput = screen.getByLabelText(/email/i);

      await user.type(nameInput, "John Doe");
      await user.type(emailInput, "john@example.com");

      const submitButton = screen.getByRole("button", { name: /^submit$/i });
      await user.click(submitButton);

      expect(
        await screen.findByText(/thank you|submitted|we'll be in touch/i)
      ).toBeInTheDocument();
    });
  });
});
