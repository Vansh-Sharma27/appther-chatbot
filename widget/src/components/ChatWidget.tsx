import {
  useState,
  useRef,
  useEffect,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import { createChatClient } from "../client";
import type { Turn } from "../types";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  isError?: boolean;
  chunkInfo?: { chunk_id: string; url: string; score: number }[];
}

interface LeadInfo {
  message: string;
  name: string;
  email: string;
  submitted: boolean;
}

let _idCounter = 0;
function generateId(): string {
  _idCounter++;
  return `msg-${_idCounter}-${Math.random().toString(36).slice(2, 8)}`;
}

export interface ChatWidgetProps {
  apiBaseUrl: string;
  title?: string;
  placeholder?: string;
}

export function ChatWidget({
  apiBaseUrl,
  title = "Appther Chat",
  placeholder = "Type your message...",
}: ChatWidgetProps) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [lead, setLead] = useState<LeadInfo | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const clientRef = useRef(createChatClient({ baseUrl: apiBaseUrl }));
  const lastAnswerMsgId = useRef<string | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function addMessage(
    role: "user" | "assistant",
    content: string,
    opts?: Partial<Message>
  ) {
    const id = generateId();
    setMessages((prev) => [...prev, { id, role, content, ...opts }]);
    return id;
  }

  const handleSend = async (e?: FormEvent) => {
      e?.preventDefault();
      const q = input.trim();
      if (!q || loading) return;

      setInput("");
      addMessage("user", q);
      setLoading(true);
      setLead(null);

      const history: Turn[] = messages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      try {
        const result = await clientRef.current.chat({
          question: q,
          history,
          onLeadSuggestion: (data) => {
            const d = data as { message: string };
            setLead({
              message: d.message,
              name: "",
              email: "",
              submitted: false,
            });
          },
        });

        const msgId = addMessage("assistant", result.answer, {
          sources: result.sources,
        });
        lastAnswerMsgId.current = msgId;
      } catch (err) {
        const errorMsg =
          err instanceof Error ? err.message : "An unexpected error occurred";
        addMessage("assistant", errorMsg, { isError: true });
      } finally {
        setLoading(false);
      }
    }

  function handleKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  async function handleFeedback(thumbsUp: boolean) {
    const lastAnswer = messages.find(
      (m) => m.id === lastAnswerMsgId.current && m.role === "assistant"
    );
    if (!lastAnswer) return;

    const userMsg = [...messages]
      .reverse()
      .find((m) => m.role === "user");

    try {
      await clientRef.current.feedback({
        question: userMsg?.content ?? "",
        answer: lastAnswer.content,
        thumbs_up: thumbsUp,
        chunks: [],
      });
    } catch {
      // Feedback failure is non-critical
    }
  }

  async function handleLeadSubmit(e: FormEvent) {
    e.preventDefault();
    if (!lead) return;

    try {
      const lastUserMsg = [...messages]
        .reverse()
        .find((m) => m.role === "user");
      await clientRef.current.lead({
        name: lead.name,
        email: lead.email,
        question: lastUserMsg?.content ?? "",
      });
      setLead({ ...lead, submitted: true });
    } catch {
      // Lead submission error is handled silently
    }
  }

  const showFeedback =
    lastAnswerMsgId.current &&
    messages.some(
      (m) =>
        m.id === lastAnswerMsgId.current &&
        m.role === "assistant" &&
        !m.isError
    );

  return (
    <div style={styles.container}>
      <button
        onClick={() => setOpen(!open)}
        style={styles.toggleButton}
        aria-label="Open chat"
      >
        {open ? "✕" : "💬"}
      </button>

      {open && (
        <div style={styles.panel}>
          <div style={styles.header}>
            <span style={styles.headerTitle}>{title}</span>
          </div>

          <div style={styles.messagesContainer}>
            {messages.length === 0 && (
              <div style={styles.emptyState}>
                Ask me anything about Appther!
              </div>
            )}

            {messages.map((msg) => (
              <div
                key={msg.id}
                style={{
                  ...styles.message,
                  ...(msg.role === "user" ? styles.userMessage : {}),
                  ...(msg.isError ? styles.errorMessage : {}),
                }}
              >
                <div style={styles.messageContent}>{msg.content}</div>
                {msg.sources && msg.sources.length > 0 && (
                  <div style={styles.sources}>
                    <span style={styles.sourcesLabel}>Sources: </span>
                    {msg.sources.map((src, i) => (
                      <a
                        key={i}
                        href={src}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={styles.sourceLink}
                      >
                        {src.replace(/https?:\/\//, "").slice(0, 30)}
                        ...
                      </a>
                    ))}
                  </div>
                )}
              </div>
            ))}

            {loading && (
              <div style={{ ...styles.message, ...styles.loadingIndicator }}>
                Thinking...
              </div>
            )}

            {showFeedback && !loading && (
              <div style={styles.feedbackRow}>
                <span style={styles.feedbackLabel}>Was this helpful?</span>
                <button
                  onClick={() => handleFeedback(true)}
                  style={styles.feedbackButton}
                  aria-label="Thumbs up"
                >
                  👍
                </button>
                <button
                  onClick={() => handleFeedback(false)}
                  style={styles.feedbackButton}
                  aria-label="Thumbs down"
                >
                  👎
                </button>
              </div>
            )}

            {lead && !lead.submitted && (
              <div style={styles.leadForm}>
                <p style={styles.leadMessage}>{lead.message}</p>
                <form onSubmit={handleLeadSubmit} style={styles.leadFormInner}>
                  <label style={styles.leadLabel}>
                    Name
                    <input
                      style={styles.leadInput}
                      value={lead.name}
                      onChange={(e) =>
                        setLead({ ...lead, name: e.target.value })
                      }
                      required
                      aria-label="Name"
                    />
                  </label>
                  <label style={styles.leadLabel}>
                    Email
                    <input
                      style={styles.leadInput}
                      type="email"
                      value={lead.email}
                      onChange={(e) =>
                        setLead({ ...lead, email: e.target.value })
                      }
                      required
                      aria-label="Email"
                    />
                  </label>
                  <button type="submit" style={styles.leadSubmitButton}>
                    Submit
                  </button>
                </form>
              </div>
            )}

            {lead?.submitted && (
              <div style={styles.leadSubmitted}>
                Thank you! We'll be in touch soon.
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSend} style={styles.inputForm}>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              disabled={loading}
              style={styles.input}
              aria-label="Message input"
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              style={styles.sendButton}
              aria-label="Send message"
            >
              Send
            </button>
          </form>
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    position: "fixed",
    bottom: "20px",
    right: "20px",
    zIndex: 9999,
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: "14px",
    lineHeight: 1.4,
  },
  toggleButton: {
    width: "56px",
    height: "56px",
    borderRadius: "50%",
    border: "none",
    background: "#2563eb",
    color: "#fff",
    fontSize: "24px",
    cursor: "pointer",
    boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    marginLeft: "auto",
  },
  panel: {
    position: "absolute",
    bottom: "68px",
    right: "0",
    width: "360px",
    maxHeight: "540px",
    background: "#fff",
    borderRadius: "12px",
    boxShadow: "0 8px 32px rgba(0,0,0,0.15)",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    border: "1px solid #e5e7eb",
  },
  header: {
    padding: "14px 16px",
    background: "#2563eb",
    color: "#fff",
    fontWeight: 600,
    fontSize: "15px",
  },
  headerTitle: {},
  messagesContainer: {
    flex: 1,
    overflowY: "auto",
    padding: "12px",
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    minHeight: "200px",
  },
  emptyState: {
    color: "#9ca3af",
    textAlign: "center",
    padding: "40px 16px",
    fontSize: "13px",
  },
  message: {
    padding: "10px 12px",
    borderRadius: "8px",
    background: "#f3f4f6",
    maxWidth: "85%",
    alignSelf: "flex-start",
    wordBreak: "break-word",
  },
  userMessage: {
    background: "#2563eb",
    color: "#fff",
    alignSelf: "flex-end",
  },
  errorMessage: {
    background: "#fef2f2",
    color: "#dc2626",
    border: "1px solid #fecaca",
  },
  loadingIndicator: {
    background: "#f3f4f6",
    color: "#6b7280",
    fontStyle: "italic",
  },
  messageContent: {
    whiteSpace: "pre-wrap",
  },
  sources: {
    marginTop: "6px",
    fontSize: "11px",
    color: "#6b7280",
  },
  sourcesLabel: {
    fontWeight: 600,
  },
  sourceLink: {
    color: "#2563eb",
    textDecoration: "underline",
    marginRight: "6px",
  },
  inputForm: {
    display: "flex",
    borderTop: "1px solid #e5e7eb",
    padding: "8px",
    gap: "6px",
  },
  input: {
    flex: 1,
    border: "1px solid #d1d5db",
    borderRadius: "6px",
    padding: "8px 10px",
    fontSize: "13px",
    outline: "none",
  },
  sendButton: {
    border: "none",
    background: "#2563eb",
    color: "#fff",
    borderRadius: "6px",
    padding: "8px 14px",
    cursor: "pointer",
    fontWeight: 600,
    fontSize: "13px",
  },
  feedbackRow: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    padding: "4px 0",
    fontSize: "12px",
    color: "#6b7280",
  },
  feedbackLabel: {
    fontSize: "12px",
  },
  feedbackButton: {
    border: "1px solid #d1d5db",
    background: "#fff",
    borderRadius: "4px",
    padding: "2px 8px",
    cursor: "pointer",
    fontSize: "14px",
  },
  leadForm: {
    padding: "12px",
    background: "#f0fdf4",
    borderRadius: "8px",
    border: "1px solid #bbf7d0",
  },
  leadMessage: {
    margin: "0 0 8px 0",
    fontSize: "13px",
    color: "#166534",
  },
  leadFormInner: {
    display: "flex",
    flexDirection: "column",
    gap: "6px",
  },
  leadLabel: {
    fontSize: "12px",
    color: "#374151",
    display: "flex",
    flexDirection: "column",
    gap: "2px",
  },
  leadInput: {
    border: "1px solid #d1d5db",
    borderRadius: "4px",
    padding: "6px 8px",
    fontSize: "13px",
  },
  leadSubmitButton: {
    border: "none",
    background: "#16a34a",
    color: "#fff",
    borderRadius: "4px",
    padding: "6px 12px",
    cursor: "pointer",
    fontWeight: 600,
    fontSize: "12px",
    marginTop: "4px",
  },
  leadSubmitted: {
    padding: "12px",
    background: "#f0fdf4",
    borderRadius: "8px",
    border: "1px solid #bbf7d0",
    color: "#166534",
    fontSize: "13px",
    textAlign: "center",
  },
};
