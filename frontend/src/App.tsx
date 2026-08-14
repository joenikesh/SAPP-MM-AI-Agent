import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { sendMessage, AgentApiError } from "./services/agentApi";
import "./App.css";

type Role = "user" | "assistant" | "error";

type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  time: Date;
};

const SESSION_STORAGE_KEY = "sap-mm-session-id";
const MAX_TEXTAREA_HEIGHT = 160;

const SUGGESTIONS = [
  "Show open purchase orders for vendor 1000123",
  "List materials below reorder point",
  "What's the status of PO 4500001234?",
];

function getOrCreateSessionId(): string {
  const existing = sessionStorage.getItem(SESSION_STORAGE_KEY);
  if (existing) return existing;

  const id = crypto.randomUUID();
  sessionStorage.setItem(SESSION_STORAGE_KEY, id);
  return id;
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function Avatar({ role }: { role: Role }) {
  if (role === "user") {
    return (
      <div className="chat-row-avatar" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none">
          <path
            d="M12 12a4.5 4.5 0 1 0 0-9 4.5 4.5 0 0 0 0 9Zm0 2c-4.14 0-7.5 2.24-7.5 5v1.25c0 .41.34.75.75.75h13.5c.41 0 .75-.34.75-.75V19c0-2.76-3.36-5-7.5-5Z"
            fill="currentColor"
          />
        </svg>
      </div>
    );
  }

  if (role === "error") {
    return (
      <div className="chat-row-avatar" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none">
          <path
            d="M12 3 1 21h22L12 3Zm0 6v6m0 3.25h.01"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
    );
  }

  return (
    <div className="chat-row-avatar" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="none">
        <path
          d="M12 2 3 6.5 12 11l9-4.5L12 2Zm-9 6.5V17l9 4.5 9-4.5V8.5M12 11v10.5"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
}

function SendIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="m3 11 18-8-8 18-2.5-7.5L3 11Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function App() {
  const [sessionId] = useState(getOrCreateSessionId);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const logRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Keep the log scrolled to the latest message.
  useEffect(() => {
    logRef.current?.scrollTo({
      top: logRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, loading]);

  // Auto-grow the textarea as the person types.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_TEXTAREA_HEIGHT)}px`;
  }, [input]);

  const handleSend = async (overrideText?: string) => {
    const trimmed = (overrideText ?? input).trim();
    if (!trimmed || loading) return;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
      time: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await sendMessage({
        session_id: sessionId,
        message: trimmed,
      });

      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: response.reply,
          time: new Date(),
        },
      ]);
    } catch (error) {
      const detail =
        error instanceof AgentApiError
          ? error.message
          : "Something went wrong talking to the agent.";

      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "error", content: detail, time: new Date() },
      ]);
    } finally {
      setLoading(false);
      textareaRef.current?.focus();
    }
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSend();
    }
  };

  const handleSuggestion = (text: string) => {
    void handleSend(text);
  };

  return (
    <div className="chat-page">
      <div className="chat-shell">
        <header className="chat-header">
          <div className="chat-header-identity">
            <div className="chat-avatar" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none">
                <path
                  d="M12 2 3 6.5 12 11l9-4.5L12 2Zm-9 6.5V17l9 4.5 9-4.5V8.5M12 11v10.5"
                  stroke="currentColor"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
            <div>
              <h1>SAP MM AI Agent</h1>
              <p className="chat-header-subtitle">
                Materials Management assistant
              </p>
            </div>
          </div>

          <span className="chat-status">
            <span className="chat-status-dot" aria-hidden="true" />
            Ready
          </span>
        </header>

        <div className="chat-log" ref={logRef} role="log" aria-live="polite">
          {messages.length === 0 && !loading && (
            <div className="chat-empty">
              <div className="chat-empty-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none">
                  <path
                    d="M12 2 3 6.5 12 11l9-4.5L12 2Zm-9 6.5V17l9 4.5 9-4.5V8.5M12 11v10.5"
                    stroke="currentColor"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </div>
              <h2>Ask the SAP MM agent</h2>
              <p className="chat-empty-desc">
                Purchase orders, vendor records, and material master data —
                in plain language.
              </p>
              <div className="chat-suggestions">
                {SUGGESTIONS.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    className="chat-suggestion-chip"
                    onClick={() => handleSuggestion(suggestion)}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className={`chat-row chat-row--${message.role}`}>
              <Avatar role={message.role} />
              <div className="chat-row-body">
                <div className={`chat-bubble chat-bubble--${message.role}`}>
                  {message.content}
                </div>
                <span className="chat-timestamp">{formatTime(message.time)}</span>
              </div>
            </div>
          ))}

          {loading && (
            <div className="chat-row chat-row--assistant">
              <Avatar role="assistant" />
              <div className="chat-thinking" aria-label="Agent is thinking">
                <span className="chat-thinking-text">Thinking…</span>
              </div>
            </div>
          )}
        </div>

        <div className="chat-composer">
          <div className="chat-composer-bar">
            <label htmlFor="chat-input" className="sr-only">
              Message the SAP MM agent
            </label>
            <textarea
              id="chat-input"
              ref={textareaRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask the SAP MM Agent…"
              rows={1}
              disabled={loading}
            />
            <button
              type="button"
              className="chat-send-btn"
              onClick={() => handleSend()}
              disabled={loading || !input.trim()}
              aria-label="Send message"
            >
              <SendIcon />
            </button>
          </div>
          <p className="chat-composer-hint">Enter to send · Shift + Enter for a new line</p>
        </div>
      </div>
    </div>
  );
}

export default App;
