export type ChatRequest = {
  session_id: string;
  message: string;
};

export type ChatResponse = {
  reply: string;
};

// Configure with a .env file: VITE_AGENT_API_URL=https://your-backend
// Falls back to local dev backend if not set.
const AGENT_API_URL =
  import.meta.env.VITE_AGENT_API_URL ?? "http://127.0.0.1:8000";

const REQUEST_TIMEOUT_MS = 30_000;

export class AgentApiError extends Error {}

export async function sendMessage(
  request: ChatRequest,
  signal?: AbortSignal
): Promise<ChatResponse> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  if (signal) {
    signal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  try {
    const response = await fetch(`${AGENT_API_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
      signal: controller.signal,
    });

    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = await response.json();
        detail = body?.detail ?? body?.message ?? detail;
      } catch {
        // Response wasn't JSON — fall back to the status text above.
      }
      throw new AgentApiError(`Agent API error (${response.status}): ${detail}`);
    }

    return (await response.json()) as ChatResponse;
  } catch (error) {
    if (error instanceof AgentApiError) throw error;

    if (error instanceof DOMException && error.name === "AbortError") {
      throw new AgentApiError(
        "The agent took too long to respond. Please try again."
      );
    }

    throw new AgentApiError(
      "Could not reach the agent service. Is the backend running?"
    );
  } finally {
    clearTimeout(timeoutId);
  }
}
