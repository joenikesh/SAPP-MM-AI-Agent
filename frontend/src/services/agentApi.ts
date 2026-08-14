export type ChatRequest = {
  session_id: string;
  message: string;
};

export type ChatResponse = {
  reply: string;
};

const AGENT_API_URL = "http://127.0.0.1:8000";

export async function sendMessage(
  request: ChatRequest
): Promise<ChatResponse> {
  const response = await fetch(
    `${AGENT_API_URL}/chat`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    }
  );

  if (!response.ok) {
    throw new Error(
      `Agent API error: ${response.status}`
    );
  }

  return response.json();
}