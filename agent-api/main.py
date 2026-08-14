from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from agent import run_agent


app = FastAPI(
    title="SAP MM AI Agent API",
    description="AI Agent for SAP MM Material Master",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -------------------------------------------------
# TEMPORARY SESSION MEMORY
# -------------------------------------------------

# Stores material creation proposals waiting for
# user confirmation.
#
# Example:
# {
#     "demo-user-1": {
#         "material_description": "AI Pump 005",
#         "material_type": "FERT",
#         ...
#     }
# }
#
# This is temporary in-memory storage.
# Restarting the Agent API clears it.

pending_creations = {}


# -------------------------------------------------
# REQUEST / RESPONSE MODELS
# -------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


# -------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "SAP MM AI Agent API"
    }


# -------------------------------------------------
# CHAT ENDPOINT
# -------------------------------------------------

@app.post(
    "/chat",
    response_model=ChatResponse
)
async def chat(request: ChatRequest):

    reply = await run_agent(
        message=request.message,
        session_id=request.session_id,
        pending_creations=pending_creations,
    )

    return ChatResponse(
        reply=reply
    )