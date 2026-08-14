# SAP MM AI Agent

An AI-powered SAP Materials Management (MM) prototype for interacting with Material Master data through natural language.

The application uses a local open-source language model to understand user requests, MCP tools to control access to SAP MM operations, and a FastAPI-based Material API backed by synthetic SAP Material Master data.

## Goal

The goal of this project is to demonstrate how an AI agent can safely interact with SAP MM Material Master processes.

The prototype supports:

- Searching material master records
- Retrieving material details
- Explaining SAP MM fields and codes
- Creating new materials
- Applying SAP MM validation rules
- Applying controlled default values
- Requiring human confirmation before write operations
- Restricting the AI agent to SAP MM Material Master tasks

The prototype currently uses synthetic SAP data and SQLite. The architecture is designed so that the Material API can later be replaced or adapted to communicate with SAP S/4HANA.

---

## Architecture

```text
┌──────────────────────┐
│ React Frontend       │
│ Port 5173            │
└──────────┬───────────┘
           │
           │ POST /chat
           ▼
┌──────────────────────┐
│ Agent API            │
│ FastAPI :8000        │
│                      │
│ Qwen 2.5 1.5B       │
│ Intent / Planning    │
└──────────┬───────────┘
           │
           │ MCP
           ▼
┌──────────────────────┐
│ MCP Server           │
│ Port 8002            │
│                      │
│ SAP MM Tools         │
│ Validation           │
│ Business Rules       │
└──────────┬───────────┘
           │
           │ HTTP
           ▼
┌──────────────────────┐
│ Material API         │
│ FastAPI :8001        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ SQLite               │
│ Synthetic SAP Data   │
└──────────────────────┘
```

---

## Main Components

### React Frontend

The React application provides the user interface for interacting with the SAP MM AI Agent.

The frontend communicates only with the Agent API.

```text
frontend/
```

Default development URL:

```text
http://localhost:5173
```

### Agent API

The Agent API receives natural-language requests from the frontend.

```text
agent-api/
```

Responsibilities include:

- Receiving chat requests
- Maintaining session context
- Using Qwen to classify user intent
- Selecting approved MCP operations
- Rejecting out-of-scope questions
- Managing confirmation before write operations
- Formatting responses

The current local model is:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

The model is used primarily as a planner and language interface.

It does not receive direct database access.

### MCP Server

The MCP server provides controlled tools that the AI agent can use.

```text
mcp-server/
```

Current MCP capabilities include:

```text
search_material_master
get_material_master
explain_material_master
create_material_master
```

The MCP layer also contains SAP MM-specific validation and defaults.

The agent cannot execute arbitrary database operations.

### Material API

The Material API represents the backend business/data service.

```text
material-api/
```

It currently uses SQLite and synthetic SAP Material Master data.

The API supports reading and creating material records.

In a future SAP S/4HANA implementation, this layer can be adapted or replaced with SAP APIs.

---

## Safety Model

The AI agent is intentionally restricted.

Allowed scope includes:

- SAP MM Material Master
- Material searches
- Material details
- Material field explanations
- Material validation
- Material creation

Out-of-scope questions are rejected.

Examples of rejected topics include:

- Weather
- News
- Politics
- General programming
- Personal advice
- General internet research
- Unrelated SAP modules

The Qwen model does not have a web-search or browser tool.

The MCP server exposes only approved SAP MM operations.

---

## Human-in-the-Loop Material Creation

Material creation requires confirmation.

Example:

```text
User:
Create a finished pump material called AI Pump 006
in plant 1000, material group FG010,
base unit EA, price 240 USD.
```

The agent prepares a proposal:

```text
Description: AI Pump 006
Material Type: FERT
Material Group: FG010
Base Unit: EA
Plant: 1000
Standard Price: 240
Currency: USD
```

The material is NOT created immediately.

The user must confirm the operation.

```text
User:
yes
```

Only after confirmation does the agent invoke:

```text
create_material_master
```

The MCP layer applies validation and SAP MM defaults before calling the Material API.

Example defaults for FERT materials:

```text
Procurement Type: E
MRP Type: PD
Valuation Class: 7920
Price Control: S
```

---

## Project Structure

```text
SAP MM AI agent/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── agent-api/
│   ├── main.py
│   ├── agent.py
│   ├── mcp_client.py
│   └── prompts.py
│
├── mcp-server/
│   ├── server.py
│   ├── material_api_client.py
│   └── rules/
│       ├── __init__.py
│       ├── sap_codes.py
│       ├── defaults.py
│       └── validation.py
│
├── material-api/
│   ├── main.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   ├── seed.py
│   └── materials.db
│
├── frontend/
│   └── src/
│
└── data/
    └── synthetic_sap_material_master_200.csv
```

---

## Running the Project

The project currently requires three backend services and one frontend development server.

### 1. Activate the Python Environment

From the project root:

```bash
cd "/Users/joenikesh/Projects/SAP MM AI agent"

source .venv/bin/activate
```

The prototype uses Python 3.12.

Verify with:

```bash
python --version
```

---

### 2. Start the Material API

Open Terminal 1:

```bash
cd "/Users/joenikesh/Projects/SAP MM AI agent"
source .venv/bin/activate

cd material-api

uvicorn main:app --reload --port 8001
```

Material API:

```text
http://127.0.0.1:8001
```

---

### 3. Start the MCP Server

Open Terminal 2:

```bash
cd "/Users/joenikesh/Projects/SAP MM AI agent"
source .venv/bin/activate

fastmcp run mcp-server/server.py:mcp --transport http --port 8002
```

MCP server:

```text
http://127.0.0.1:8002
```

---

### 4. Start the Agent API

Open Terminal 3:

```bash
cd "/Users/joenikesh/Projects/SAP MM AI agent"
source .venv/bin/activate

cd agent-api

uvicorn main:app --reload --port 8000
```

Agent API documentation:

```text
http://127.0.0.1:8000/docs
```

---

### 5. Start the React Frontend

Open Terminal 4:

```bash
cd "/Users/joenikesh/Projects/SAP MM AI agent/frontend"

npm install
npm run dev
```

Frontend:

```text
http://localhost:5173
```

---

## Example Questions

Search:

```text
Find pump materials in plant 1000.
```

Retrieve:

```text
Show material SYN-FG-000001.
```

SAP MM knowledge:

```text
What does FERT mean in SAP MM?
```

Create:

```text
Create a finished pump material called Demo Pump 001
in plant 1000, material group FG010,
base unit EA, price 250 USD.
```

The agent will request confirmation before creating the material.

---

## Current Technology Stack

### Backend

- Python 3.12
- FastAPI
- FastMCP
- SQLite
- SQLAlchemy
- Hugging Face Transformers
- Qwen 2.5 1.5B Instruct

### Frontend

- React
- TypeScript
- Vite

### AI / Integration

- Local open-source LLM
- Model Context Protocol (MCP)
- SAP MM-specific tools and validation

---

## Current Status

Working:

- Material search
- Material retrieval
- Material explanation
- Material creation
- MCP tool integration
- SAP MM validation rules
- SAP MM default values
- Human confirmation before creation
- Session-based pending creation
- Domain-restricted agent
- React-to-Agent API integration
- Local open-source LLM

Planned next:

- Production-quality React UI
- Approval cards in the frontend
- Material update workflow
- Audit trail
- Better structured agent responses
- Authentication and authorization
- Persistent session storage
- Automated tests
- SAP S/4HANA integration

---

## Prototype Disclaimer

This project is currently a prototype.

It uses synthetic SAP Material Master data and does not connect to a production SAP S/4HANA system.

Before production use, additional controls would be required, including authentication, authorization, audit logging, secure credential management, production SAP APIs, transaction controls, observability, and enterprise security review.