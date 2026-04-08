# Deep Research Agent

A research agent that answers complex, multi-part queries by decomposing them into focused sub-questions, searching the web, and synthesising findings — all while operating under a strict **2,048-token context window** per LLM call.

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│ Query Decomposer │  →  2-5 focused sub-questions
└────────┬────────┘
         │
         ▼  (for each sub-query)
┌─────────────────┐     ┌────────────┐
│   Researcher     │◄───►│ Tavily API │
│  search + chunk  │     └────────────┘
└────────┬────────┘
         │  raw results + source URLs
         ▼
┌─────────────────┐
│  Memory Manager  │
│ ┌─────────────┐  │  Vector Store   — long-term chunk storage (ChromaDB)
│ ├─────────────┤  │  Summarisation  — 3-level compression cascade
│ ├─────────────┤  │  Working Memory — FIFO buffer, ≤2048 tokens
│ └─────────────┘  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Synthesiser    │  Iterative refinement: builds the answer one sub-query at a time
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Follow-Up (RAG) │  Retrieves from vector store for conversational follow-ups
└─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for n8n, optional)
- API keys: [Groq](https://console.groq.com) (free) and [Tavily](https://tavily.com/)

### 1. Clone and install

```bash
git clone <repo-url>
cd Deep-Research-Agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

Default configuration uses **Groq** (free, fast) with **Llama 3.3 70B**. See `.env.example` for how to switch providers.

### 3. Run the API server

```bash
uvicorn agent.main:app --reload --port 8000
```

### 4. Open the UI

Navigate to **http://localhost:8000** in your browser.

The chat-based UI supports:
- **Initial research** — decompose, search, and synthesise complex queries
- **Follow-up questions** — ask follow-ups that use RAG retrieval from the vector store
- **Side panel** — real-time token budget chart, sources, and constraint compliance
- **Markdown rendering** — answers displayed with proper formatting
- **New Session** — clear memory and start fresh

### 5. Or use the API directly

```bash
# Full research pipeline
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main benefits and risks of nuclear fusion energy?"}'

# Follow-up using RAG retrieval
curl -X POST http://localhost:8000/followup \
  -H "Content-Type: application/json" \
  -d '{"query": "What about fuel sources?", "session_id": "<session_id>"}'
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Chat-based web UI |
| `/research` | POST | Full pipeline — decompose, research, synthesise |
| `/followup` | POST | Follow-up via RAG retrieval from vector store |
| `/decompose` | POST | Break a query into sub-queries |
| `/search` | POST | Research a single sub-query (returns summary + source URLs) |
| `/synthesise` | POST | Produce an answer from pre-computed findings |
| `/memory/state` | GET | Inspect vector store contents |
| `/memory/reset` | POST | Clear vector store between sessions |
| `/sessions` | GET | List all saved research sessions |
| `/sessions/{id}` | GET | Load a specific session log |

## Memory Constraint

**Rule**: No LLM call receives more than 2,048 tokens of research context.

This is enforced by:
- **`tiktoken`** for exact token counting before every call
- **Summarisation cascade** that compresses raw results (Level 0 → 1 → 2)
- **Working memory buffer** with FIFO eviction
- **Draft self-compression** when the iterative answer grows too large
- **RAG retrieval** for follow-ups — reuses stored context instead of re-searching

The UI displays a **per-call token bar chart** in the side panel so constraint compliance is visually verifiable.

See [evaluation.md](evaluation.md) for detailed analysis of trade-offs.

## Session Logging

Every `/research` call is automatically logged to `logs/<session_id>.json` with the full pipeline trace: query, sub-queries, findings, source URLs, answer, memory state, and token budget.

## n8n Workflow

The agent can also be orchestrated via n8n for visual workflow management.

```bash
docker compose up -d    # Start n8n at http://localhost:5678
```

1. Open n8n → **Workflows** → **Import from file** → select `n8n/workflows/research_pipeline.json`
2. Activate the workflow
3. Trigger: `curl -X POST http://localhost:5678/webhook/research -H "Content-Type: application/json" -d '{"query": "..."}'`

## Running Tests

```bash
# Unit tests (no API keys needed)
python -m pytest tests/test_budget.py tests/test_memory.py -v

# End-to-end tests (requires API keys in .env)
python -m pytest tests/test_e2e.py -v
```

## Project Structure

```
├── agent/
│   ├── config.py          # Constraint definitions and settings
│   ├── budget.py          # Token counting and budget enforcement
│   ├── llm.py             # LLM wrapper with budget checking + logging
│   ├── memory.py          # Vector store + summarisation cascade + working memory
│   ├── decomposer.py      # Query → sub-queries
│   ├── researcher.py      # Web search + parallel summarisation + source URLs
│   ├── synthesizer.py     # Iterative refinement answer generation
│   ├── logger.py          # Session logging to JSON
│   ├── main.py            # FastAPI app (research + follow-up + UI)
│   └── static/index.html  # Chat-based UI with metrics sidebar
├── n8n/workflows/         # Exportable n8n workflow JSON
├── tests/                 # Unit and integration tests
├── examples/              # Sample queries
├── logs/                  # Session logs (gitignored)
├── evaluation.md          # Architecture trade-offs analysis
├── docker-compose.yml     # n8n container
└── requirements.txt
```
