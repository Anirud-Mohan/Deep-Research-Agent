# Deep Research Agent

## Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys) (`gpt-5-mini` + `gpt-4o-mini`)
- A [Tavily API key](https://tavily.com/) (web search)
- Docker (only needed for the optional n8n workflow)

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd Deep-Research-Agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

### 5. Start the server

```bash
uvicorn agent.main:app --reload --port 8000
```

### 6. Open the UI

Navigate to **http://localhost:8000** in your browser.

---

## Optional: n8n Workflow

n8n provides a visual view of the same pipeline. Docker is required.

```bash
docker compose up -d
```

1. Open **http://localhost:5678** and complete the n8n first-run setup.
2. Go to **Workflows → Import from file** and select `n8n/workflows/research_pipeline.json`.
3. Activate the workflow.
4. Trigger it:

```bash
curl -X POST http://localhost:5678/webhook/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main risks and benefits of nuclear fusion energy?"}'
```

---

## Running Tests

```bash
# End-to-end tests (requires API keys in .env)
python -m pytest tests/test_e2e.py -v
```
