"""End-to-end test for the full research pipeline.

Requires OPENAI_API_KEY and TAVILY_API_KEY to be set in the environment.
Skipped automatically when keys are missing.
"""

import os

import pytest
from httpx import ASGITransport, AsyncClient

from agent.main import app
from agent.memory import reset_vector_store
from dotenv import load_dotenv
load_dotenv()

pytestmark = pytest.mark.asyncio

HAVE_KEYS = bool(os.getenv("OPENAI_API_KEY") and os.getenv("TAVILY_API_KEY"))


@pytest.fixture(autouse=True)
def _clean_vector_store():
    reset_vector_store()
    yield
    reset_vector_store()


@pytest.mark.skipif(not HAVE_KEYS, reason="API keys not set")
async def test_full_research_pipeline():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/research",
            json={"query": "What are the main benefits and risks of nuclear fusion energy?"},
            timeout=120,
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["answer"], "Answer should not be empty"
        assert data["session_id"], "Should have a session ID"
        assert len(data["sub_queries"]) >= 2, "Should decompose into at least 2 sub-queries"

        assert isinstance(data["sources"], list), "Sources should be a list"
        if data["sources"]:
            assert "url" in data["sources"][0]

        budget = data["budget"]
        assert budget["total_llm_calls"] >= 3

        for call in budget["per_call_breakdown"]:
            assert call["prompt_tokens"] > 0

        memory = data["memory_state"]
        assert memory["num_findings"] >= 1

        assert data["elapsed_seconds"] > 0


@pytest.mark.skipif(not HAVE_KEYS, reason="API keys not set")
async def test_decompose_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/decompose",
            json={"query": "Compare AI adoption in healthcare vs manufacturing in the US and EU"},
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sub_queries"]) >= 2


@pytest.mark.skipif(not HAVE_KEYS, reason="API keys not set")
async def test_search_returns_sources():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/search",
            json={"sub_query": "Benefits of nuclear fusion energy"},
            timeout=60,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) >= 1
        assert "url" in data["sources"][0]
        assert "title" in data["sources"][0]


@pytest.mark.skipif(not HAVE_KEYS, reason="API keys not set")
async def test_memory_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/memory/state")
        assert resp.status_code == 200

        resp = await client.post("/memory/reset")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


async def test_sessions_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/sessions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


@pytest.mark.skipif(not HAVE_KEYS, reason="API keys not set")
async def test_followup_with_rag():
    """Research a topic then ask a follow-up that uses vector retrieval."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res1 = await client.post(
            "/search",
            json={"sub_query": "Benefits of nuclear fusion energy"},
            timeout=60,
        )
        assert res1.status_code == 200

        res2 = await client.post(
            "/followup",
            json={
                "query": "What about fusion fuel sources?",
                "session_id": "test",
                "original_query": "Benefits of nuclear fusion energy",
            },
            timeout=60,
        )
        assert res2.status_code == 200
        data = res2.json()
        assert data["answer"]
        assert data["is_new_research"] is False
        assert data["used_retrieval"] is True
        assert data["retrieved_chunks"] >= 1
        assert data["elapsed_seconds"] > 0
        assert data["session_id"] == "test"
