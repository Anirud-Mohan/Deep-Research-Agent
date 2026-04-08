"""FastAPI application — exposes endpoints consumed by n8n, the UI, and direct callers."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent.budget import BudgetTracker, count_tokens, truncate_to_tokens
from agent.config import settings
from agent.decomposer import decompose
from agent.llm import call_llm
from agent.logger import list_sessions, load_session, save_session
from agent.memory import WorkingMemory, reset_vector_store, retrieve_chunks
from agent.researcher import research_subquery
from agent.synthesizer import synthesise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent")

app = FastAPI(
    title="Deep Research Agent",
    description="Research agent with memory constraints",
)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ResearchRequest(BaseModel):
    query: str


class DecomposeResponse(BaseModel):
    sub_queries: list[str]


class SubQueryRequest(BaseModel):
    sub_query: str
    session_id: str = "default"


class SubQueryResponse(BaseModel):
    sub_query: str
    summary: str
    sources: list[dict[str, str]]


class SynthesiseRequest(BaseModel):
    query: str
    findings: list[dict[str, str]]


class FollowUpRequest(BaseModel):
    query: str
    session_id: str


class ResearchResponse(BaseModel):
    session_id: str
    answer: str
    sub_queries: list[str]
    sources: list[dict[str, str]]
    memory_state: dict
    budget: dict
    elapsed_seconds: float


class FollowUpResponse(BaseModel):
    answer: str
    used_retrieval: bool
    retrieved_chunks: int
    sources: list[dict[str, str]]
    budget: dict
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Granular endpoints (called individually by n8n nodes)
# ---------------------------------------------------------------------------


@app.post("/decompose", response_model=DecomposeResponse)
async def api_decompose(req: ResearchRequest):
    """Break a complex query into sub-queries."""
    tracker = BudgetTracker()
    subs = await decompose(req.query, tracker)
    return DecomposeResponse(sub_queries=subs)


@app.post("/search", response_model=SubQueryResponse)
async def api_search(req: SubQueryRequest):
    """Research a single sub-query: search + summarise + store."""
    tracker = BudgetTracker()
    result = await research_subquery(req.sub_query, tracker)
    return SubQueryResponse(
        sub_query=result.sub_query,
        summary=result.summary,
        sources=result.sources,
    )


@app.post("/synthesise", response_model=dict)
async def api_synthesise(req: SynthesiseRequest):
    """Synthesise a final answer from pre-computed findings."""
    tracker = BudgetTracker()
    answer = await synthesise(req.query, req.findings, tracker)
    return {"answer": answer, "budget": tracker.summary()}


@app.get("/memory/state")
async def api_memory_state():
    """Introspect the vector store."""
    chunks = retrieve_chunks("research", top_k=10)
    return {"stored_chunks": len(chunks), "sample_chunks": chunks[:3]}


@app.post("/memory/reset")
async def api_memory_reset():
    """Wipe vector store between sessions."""
    reset_vector_store()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Session log endpoints
# ---------------------------------------------------------------------------


@app.get("/sessions")
async def api_list_sessions():
    """List all saved research sessions."""
    return list_sessions()


@app.get("/sessions/{session_id}")
async def api_get_session(session_id: str):
    """Load a specific session by ID."""
    data = load_session(session_id)
    if data is None:
        return {"error": "Session not found"}
    return data


# ---------------------------------------------------------------------------
# All-in-one research endpoint
# ---------------------------------------------------------------------------


@app.post("/research", response_model=ResearchResponse)
async def api_research(req: ResearchRequest):
    """Full pipeline: decompose -> research each sub-query -> synthesise."""
    session_id = uuid.uuid4().hex[:12]
    tracker = BudgetTracker()
    memory = WorkingMemory()
    pipeline_start = time.perf_counter()

    log.info("=== NEW RESEARCH SESSION %s ===", session_id)
    log.info("Query: %s", req.query)

    log.info("--- Step 1/3: Decomposing query ---")
    sub_queries = await decompose(req.query, tracker)
    sub_queries = sub_queries[:3]
    log.info("Decomposed into %d sub-queries: %s", len(sub_queries), sub_queries)

    findings: list[dict[str, str]] = []
    all_sources: list[dict[str, str]] = []

    for i, sq in enumerate(sub_queries, 1):
        log.info("--- Step 2/3: Researching sub-query %d/%d ---", i, len(sub_queries))
        result = await research_subquery(sq, tracker)
        memory.add_finding(sq, result.summary)
        findings.append({"subquery": sq, "summary": result.summary})
        all_sources.extend(result.sources)

    log.info("--- Step 3/3: Synthesising answer ---")
    answer = await synthesise(req.query, findings, tracker)

    elapsed = time.perf_counter() - pipeline_start
    log.info(
        "=== SESSION %s COMPLETE in %.1fs  (%d LLM calls, %d tokens) ===",
        session_id, elapsed, tracker.total_calls, tracker.total_tokens,
    )

    save_session(
        session_id=session_id,
        query=req.query,
        sub_queries=sub_queries,
        findings=findings,
        sources=all_sources,
        answer=answer,
        memory_state=memory.state_snapshot(),
        budget=tracker.summary(),
    )

    return ResearchResponse(
        session_id=session_id,
        answer=answer,
        sub_queries=sub_queries,
        sources=all_sources,
        memory_state=memory.state_snapshot(),
        budget=tracker.summary(),
        elapsed_seconds=round(elapsed, 1),
    )


# ---------------------------------------------------------------------------
# Follow-up endpoint — uses RAG retrieval from vector store
# ---------------------------------------------------------------------------

_FOLLOWUP_SYSTEM = (
    "You are a research assistant. Answer the user's follow-up question "
    "using ONLY the retrieved context below. Be concise and factual. "
    "If the context does not contain enough information to answer, "
    "say so clearly."
)


@app.post("/followup", response_model=FollowUpResponse)
async def api_followup(req: FollowUpRequest):
    """Answer a follow-up using RAG retrieval from the vector store.

    First attempts to answer from stored research chunks. If the vector
    store has insufficient context, falls back to a targeted web search.
    """
    t0 = time.perf_counter()
    tracker = BudgetTracker()

    log.info("=== FOLLOW-UP for session %s ===", req.session_id)
    log.info("Follow-up query: %s", req.query)

    chunks = retrieve_chunks(req.query, top_k=5)
    all_sources: list[dict[str, str]] = []
    used_retrieval = len(chunks) > 0

    if chunks:
        log.info("Retrieved %d chunks from vector store", len(chunks))
        context = "\n\n".join(
            f"[Chunk {i+1}]\n{c}" for i, c in enumerate(chunks)
        )
        context = truncate_to_tokens(context, tracker.max_context_tokens)

        user_msg = f"Question: {req.query}\n\nRetrieved context:\n{context}"
        if count_tokens(user_msg) > tracker.max_context_tokens:
            user_msg = truncate_to_tokens(user_msg, tracker.max_context_tokens)

        answer = await call_llm(
            system=_FOLLOWUP_SYSTEM,
            user=user_msg,
            tracker=tracker,
            step="followup_rag",
        )
    else:
        log.info("No chunks in vector store, falling back to web search")
        result = await research_subquery(req.query, tracker)
        all_sources = result.sources
        answer = result.summary

    elapsed = time.perf_counter() - t0
    log.info("=== FOLLOW-UP COMPLETE in %.1fs ===", elapsed)

    return FollowUpResponse(
        answer=answer,
        used_retrieval=used_retrieval,
        retrieved_chunks=len(chunks),
        sources=all_sources,
        budget=tracker.summary(),
        elapsed_seconds=round(elapsed, 1),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def ui():
    """Serve the research agent UI."""
    with open("agent/static/index.html") as f:
        return HTMLResponse(content=f.read())
