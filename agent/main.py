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
    original_query: str = ""


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
    is_new_research: bool
    used_retrieval: bool
    retrieved_chunks: int
    sources: list[dict[str, str]]
    sub_queries: list[str]
    memory_state: dict
    budget: dict
    elapsed_seconds: float
    session_id: str


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
    return {"stored_chunks": len(chunks), "sample_chunks": [c["text"][:200] for c in chunks[:3]]}


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

_CLASSIFY_SYSTEM = (
    "You decide whether a user's new message is a follow-up to a previous "
    "research topic or a completely new question.\n\n"
    "Return ONLY the word: FOLLOWUP or NEW"
)

_FOLLOWUP_SYSTEM = (
    "You are a research assistant. Answer the user's follow-up question "
    "using ONLY the retrieved context below. Be detailed and factual.\n\n"
    "If the context does NOT contain enough information to answer the "
    "question, respond with EXACTLY the text: INSUFFICIENT_CONTEXT\n"
    "Do not add anything else in that case."
)


async def _classify_query(
    new_query: str, original_query: str, tracker: BudgetTracker
) -> bool:
    """Return True if *new_query* is a follow-up to *original_query*."""
    user_msg = (
        f"Previous research topic: {original_query}\n"
        f"New message: {new_query}"
    )
    result = await call_llm(
        system=_CLASSIFY_SYSTEM,
        user=user_msg,
        tracker=tracker,
        step="classify",
        max_output_tokens=10,
        model=settings.summary_llm_model,
    )
    return "FOLLOWUP" in result.upper()


@app.post("/followup", response_model=FollowUpResponse)
async def api_followup(req: FollowUpRequest):
    """Smart routing: classifies the query as a follow-up or new research.

    Follow-ups use RAG retrieval from the vector store.
    New queries trigger the full research pipeline.
    """
    t0 = time.perf_counter()
    tracker = BudgetTracker()

    log.info("=== INCOMING QUERY for session %s ===", req.session_id)
    log.info("Query: %s", req.query)

    is_followup = False
    if req.original_query:
        is_followup = await _classify_query(req.query, req.original_query, tracker)
        log.info("Classification: %s", "FOLLOWUP" if is_followup else "NEW RESEARCH")

    # --- New research path ---
    if not is_followup:
        log.info("Routing to full research pipeline")
        reset_vector_store()
        session_id = uuid.uuid4().hex[:12]
        memory = WorkingMemory()

        sub_queries = await decompose(req.query, tracker)
        sub_queries = sub_queries[:3]
        log.info("Decomposed into %d sub-queries: %s", len(sub_queries), sub_queries)

        findings: list[dict[str, str]] = []
        all_sources: list[dict[str, str]] = []

        for i, sq in enumerate(sub_queries, 1):
            log.info("--- Researching sub-query %d/%d ---", i, len(sub_queries))
            result = await research_subquery(sq, tracker)
            memory.add_finding(sq, result.summary)
            findings.append({"subquery": sq, "summary": result.summary})
            all_sources.extend(result.sources)

        answer = await synthesise(req.query, findings, tracker)
        elapsed = time.perf_counter() - t0

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

        log.info("=== NEW RESEARCH COMPLETE in %.1fs ===", elapsed)

        return FollowUpResponse(
            answer=answer,
            is_new_research=True,
            used_retrieval=False,
            retrieved_chunks=0,
            sources=all_sources,
            sub_queries=sub_queries,
            memory_state=memory.state_snapshot(),
            budget=tracker.summary(),
            elapsed_seconds=round(elapsed, 1),
            session_id=session_id,
        )

    # --- Follow-up path (RAG retrieval with research fallback) ---
    log.info("Routing to RAG retrieval")
    chunk_results = retrieve_chunks(req.query, top_k=5)
    all_sources: list[dict[str, str]] = []
    used_retrieval = False
    answer = ""

    if chunk_results:
        log.info("Retrieved %d chunks from vector store", len(chunk_results))

        # Extract unique sources from chunk metadata
        seen_urls: set[str] = set()
        for cr in chunk_results:
            url = cr["metadata"].get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_sources.append({
                    "title": cr["metadata"].get("title", ""),
                    "url": url,
                })

        # Build context with proper budget: reserve tokens for the question prefix
        question_prefix = f"Question: {req.query}\n\nRetrieved context:\n"
        prefix_tokens = count_tokens(question_prefix)
        context_budget = tracker.max_context_tokens - prefix_tokens

        context = "\n\n".join(
            f"[Chunk {i+1}]\n{cr['text']}" for i, cr in enumerate(chunk_results)
        )
        context = truncate_to_tokens(context, max(context_budget, 100))

        user_msg = question_prefix + context

        answer = await call_llm(
            system=_FOLLOWUP_SYSTEM,
            user=user_msg,
            tracker=tracker,
            step="followup_rag",
        )

        if "INSUFFICIENT_CONTEXT" not in answer.upper():
            used_retrieval = True
        else:
            log.info("RAG context insufficient, falling back to web search")
            answer = ""
            all_sources = []

    if not used_retrieval:
        log.info("Performing targeted web search for follow-up")
        result = await research_subquery(req.query, tracker)
        all_sources = result.sources
        answer = result.summary

    elapsed = time.perf_counter() - t0
    log.info("=== FOLLOW-UP COMPLETE in %.1fs ===", elapsed)

    return FollowUpResponse(
        answer=answer,
        is_new_research=False,
        used_retrieval=used_retrieval,
        retrieved_chunks=len(chunk_results),
        sources=all_sources,
        sub_queries=[],
        memory_state={},
        budget=tracker.summary(),
        elapsed_seconds=round(elapsed, 1),
        session_id=req.session_id,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def ui():
    """Serve the research agent UI."""
    with open("agent/static/index.html") as f:
        return HTMLResponse(content=f.read())
