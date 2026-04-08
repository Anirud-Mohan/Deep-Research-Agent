# Evaluation — Architecture Trade-offs

## 1. Constraint Definition

| Constraint | Value | Rationale |
|---|---|---|
| **Max research-context tokens per LLM call** | 2,048 | Tight enough to force real memory management; loose enough to produce useful answers. System prompt tokens are excluded — only the dynamic research content counts against this budget. |

Every LLM call is routed through a single `call_llm` wrapper that measures the user-message token count with `tiktoken` and **raises an error** if it exceeds the limit, guaranteeing enforcement at runtime.

---

## 2. Memory Architecture — Why a Hybrid?

The agent uses three memory tiers:

| Tier | Mechanism | Role |
|---|---|---|
| **Long-term** | ChromaDB vector store | Stores every raw research chunk as an embedding. Never occupies context window tokens. |
| **Compression** | Summarisation cascade (3 levels) | Progressively reduces raw search results → source summaries → per-sub-query findings. |
| **Active** | Working-memory buffer (FIFO eviction) | Fixed-size buffer that holds only what the LLM needs right now. |

### Why not pure Vector RAG?

Vector search alone retrieves the *k* most similar chunks to the current query. Under a 2K budget that means ~3-5 short chunks. The problem: chunks are decontextualised fragments — the LLM receives isolated sentences with no narrative structure. Answer quality degrades when the question needs synthesis across many sources.

### Why not pure summarisation?

Summarisation compresses well but is lossy. If a later synthesis step needs a specific statistic that was compressed away, it's gone. Keeping the raw chunks in a vector store means we can *re-retrieve* precise details on demand.

### The hybrid advantage

The cascade handles the "wide view" (compressed summaries of everything), while the vector store provides "zoom-in" capability (retrieve specific chunks when detail is needed). Both remain within the 2K budget because summaries are short and vector retrievals are top-k bounded.

---

## 3. Synthesis Strategy — Iterative Refinement

The agent builds its final answer one sub-query at a time:

```
draft₀ = ""
for each sub-query finding fᵢ:
    if tokens(draftᵢ₋₁) + tokens(fᵢ) > 2048:
        draftᵢ₋₁ = compress(draftᵢ₋₁)
    draftᵢ = LLM(draftᵢ₋₁ + fᵢ, "refine the answer")
return draftₙ
```

### Alternatives considered

| Strategy | Pros | Cons |
|---|---|---|
| **Iterative refinement** (chosen) | Coherent narrative; each step sees the full evolving answer; naturally stays in budget | Later sub-queries can overshadow earlier ones if draft is compressed |
| **Hierarchical merge** | Balanced information weight | More LLM calls; loses coherence across branches; harder to debug |
| **Section-by-section** | Simple; easy to parallelise | No cross-section awareness; disjointed output |

Iterative refinement was chosen because it produces the most coherent prose and mirrors how a human researcher would incrementally build understanding.

### Draft self-compression

When the running draft exceeds `max_draft_tokens` (1,200), it is compressed before the next iteration. This means some nuance from early sub-queries is lost — a deliberate trade-off for staying within the memory constraint. The compression target (1,200 tokens) was chosen to leave ~800 tokens of headroom for the next sub-query's findings.

---

## 4. Query Decomposition

Complex queries are broken into 2-5 sub-questions by a single LLM call. The decomposer prompt asks for a JSON array, with fallback regex parsing for robustness.

**Trade-off**: More sub-queries → deeper research but more LLM calls and more compression. Fewer sub-queries → cheaper and faster but shallower. The 2-5 range balances depth with budget.

---

## 5. Where the Architecture Breaks

Transparency about limitations:

1. **Numerical precision**: Summarisation can distort or drop specific numbers. A query like "What was Tesla's exact revenue in Q3 2024?" may return a rounded or approximate figure after compression.

2. **Very broad queries**: A question touching 10+ distinct topics will exhaust the sub-query budget. The agent gracefully handles this by researching as many sub-queries as feasible and synthesising from what it has, but coverage will be incomplete.

3. **Contradictory sources**: The cascade merges source summaries without explicit conflict resolution. If sources disagree, the synthesised answer may present one side or awkwardly blend both.

4. **Temporal sensitivity**: ChromaDB stores chunks without timestamps. For rapidly evolving topics, stale chunks from a previous session (if the store is not reset) could contaminate results.

---

## 6. Follow-Up Queries and RAG Retrieval

The agent supports conversational follow-ups via the `/followup` endpoint. This is where the vector store proves its value beyond simple storage:

1. User asks an initial research question → full pipeline runs, chunks stored in ChromaDB.
2. User asks a follow-up → agent retrieves the top-k most relevant chunks from ChromaDB.
3. If chunks are found, the agent synthesises from retrieved context **without any web search** — faster and cheaper.
4. If the vector store is empty, it falls back to a standard web search.

This demonstrates the complete hybrid memory architecture: the summarisation cascade handles the initial research, and the vector store enables efficient follow-up retrieval. Both paths respect the 2,048-token constraint.

**Trade-off**: Follow-up answers from retrieval are limited to what was already researched. If the follow-up introduces a genuinely new topic, the retrieved chunks may be tangentially relevant at best. A more sophisticated approach would score the retrieval quality and selectively fall back to web search — a future improvement.

---

## 7. Token Budget Analysis

Observed token usage from a real research session (3 sub-queries, Llama 3.3 70B via Groq):

| Step | Calls | ~Tokens per call (prompt) | Total |
|---|---|---|---|
| Decompose | 1 | ~110 | ~110 |
| Summarise sources (2 per sub-query) | 6 | ~100-380 | ~940 |
| Summarise sub-queries | 3 | ~170-270 | ~625 |
| Iterative refinement | 3 | ~250-570 | ~1,250 |
| **Total** | **13** | | **~2,925 prompt / 4,578 total** |

**Key observation**: The maximum prompt tokens in any single call was **570** — well under the 2,048 limit. The summarisation cascade is effective at keeping each call's context compact.

Follow-up queries using RAG retrieval typically use **1 LLM call** with ~200-500 prompt tokens.

---

## 8. LLM Provider Flexibility

The agent uses the OpenAI client library with a configurable `base_url`, making it compatible with any OpenAI-compatible API:

| Provider | Model | Notes |
|---|---|---|
| Groq | Llama 3.3 70B | Free tier, ~0.5-1s per call, used in development |
| OpenAI | GPT-4o-mini | Reliable, good JSON output for decomposer |
| OpenRouter | Any (Kimi K2, Mistral, etc.) | Unified access to many models |
| Together AI | Llama, Qwen, DeepSeek | Free credits on signup |

Switching providers requires only changing `OPENAI_API_KEY` and `LLM_BASE_URL` in the `.env` file — no code changes.

Token counting uses `tiktoken` with `cl100k_base` encoding as a fallback for non-OpenAI models. This is an approximation (~5% variance), but sufficient for budget enforcement since the constraint has built-in headroom.

---

## 9. Design Decisions Summary

| Decision | Choice | Key reason |
|---|---|---|
| Vector DB | ChromaDB (in-process) | Zero infrastructure, built-in embeddings, easy to demo |
| LLM | Configurable (default: Llama 3.3 70B via Groq) | Free, fast, swappable via env vars |
| Search | Tavily | Returns clean text (no HTML parsing), built for AI agents |
| Orchestration | n8n + FastAPI | n8n handles visual routing; FastAPI handles compute — clean separation |
| Token counting | tiktoken (exact or cl100k_base fallback) | No estimation — exact or near-exact counts for budget enforcement |
| Eviction policy | FIFO on working memory | Simple, predictable, easy to explain; relevance-based eviction is a future improvement |
| Follow-up strategy | RAG retrieval first, web search fallback | Demonstrates the vector store's retrieval value |
| UI architecture | Chat-based with metrics sidebar | Natural conversation flow; constraint compliance always visible |
