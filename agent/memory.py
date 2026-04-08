"""Memory manager — the heart of the agent.

Three tiers:
1. **Vector store** (ChromaDB) — long-term storage of all research chunks.
2. **Summarisation cascade** — progressive compression pipeline.
3. **Working-memory buffer** — the token-limited window the LLM sees.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import chromadb

from agent.budget import BudgetTracker, count_tokens, truncate_to_tokens
from agent.config import settings
from agent.llm import call_llm

# ---------------------------------------------------------------------------
# Vector store (Tier 1 — long-term)
# ---------------------------------------------------------------------------

_chroma_client = chromadb.Client()

COLLECTION_NAME = "research_chunks"


def _get_collection() -> chromadb.Collection:
    return _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def store_chunks(texts: list[str], metadatas: list[dict] | None = None) -> None:
    """Embed and persist *texts* in the vector store."""
    col = _get_collection()
    ids = [uuid.uuid4().hex for _ in texts]
    kwargs: dict = {"documents": texts, "ids": ids}
    if metadatas:
        kwargs["metadatas"] = metadatas
    col.add(**kwargs)


def retrieve_chunks(query: str, top_k: int = settings.vector_top_k) -> list[str]:
    """Return the *top_k* most relevant stored chunks for *query*."""
    col = _get_collection()
    if col.count() == 0:
        return []
    results = col.query(query_texts=[query], n_results=min(top_k, col.count()))
    return results["documents"][0] if results["documents"] else []


def reset_vector_store() -> None:
    """Wipe the vector store (used between sessions)."""
    try:
        _chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Summarisation cascade (Tier 2 — compression)
# ---------------------------------------------------------------------------

_SUMMARISE_SOURCE_PROMPT = (
    "You are a research assistant. Condense the following source material "
    "into a concise summary of at most {target} tokens. "
    "Preserve key facts, statistics, and claims. Omit filler."
)

_SUMMARISE_SUBQUERY_PROMPT = (
    "You are a research assistant. Below are summaries from several sources "
    "that answer the sub-question: \"{subquery}\"\n"
    "Synthesise them into a single coherent summary of at most {target} tokens."
)

_COMPRESS_DRAFT_PROMPT = (
    "Compress the following draft answer, preserving every key claim and "
    "conclusion. Target length: at most {target} tokens."
)


async def summarise_source(
    raw_text: str,
    tracker: BudgetTracker,
) -> str:
    """Level-0 → Level-1: compress one raw source page."""
    text = truncate_to_tokens(raw_text, tracker.max_context_tokens)
    return await call_llm(
        system=_SUMMARISE_SOURCE_PROMPT.format(target=settings.source_summary_tokens),
        user=text,
        tracker=tracker,
        step="summarise_source",
    )


async def summarise_subquery(
    subquery: str,
    source_summaries: list[str],
    tracker: BudgetTracker,
) -> str:
    """Level-1 → Level-2: merge per-source summaries for one sub-query."""
    combined = f"Sub-question: {subquery}\n\n"
    for i, s in enumerate(source_summaries, 1):
        combined += f"Source {i}:\n{s}\n\n"

    combined = truncate_to_tokens(combined, tracker.max_context_tokens)

    return await call_llm(
        system=_SUMMARISE_SUBQUERY_PROMPT.format(
            subquery=subquery, target=settings.subquery_summary_tokens
        ),
        user=combined,
        tracker=tracker,
        step="summarise_subquery",
    )


async def compress_draft(
    draft: str,
    target_tokens: int,
    tracker: BudgetTracker,
) -> str:
    """Compress the running draft answer when it grows too large."""
    return await call_llm(
        system=_COMPRESS_DRAFT_PROMPT.format(target=target_tokens),
        user=draft,
        tracker=tracker,
        step="compress_draft",
    )


# ---------------------------------------------------------------------------
# Working-memory buffer (Tier 3 — active context)
# ---------------------------------------------------------------------------


@dataclass
class WorkingMemory:
    """Fixed-size buffer that holds the context visible to the LLM.

    The buffer stores sub-query findings in insertion order.  When a new
    finding would exceed the budget, the oldest findings are evicted (FIFO).
    """

    max_tokens: int = settings.max_context_tokens
    findings: list[dict] = field(default_factory=list)

    def add_finding(self, subquery: str, summary: str) -> None:
        self.findings.append({"subquery": subquery, "summary": summary})
        self._evict()

    def _evict(self) -> None:
        """Drop oldest findings until total fits in budget."""
        while self._total_tokens() > self.max_tokens and len(self.findings) > 1:
            self.findings.pop(0)

    def _total_tokens(self) -> int:
        return sum(count_tokens(f["summary"]) for f in self.findings)

    def render(self) -> str:
        """Format buffer contents as a single string for the LLM."""
        parts: list[str] = []
        for f in self.findings:
            parts.append(f"### {f['subquery']}\n{f['summary']}")
        return "\n\n".join(parts)

    @property
    def token_count(self) -> int:
        return count_tokens(self.render()) if self.findings else 0

    def state_snapshot(self) -> dict:
        return {
            "num_findings": len(self.findings),
            "token_count": self.token_count,
            "max_tokens": self.max_tokens,
            "findings": self.findings,
        }
