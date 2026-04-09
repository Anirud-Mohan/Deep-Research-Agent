"""Web search and chunking for a single sub-query."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from tavily import AsyncTavilyClient

from agent.budget import BudgetTracker
from agent.config import settings
from agent.memory import chunk_text, store_chunks, summarise_source, summarise_subquery

log = logging.getLogger("agent")


@dataclass
class SubQueryResult:
    sub_query: str
    summary: str
    sources: list[dict[str, str]] = field(default_factory=list)


async def research_subquery(
    subquery: str,
    tracker: BudgetTracker,
) -> SubQueryResult:
    """Search the web for *subquery*, summarise findings, store in vector DB.

    Returns a SubQueryResult with the summary and source URLs.
    """
    log.info("Searching web for: %s", subquery)
    t0 = time.perf_counter()

    client = AsyncTavilyClient(api_key=settings.tavily_api_key)
    response = await client.search(
        query=subquery,
        max_results=settings.max_search_results,
        include_raw_content=True,
        search_depth="advanced",
        days=settings.search_recency_days,
    )

    log.info("Search returned %d results in %.1fs", len(response.get("results", [])), time.perf_counter() - t0)

    snippets: list[str] = []
    sources: list[dict[str, str]] = []
    all_chunks: list[str] = []
    all_chunk_metas: list[dict] = []

    for i, result in enumerate(response.get("results", [])):
        snippet = result.get("content", "")
        raw_content = result.get("raw_content") or snippet
        if not snippet:
            continue

        snippets.append(snippet)
        sources.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
        })

        title = result.get("title", "")
        url = result.get("url", "")
        chunks = chunk_text(raw_content)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_chunk_metas.append({
                "subquery": subquery,
                "source_idx": i,
                "title": title,
                "url": url,
            })

    if not snippets:
        return SubQueryResult(
            sub_query=subquery,
            summary=f"No results found for: {subquery}",
        )

    if all_chunks:
        store_chunks(texts=all_chunks, metadatas=all_chunk_metas)
        log.info("Stored %d chunks from %d sources in vector store", len(all_chunks), len(snippets))

    source_summaries = await asyncio.gather(
        *[summarise_source(raw, tracker) for raw in snippets]
    )

    subquery_summary = await summarise_subquery(
        subquery, list(source_summaries), tracker
    )

    return SubQueryResult(
        sub_query=subquery,
        summary=subquery_summary,
        sources=sources,
    )
