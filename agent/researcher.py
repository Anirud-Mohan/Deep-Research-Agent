"""Web search and chunking for a single sub-query."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from tavily import AsyncTavilyClient

from agent.budget import BudgetTracker
from agent.config import settings
from agent.memory import store_chunks, summarise_source, summarise_subquery

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
        include_raw_content=False,
    )

    log.info("Search returned %d results in %.1fs", len(response.get("results", [])), time.perf_counter() - t0)

    raw_texts: list[str] = []
    sources: list[dict[str, str]] = []
    for result in response.get("results", []):
        text = result.get("content", "")
        if text:
            raw_texts.append(text)
            sources.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
            })

    if not raw_texts:
        return SubQueryResult(
            sub_query=subquery,
            summary=f"No results found for: {subquery}",
        )

    store_chunks(
        texts=raw_texts,
        metadatas=[{"subquery": subquery, "source_idx": i} for i in range(len(raw_texts))],
    )

    source_summaries = await asyncio.gather(
        *[summarise_source(raw, tracker) for raw in raw_texts]
    )

    subquery_summary = await summarise_subquery(
        subquery, list(source_summaries), tracker
    )

    return SubQueryResult(
        sub_query=subquery,
        summary=subquery_summary,
        sources=sources,
    )
