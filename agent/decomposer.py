"""Break a complex research question into focused sub-queries."""

from __future__ import annotations

import json
import re

from agent.budget import BudgetTracker
from agent.llm import call_llm

_SYSTEM = (
    "You are a research planning assistant. Given a complex question, "
    "decompose it into 2-5 focused, self-contained sub-questions that "
    "together fully address the original query.\n\n"
    "Return ONLY a JSON array of strings. Example:\n"
    '[" sub-question 1", "sub-question 2", "sub-question 3"]'
)


async def decompose(query: str, tracker: BudgetTracker) -> list[str]:
    """Return a list of sub-queries derived from *query*."""
    raw = await call_llm(
        system=_SYSTEM,
        user=query,
        tracker=tracker,
        step="decompose",
        temperature=0.2,
    )

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(s) for s in parsed]
        except json.JSONDecodeError:
            pass

    return [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]
