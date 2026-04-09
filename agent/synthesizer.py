"""Iterative-refinement answer synthesiser.

Builds the final answer one sub-query at a time.  At each step the
running draft + the next sub-query's findings must fit inside the
per-call token budget.  If the draft grows too large it is compressed
before the next iteration.
"""

from __future__ import annotations

from agent.budget import BudgetTracker, count_tokens
from agent.config import settings
from agent.llm import call_llm
from agent.memory import compress_draft

_REFINE_SYSTEM = (
    "You are a deep-research writer producing detailed, comprehensive reports.\n\n"
    "You will receive:\n"
    "1. The original question.\n"
    "2. Your answer so far (may be empty on the first pass).\n"
    "3. New research findings for a specific sub-question.\n\n"
    "Incorporate the new findings into the answer. Your output must be:\n"
    "- Thorough: Cover every angle with in-depth analysis, specific data "
    "points, statistics, dates, and named examples wherever available.\n"
    "- Well-structured: Use markdown headings (##, ###), bullet points, "
    "and numbered lists to organise the information clearly.\n"
    "- Analytical: Don't just list facts — explain implications, compare "
    "perspectives, highlight trade-offs, and draw connections.\n"
    "- Detailed: Aim for depth over brevity. A research report should be "
    "substantive enough to genuinely inform the reader.\n\n"
    "IMPORTANT: Write your response as a polished final answer directly. "
    "Do NOT include labels like 'Draft', 'Current Draft', 'Draft Answer', "
    "or any other meta-headers. Just write the answer."
)


async def synthesise(
    original_query: str,
    subquery_findings: list[dict[str, str]],
    tracker: BudgetTracker,
) -> str:
    """Produce a final answer via iterative refinement.

    *subquery_findings* is a list of ``{"subquery": ..., "summary": ...}``
    dicts in the order they were researched.
    """
    draft = ""

    for finding in subquery_findings:
        user_block = _build_user_message(original_query, draft, finding)
        user_tokens = count_tokens(user_block)

        if user_tokens > tracker.max_context_tokens:
            draft = await compress_draft(
                draft,
                target_tokens=settings.max_draft_tokens,
                tracker=tracker,
            )
            user_block = _build_user_message(original_query, draft, finding)

        draft = await call_llm(
            system=_REFINE_SYSTEM,
            user=user_block,
            tracker=tracker,
            step=f"refine:{finding['subquery'][:40]}",
            max_output_tokens=settings.synthesis_max_tokens,
        )

    return draft


def _build_user_message(
    original_query: str,
    draft: str,
    finding: dict[str, str],
) -> str:
    parts = [f"## Original question\n{original_query}"]
    if draft:
        parts.append(f"## Answer so far\n{draft}")
    parts.append(
        f"## New findings for: {finding['subquery']}\n{finding['summary']}"
    )
    return "\n\n".join(parts)
