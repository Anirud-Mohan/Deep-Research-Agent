"""Thin wrapper around the OpenAI chat-completion API.

Every LLM call in the project goes through `call_llm` so that:
1. Token usage is recorded by the BudgetTracker.
2. The research-context portion of the prompt is verified against the
   per-call token budget *before* the request is sent.
"""

from __future__ import annotations

import logging
import time

from openai import AsyncOpenAI

from agent.budget import BudgetTracker, count_tokens
from agent.config import settings

log = logging.getLogger("agent")

_client = AsyncOpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.llm_base_url or None,
)


async def call_llm(
    *,
    system: str,
    user: str,
    tracker: BudgetTracker,
    step: str,
    temperature: float = 0.2,
    max_output_tokens: int = 1024,
) -> str:
    """Send a chat completion and return the assistant content.

    Raises ``ValueError`` if the *user* message (research context)
    exceeds the tracker's per-call budget.
    """
    context_tokens = count_tokens(user)
    if context_tokens > tracker.max_context_tokens:
        raise ValueError(
            f"[{step}] Research context is {context_tokens} tokens, "
            f"exceeding the {tracker.max_context_tokens}-token budget."
        )

    log.info("[%s] Calling LLM (%d context tokens)...", step, context_tokens)
    t0 = time.perf_counter()

    response = await _client.chat.completions.create(
        model=settings.llm_model,
        temperature=temperature,
        max_tokens=max_output_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    elapsed = time.perf_counter() - t0
    usage = response.usage
    tracker.record(
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        step=step,
    )

    log.info(
        "[%s] Done in %.1fs  (prompt=%d, completion=%d tokens)",
        step,
        elapsed,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )

    return response.choices[0].message.content or ""
