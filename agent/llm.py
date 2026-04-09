"""Thin wrapper around the OpenAI chat-completion API.

Every LLM call in the project goes through `call_llm` so that:
1. Token usage is recorded by the BudgetTracker.
2. The research-context portion of the prompt is verified against the
   per-call token budget *before* the request is sent.
3. Rate-limit errors (429) are retried with exponential backoff.
"""

from __future__ import annotations

import asyncio
import logging
import time

from openai import AsyncOpenAI, RateLimitError

from agent.budget import BudgetTracker, count_tokens
from agent.config import settings

log = logging.getLogger("agent")

_client = AsyncOpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.llm_base_url or None,
)

# Maximum number of retry attempts on 429 errors.
_MAX_RETRIES = 4
# Initial wait in seconds; doubles on each retry (1s, 2s, 4s, 8s).
_RETRY_BASE_SECONDS = 1.0


async def call_llm(
    *,
    system: str,
    user: str,
    tracker: BudgetTracker,
    step: str,
    temperature: float = 0.2,
    max_output_tokens: int = 1024,
    model: str | None = None,
    enforce_budget: bool = True,
) -> str:
    """Send a chat completion and return the assistant content.

    *model* overrides ``settings.llm_model`` for this call, allowing
    lightweight steps to use a smaller/faster model while synthesis uses
    the primary (large) model.

    *enforce_budget* controls whether the 2048-token research-context
    constraint is checked against the user message.  Set to ``False`` for
    internal pipeline calls (e.g. ``compress_draft``) whose input is
    agent-generated, not web-fetched, so the constraint's purpose does not
    apply.  All calls that process externally-retrieved research content
    must leave this at the default ``True``.

    When the resolved model is the primary model and
    ``settings.llm_is_reasoning_model`` is True, the call is automatically
    adapted for OpenAI reasoning models: ``temperature`` is omitted and
    ``max_completion_tokens`` is used instead of ``max_tokens``.

    Raises ``ValueError`` if *enforce_budget* is True and the user message
    exceeds the tracker's per-call budget.

    Automatically retries up to ``_MAX_RETRIES`` times on 429 rate-limit
    errors using exponential backoff.
    """
    resolved_model = model or settings.llm_model

    # A call is "reasoning" if it uses the primary model and that model is
    # configured as a reasoning model. Summary-model calls always use standard
    # params regardless of llm_is_reasoning_model.
    is_reasoning = (
        resolved_model == settings.llm_model and settings.llm_is_reasoning_model
    )

    context_tokens = count_tokens(user)
    if enforce_budget and context_tokens > tracker.max_context_tokens:
        raise ValueError(
            f"[{step}] Research context is {context_tokens} tokens, "
            f"exceeding the {tracker.max_context_tokens}-token budget."
        )

    log.info(
        "[%s|%s%s] Calling LLM (%d context tokens)...",
        step, resolved_model, "/reasoning" if is_reasoning else "", context_tokens,
    )
    t0 = time.perf_counter()

    last_error: RateLimitError | None = None
    for attempt in range(_MAX_RETRIES + 1):
        if attempt > 0:
            wait = _RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            log.warning(
                "[%s] Rate limited (attempt %d/%d). Waiting %.0fs before retry...",
                step, attempt, _MAX_RETRIES, wait,
            )
            await asyncio.sleep(wait)

        try:
            if is_reasoning:
                # Reasoning models: no temperature, use max_completion_tokens.
                response = await _client.chat.completions.create(
                    model=resolved_model,
                    max_completion_tokens=max_output_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
            else:
                response = await _client.chat.completions.create(
                    model=resolved_model,
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
            break
        except RateLimitError as e:
            last_error = e
            if attempt == _MAX_RETRIES:
                log.error(
                    "[%s] Rate limit exceeded after %d retries. "
                    "Consider upgrading your API plan or waiting before retrying.",
                    step, _MAX_RETRIES,
                )
                raise
    else:
        raise last_error  # type: ignore[misc]

    elapsed = time.perf_counter() - t0
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    reasoning_tokens = (
        getattr(usage.completion_tokens_details, "reasoning_tokens", 0)
        if usage and usage.completion_tokens_details
        else 0
    )

    tracker.record(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        step=step,
        context_tokens=context_tokens,
        budgeted=enforce_budget,
    )

    if reasoning_tokens:
        log.info(
            "[%s] Done in %.1fs  (prompt=%d, completion=%d [reasoning=%d] tokens)",
            step, elapsed, prompt_tokens, completion_tokens, reasoning_tokens,
        )
    else:
        log.info(
            "[%s] Done in %.1fs  (prompt=%d, completion=%d tokens)",
            step, elapsed, prompt_tokens, completion_tokens,
        )

    return response.choices[0].message.content or ""
