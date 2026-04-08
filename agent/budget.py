"""Token budget tracker and enforcement.

Every LLM call in the pipeline goes through `BudgetTracker` so we can
prove that the memory constraint is respected at every step.
"""

from __future__ import annotations

import tiktoken

from agent.config import settings

try:
    _encoder = tiktoken.encoding_for_model(settings.llm_model)
except KeyError:
    _encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the exact token count for *text* using the model's tokenizer."""
    return len(_encoder.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* to at most *max_tokens*, splitting on token boundaries."""
    tokens = _encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _encoder.decode(tokens[:max_tokens])


class BudgetTracker:
    """Tracks cumulative token usage across an entire research session.

    Also provides a helper to *fit* a list of text segments into the
    per-call context budget, trimming the lowest-priority segments first.
    """

    def __init__(self, max_context_tokens: int = settings.max_context_tokens):
        self.max_context_tokens = max_context_tokens
        self.calls: list[dict] = []

    # -- recording -------------------------------------------------------------

    def record(self, *, prompt_tokens: int, completion_tokens: int, step: str):
        """Record one LLM invocation."""
        self.calls.append(
            {
                "step": step,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )

    @property
    def total_prompt_tokens(self) -> int:
        return sum(c["prompt_tokens"] for c in self.calls)

    @property
    def total_completion_tokens(self) -> int:
        return sum(c["completion_tokens"] for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def total_calls(self) -> int:
        return len(self.calls)

    def summary(self) -> dict:
        return {
            "total_llm_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "max_context_tokens_per_call": self.max_context_tokens,
            "per_call_breakdown": self.calls,
        }

    # -- budget fitting --------------------------------------------------------

    def fit_segments(
        self,
        segments: list[tuple[str, str]],
        reserved: int = 0,
    ) -> list[tuple[str, str]]:
        """Return as many segments as fit in the context budget.

        *segments* is a list of ``(label, text)`` pairs ordered from
        **highest** to **lowest** priority.  We greedily add segments
        until the budget is exhausted, then drop the rest.

        *reserved* tokens are subtracted up-front (e.g. for a system prompt
        whose size is already known).
        """
        budget = self.max_context_tokens - reserved
        kept: list[tuple[str, str]] = []
        used = 0
        for label, text in segments:
            tok = count_tokens(text)
            if used + tok <= budget:
                kept.append((label, text))
                used += tok
            else:
                remaining = budget - used
                if remaining > 50:
                    kept.append((label, truncate_to_tokens(text, remaining)))
                break
        return kept
