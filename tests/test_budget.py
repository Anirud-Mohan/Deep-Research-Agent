"""Tests for budget tracking and constraint enforcement."""

from agent.budget import BudgetTracker, count_tokens, truncate_to_tokens


def test_count_tokens_basic():
    assert count_tokens("hello") >= 1
    assert count_tokens("") == 0


def test_count_tokens_longer():
    text = "The quick brown fox jumps over the lazy dog"
    tokens = count_tokens(text)
    assert 5 < tokens < 20


def test_truncate_no_op_when_within_limit():
    text = "hello world"
    assert truncate_to_tokens(text, 100) == text


def test_truncate_cuts_to_limit():
    text = "one two three four five six seven eight nine ten"
    truncated = truncate_to_tokens(text, 3)
    assert count_tokens(truncated) <= 3
    assert len(truncated) < len(text)


def test_tracker_records_calls():
    t = BudgetTracker(max_context_tokens=2048)
    t.record(prompt_tokens=100, completion_tokens=50, step="test_step")
    t.record(prompt_tokens=200, completion_tokens=80, step="test_step_2")

    assert t.total_calls == 2
    assert t.total_prompt_tokens == 300
    assert t.total_completion_tokens == 130
    assert t.total_tokens == 430


def test_tracker_summary_shape():
    t = BudgetTracker()
    t.record(prompt_tokens=10, completion_tokens=5, step="x")
    s = t.summary()
    assert "total_llm_calls" in s
    assert "per_call_breakdown" in s
    assert s["total_llm_calls"] == 1


def test_fit_segments_within_budget():
    t = BudgetTracker(max_context_tokens=500)
    segments = [
        ("a", "short text"),
        ("b", "another short text"),
    ]
    kept = t.fit_segments(segments)
    assert len(kept) == 2


def test_fit_segments_drops_overflow():
    t = BudgetTracker(max_context_tokens=10)
    long_text = "word " * 200
    segments = [
        ("important", "hi"),
        ("overflow", long_text),
    ]
    kept = t.fit_segments(segments)
    assert len(kept) <= 2
    assert kept[0][0] == "important"
