"""Tests for memory manager — vector store and working memory buffer."""

from agent.budget import count_tokens
from agent.memory import (
    WorkingMemory,
    reset_vector_store,
    retrieve_chunks,
    store_chunks,
)


class TestVectorStore:
    def setup_method(self):
        reset_vector_store()

    def test_store_and_retrieve(self):
        store_chunks(
            ["AI in healthcare improves diagnostics", "Manufacturing uses robots"],
            metadatas=[{"subquery": "ai healthcare"}, {"subquery": "ai manufacturing"}],
        )
        results = retrieve_chunks("healthcare AI diagnostics", top_k=1)
        assert len(results) == 1
        text = results[0]["text"].lower()
        assert "healthcare" in text or "diagnostics" in text

    def test_retrieve_empty(self):
        results = retrieve_chunks("anything")
        assert results == []

    def test_reset_clears_data(self):
        store_chunks(["some data"])
        reset_vector_store()
        results = retrieve_chunks("some data")
        assert results == []


class TestWorkingMemory:
    def test_add_and_render(self):
        wm = WorkingMemory(max_tokens=2048)
        wm.add_finding("What is AI?", "AI is artificial intelligence.")
        rendered = wm.render()
        assert "What is AI?" in rendered
        assert "AI is artificial intelligence." in rendered

    def test_eviction_under_pressure(self):
        wm = WorkingMemory(max_tokens=50)
        wm.add_finding("q1", "first finding with some detail about the topic")
        wm.add_finding("q2", "second finding with some other detail about things")
        wm.add_finding("q3", "third finding that is also quite long and detailed")
        assert wm.token_count <= 50

    def test_state_snapshot(self):
        wm = WorkingMemory()
        wm.add_finding("q1", "summary1")
        snap = wm.state_snapshot()
        assert snap["num_findings"] == 1
        assert snap["max_tokens"] == 2048
        assert "token_count" in snap

    def test_empty_memory(self):
        wm = WorkingMemory()
        assert wm.token_count == 0
        assert wm.render() == ""
        assert wm.state_snapshot()["num_findings"] == 0
