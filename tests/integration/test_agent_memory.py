"""Integration tests for the AgentMemory facade.

Tests the full pipeline: remember → recall → context → consolidate.
"""

from __future__ import annotations

import pytest

from memore import AgentMemory, Config
from memore.memory.enums import MemoryType
from memore.memory.item import MemoryItem


@pytest.fixture
def memory() -> AgentMemory:
    """A fresh AgentMemory with in-memory backend for testing."""
    return AgentMemory(config=Config(auto_consolidate=False))


class TestCoreAPI:
    def test_remember_returns_id(self, memory: AgentMemory):
        mid = memory.remember("Hello world", memory_type="episodic")
        assert mid is not None
        assert isinstance(mid, str)
        assert mid.startswith("ep_")

    def test_remember_and_recall_roundtrip(self, memory: AgentMemory):
        memory.remember("User prefers Python", memory_type="semantic", tags=["python", "preference"])
        results = memory.recall("Python", memory_types=["semantic"])
        assert len(results) >= 1
        assert "Python" in results[0].content

    def test_remember_semantic_retrievable(self, memory: AgentMemory):
        memory.remember("Paris is the capital of France", memory_type="semantic")
        results = memory.recall("Paris", memory_types=["semantic"], include_working=False, include_sensory=False)
        assert len(results) >= 1

    def test_recall_empty(self, memory: AgentMemory):
        results = memory.recall("nothing")
        assert len(results) == 0

    def test_recall_with_limit(self, memory: AgentMemory):
        for i in range(10):
            memory.remember(f"Memory number {i}", memory_type="episodic")
        results = memory.recall("Memory", limit=3)
        assert len(results) <= 3


class TestWorkingMemory:
    def test_focus_adds_to_working(self, memory: AgentMemory):
        mid = memory.focus("Current task")
        assert mid is not None
        context = memory.get_context()
        assert any(item.id == mid for item in context)

    def test_get_context_returns_working_items(self, memory: AgentMemory):
        memory.focus("Task 1")
        memory.focus("Task 2")
        context = memory.get_context(window_size=5)
        assert len(context) >= 2

    def test_attend_to_adjusts_weight(self, memory: AgentMemory):
        mid = memory.focus("Important task")
        memory.attend_to(mid, 0.9)
        context = memory.get_context()
        assert context[0].attention_weight == 0.9


class TestAssociations:
    def test_associate_and_retrieve(self, memory: AgentMemory):
        mid1 = memory.remember("Python is great", memory_type="semantic")
        mid2 = memory.remember("Django is a Python framework", memory_type="semantic")
        memory.associate(mid1, mid2, strength=0.9)
        associated = memory.retrieve_associated(mid1)
        assert len(associated) >= 1
        assert any(item.id == mid2 for item, _ in associated)


class TestEmotionalTagging:
    def test_remember_with_emotion(self, memory: AgentMemory):
        mid = memory.remember(
            "Exciting project launch!",
            memory_type="episodic",
            emotional_valence=0.9,
            emotional_arousal=0.8,
        )
        item = memory.get(mid)
        assert item is not None
        assert item.valence == 0.9
        assert item.arousal == 0.8

    def test_tag_emotion_after_fact(self, memory: AgentMemory):
        mid = memory.remember("A neutral event", memory_type="episodic")
        memory.tag_emotion(mid, valence=0.5, arousal=0.3)
        item = memory.get(mid)
        assert item is not None
        assert item.valence == 0.5
        assert item.arousal == 0.3


class TestConsolidation:
    def test_consolidate_returns_report(self, memory: AgentMemory):
        for i in range(5):
            memory.remember(f"Memory {i}", memory_type="episodic")
        report = memory.consolidate()
        assert "promotions" in report
        assert "archived" in report

    def test_rehearse_boosts_strength(self, memory: AgentMemory):
        mid = memory.remember("Important fact", memory_type="semantic", importance=0.5)
        memory.rehearse(mid)
        item = memory.get(mid)
        assert item is not None
        assert item.strength > 0.5


class TestSearchAndTag:
    def test_search_by_type(self, memory: AgentMemory):
        memory.remember("Event happened", memory_type="episodic")
        memory.remember("General knowledge", memory_type="semantic")
        sem_results = memory.search("knowledge", memory_types=["semantic"])
        ep_results = memory.search("Event", memory_types=["episodic"])
        assert len(sem_results) >= 1
        assert len(ep_results) >= 1

    def test_tagging(self, memory: AgentMemory):
        mid = memory.remember("Some content", memory_type="episodic")
        memory.tag(mid, "important", "review")
        item = memory.get(mid)
        assert item is not None
        assert "important" in item.tags
        assert "review" in item.tags

    def test_forget_archives(self, memory: AgentMemory):
        from memore.memory.enums import ConsolidationStage
        mid = memory.remember("To be forgotten", memory_type="episodic")
        memory.forget(mid)
        item = memory.get(mid)
        assert item is not None
        assert item.consolidation_stage == ConsolidationStage.ARCHIVED

    def test_stats(self, memory: AgentMemory):
        memory.remember("First", memory_type="episodic")
        memory.remember("Second", memory_type="semantic")
        stats = memory.stats()
        assert stats["total"] >= 2

    def test_clear(self, memory: AgentMemory):
        memory.remember("Something", memory_type="episodic")
        memory.clear()
        stats = memory.stats()
        assert stats["total"] == 0


class TestEdgeCases:
    def test_sensory_memory(self, memory: AgentMemory):
        mid = memory.remember("Quick perception", memory_type="sensory")
        assert mid is not None
        assert mid.startswith("se_")
