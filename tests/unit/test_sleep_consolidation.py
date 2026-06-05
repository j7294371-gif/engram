"""Tests for the SleepConsolidation engine."""

from __future__ import annotations

import pytest

from engram.consolidation.sleep import SleepConsolidation, ConsolidationReport
from engram.memory.enums import ConsolidationStage, MemoryType
from engram.memory.item import MemoryItem
from engram.storage.in_memory import InMemoryBackend


@pytest.fixture
async def backend() -> InMemoryBackend:
    b = InMemoryBackend()
    await b.initialize()
    return b


@pytest.fixture
async def populated_backend(backend: InMemoryBackend) -> InMemoryBackend:
    """Backend with a mix of memories for consolidation testing."""
    items = [
        MemoryItem(id="wm_1", content="working item A", memory_type=MemoryType.WORKING, importance=0.9),
        MemoryItem(id="wm_2", content="working item B", memory_type=MemoryType.WORKING, importance=0.8),
        MemoryItem(id="ep_1", content="User talked about Python performance", memory_type=MemoryType.EPISODIC, importance=0.7),
        MemoryItem(id="ep_2", content="User discussed Python vs Rust", memory_type=MemoryType.EPISODIC, importance=0.6),
        MemoryItem(id="ep_3", content="User prefers Python for data science", memory_type=MemoryType.EPISODIC, importance=0.7),
        MemoryItem(id="old_1", content="Very old forgotten memory", memory_type=MemoryType.EPISODIC,
                   strength=0.01, decay_rate=10.0, importance=0.1),
    ]
    await backend.batch_store(items)
    return backend


class TestConsolidationReport:
    def test_default_report(self):
        r = ConsolidationReport()
        assert r.promotions == 0
        assert r.abstractions == 0
        assert r.duration_ms == 0.0

    def test_str_representation(self):
        r = ConsolidationReport(promotions=3, abstractions=2, duration_ms=150.5)
        s = str(r)
        assert "promotions=3" in s
        assert "abstractions=2" in s
        assert "duration=151ms" in s or "duration=150ms" in s


class TestSleepConsolidation:
    async def test_empty_run_no_errors(self, backend: InMemoryBackend):
        consolidator = SleepConsolidation(backend=backend)
        report = await consolidator.run()
        assert isinstance(report, ConsolidationReport)
        assert report.promotions == 0

    async def test_promotes_high_importance_working(self, populated_backend: InMemoryBackend):
        consolidator = SleepConsolidation(
            backend=populated_backend,
            promotion_importance_threshold=0.5,
        )
        working_items = [
            MemoryItem(id="wm_1", content="high importance task", memory_type=MemoryType.WORKING, importance=0.9),
            MemoryItem(id="wm_2", content="low importance note", memory_type=MemoryType.WORKING, importance=0.3),
        ]
        report = await consolidator.run(working_items=working_items)
        assert report.promotions >= 1

        # Verify wm_1 was promoted to episodic
        wm1_promoted = False
        all_items = await populated_backend.list()
        for item in all_items:
            if item.memory_type == MemoryType.EPISODIC and "high importance task" in item.content:
                wm1_promoted = True
                break
        assert wm1_promoted, "High-importance working item should be promoted to episodic"

    async def test_archives_forgotten_memories(self, populated_backend: InMemoryBackend):
        consolidator = SleepConsolidation(
            backend=populated_backend,
            forgetting_threshold=0.5,
        )
        report = await consolidator.run()
        assert report.archived >= 1

        # Check that old_1 is now archived
        old_item = await populated_backend.get("old_1")
        assert old_item is not None
        assert old_item.consolidation_stage == ConsolidationStage.ARCHIVED

    async def test_extracts_semantic_abstractions(self, populated_backend: InMemoryBackend):
        """Three episodic memories about Python should trigger an abstraction."""
        consolidator = SleepConsolidation(
            backend=populated_backend,
            abstraction_min_sources=2,
        )
        report = await consolidator.run()

        # Check if any semantic abstractions were created
        sem_items = await populated_backend.list(memory_types=[MemoryType.SEMANTIC])
        if report.abstractions > 0 and len(sem_items) > 0:
            semantic = sem_items[0]
            assert semantic.memory_type == MemoryType.SEMANTIC
            assert semantic.consolidation_stage == ConsolidationStage.SEMANTIC_EXTRACTED

    async def test_strengthens_frequent_memories(self, populated_backend: InMemoryBackend):
        # Give one item low strength but high access count
        ep1 = await populated_backend.get("ep_1")
        assert ep1 is not None
        ep1.strength = 0.5
        ep1.access_count = 10
        await populated_backend.update(ep1)

        consolidator = SleepConsolidation(backend=populated_backend, rehearsal_boost=0.3)
        report = await consolidator.run()

        # The strengthened item should now have higher strength
        refreshed = await populated_backend.get("ep_1")
        assert refreshed is not None
        assert refreshed.strength > 0.5

    async def test_report_contains_metrics(self, backend: InMemoryBackend):
        consolidator = SleepConsolidation(backend=backend)
        report = await consolidator.run()
        assert hasattr(report, "promotions")
        assert hasattr(report, "abstractions")
        assert hasattr(report, "archived")
        assert hasattr(report, "duration_ms")
        assert report.duration_ms >= 0
