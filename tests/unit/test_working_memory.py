"""Tests for WorkingMemory — bounded capacity and attention weighting."""

from __future__ import annotations

import pytest

from memore.memory.enums import MemoryType
from memore.memory.item import MemoryItem
from memore.pipeline.working_memory import WorkingMemory


@pytest.fixture
def wm() -> WorkingMemory:
    return WorkingMemory(capacity=3)


class TestBasicOperations:
    def test_add_item(self, wm: WorkingMemory):
        item = MemoryItem(id="w1", content="task context", memory_type=MemoryType.WORKING)
        wm.add(item)
        assert wm.size() == 1

    def test_get_item(self, wm: WorkingMemory):
        item = MemoryItem(id="w1", content="task context", memory_type=MemoryType.WORKING)
        wm.add(item)
        retrieved = wm.get("w1")
        assert retrieved is not None
        assert retrieved.content == "task context"

    def test_get_nonexistent(self, wm: WorkingMemory):
        assert wm.get("nonexistent") is None

    def test_remove_item(self, wm: WorkingMemory):
        item = MemoryItem(id="w1", content="remove me", memory_type=MemoryType.WORKING)
        wm.add(item)
        removed = wm.remove("w1")
        assert removed is not None
        assert wm.size() == 0


class TestCapacity:
    def test_evicts_lowest_attention(self, wm: WorkingMemory):
        """When full, the item with lowest attention should be evicted."""
        items = []
        for i in range(4):
            item = MemoryItem(
                id=f"w{i}", content=f"item {i}", memory_type=MemoryType.WORKING,
                attention_weight=(i + 1) * 0.2,
            )
            items.append(item)
            wm.add(item)

        assert wm.size() == 3
        # Item 0 had the lowest attention weight (0.2) — should be evicted
        assert wm.get("w0") is None
        # Others should remain
        for i in range(1, 4):
            assert wm.get(f"w{i}") is not None

    def test_is_full(self, wm: WorkingMemory):
        capacity = wm.capacity
        for i in range(capacity):
            assert not wm.is_full()
            wm.add(MemoryItem(id=f"w{i}", content=f"item {i}", memory_type=MemoryType.WORKING))
        assert wm.is_full()


class TestAttention:
    def test_get_context_returns_sorted(self, wm: WorkingMemory):
        for i in range(3):
            item = MemoryItem(
                id=f"w{i}", content=f"item {i}", memory_type=MemoryType.WORKING,
                attention_weight=1.0 - i * 0.3,
            )
            wm.add(item)
        context = wm.get_context(window_size=3)
        assert context[0].attention_weight >= context[1].attention_weight

    def test_get_boosts_attention(self, wm: WorkingMemory):
        item = MemoryItem(id="w1", content="focus", memory_type=MemoryType.WORKING, attention_weight=0.5)
        wm.add(item)
        wm.get("w1")
        # Should have boosted attention
        assert wm.get("w1").attention_weight > 0.5

    def test_focus_sets_high_attention(self, wm: WorkingMemory):
        item = wm.focus("urgent task")
        assert item.attention_weight == 1.0

    def test_attend_to(self, wm: WorkingMemory):
        wm.add(MemoryItem(id="w1", content="test", memory_type=MemoryType.WORKING))
        wm.attend_to("w1", 0.9)
        # Use get_context to avoid the attention boost from get()
        ctx = wm.get_context(window_size=5)
        found = [i for i in ctx if i.id == "w1"]
        assert len(found) == 1
        assert found[0].attention_weight == 0.9


class TestEvictionCallback:
    def test_on_evict_called(self):
        evicted_items = []

        def on_evict(item):
            evicted_items.append(item)

        wm = WorkingMemory(capacity=2, on_evict=on_evict)
        for i in range(3):
            wm.add(MemoryItem(id=f"w{i}", content=f"item {i}", memory_type=MemoryType.WORKING))
        assert len(evicted_items) == 1

    def test_clear(self, wm: WorkingMemory):
        wm.add(MemoryItem(id="w1", content="test", memory_type=MemoryType.WORKING))
        wm.clear()
        assert wm.size() == 0
