"""Tests for the SensoryBuffer (ring buffer with TTL)."""

from __future__ import annotations

import time

import pytest

from engram.memory.enums import MemoryType
from engram.memory.item import MemoryItem
from engram.pipeline.sensory_buffer import SensoryBuffer


@pytest.fixture
def buffer() -> SensoryBuffer:
    return SensoryBuffer(capacity=5, ttl_seconds=60.0)


@pytest.fixture
def fast_decay_buffer() -> SensoryBuffer:
    return SensoryBuffer(capacity=5, ttl_seconds=0.1)


class TestAddAndRetrieve:
    def test_add_item(self, buffer: SensoryBuffer):
        item = MemoryItem(id="s1", content="sensory input", memory_type=MemoryType.SENSORY)
        buffer.add(item)
        assert buffer.size() == 1

    def test_retrieve_by_id(self, buffer: SensoryBuffer):
        item = MemoryItem(id="s1", content="sensory input", memory_type=MemoryType.SENSORY)
        buffer.add(item)
        retrieved = buffer.get("s1")
        assert retrieved is not None
        assert retrieved.content == "sensory input"

    def test_retrieve_nonexistent(self, buffer: SensoryBuffer):
        assert buffer.get("nonexistent") is None


class TestCapacity:
    def test_evicts_oldest_when_full(self, buffer: SensoryBuffer):
        for i in range(6):
            item = MemoryItem(id=f"s{i}", content=f"item {i}", memory_type=MemoryType.SENSORY)
            buffer.add(item)
        # The buffer capacity is 5, so item 0 should be evicted
        assert buffer.size() == 5
        assert buffer.get("s0") is None

    def test_latest_items_available(self, buffer: SensoryBuffer):
        for i in range(6):
            item = MemoryItem(id=f"s{i}", content=f"item {i}", memory_type=MemoryType.SENSORY)
            buffer.add(item)
        # Items 1-5 should still be there
        for i in range(1, 6):
            assert buffer.get(f"s{i}") is not None


class TestTTL:
    def test_items_expire(self, fast_decay_buffer: SensoryBuffer):
        item = MemoryItem(id="s1", content="ephemeral", memory_type=MemoryType.SENSORY)
        fast_decay_buffer.add(item)
        assert fast_decay_buffer.get("s1") is not None
        time.sleep(0.15)
        assert fast_decay_buffer.get("s1") is None

    def test_expired_items_not_in_all(self, fast_decay_buffer: SensoryBuffer):
        item = MemoryItem(id="s1", content="ephemeral", memory_type=MemoryType.SENSORY)
        fast_decay_buffer.add(item)
        time.sleep(0.15)
        assert len(fast_decay_buffer.all()) == 0


class TestSearch:
    def test_keyword_search(self, buffer: SensoryBuffer):
        buffer.add(MemoryItem(id="s1", content="Python programming", memory_type=MemoryType.SENSORY))
        buffer.add(MemoryItem(id="s2", content="Rust systems programming", memory_type=MemoryType.SENSORY))
        results = buffer.search("Python")
        assert len(results) == 1
        assert results[0].id == "s1"

    def test_search_empty_buffer(self, buffer: SensoryBuffer):
        assert len(buffer.search("anything")) == 0


class TestClear:
    def test_clear_removes_all(self, buffer: SensoryBuffer):
        for i in range(3):
            buffer.add(MemoryItem(id=f"s{i}", content=f"item {i}", memory_type=MemoryType.SENSORY))
        buffer.clear()
        assert buffer.size() == 0
