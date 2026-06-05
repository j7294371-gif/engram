"""Integration tests for the InMemory storage backend.

These tests exercise the full StorageBackend contract against
the InMemory implementation. As other backends are added,
they will be parameterized here too.
"""

from __future__ import annotations

import pytest

from memore.exceptions import DuplicateMemoryError, MemoryNotFoundError
from memore.memory.enums import MemoryType
from memore.memory.item import MemoryItem
from memore.storage.in_memory import InMemoryBackend


@pytest.fixture
async def backend() -> InMemoryBackend:
    b = InMemoryBackend()
    await b.initialize()
    return b


class TestCRUD:
    async def test_store_and_get(self, backend: InMemoryBackend):
        item = MemoryItem(id="crud_1", content="hello", memory_type=MemoryType.EPISODIC)
        mid = await backend.store(item)
        assert mid == "crud_1"
        retrieved = await backend.get("crud_1")
        assert retrieved is not None
        assert retrieved.content == "hello"

    async def test_store_duplicate_raises(self, backend: InMemoryBackend):
        item = MemoryItem(id="dup_1", content="first", memory_type=MemoryType.EPISODIC)
        await backend.store(item)
        with pytest.raises(DuplicateMemoryError):
            await backend.store(item)

    async def test_get_nonexistent(self, backend: InMemoryBackend):
        result = await backend.get("does_not_exist")
        assert result is None

    async def test_update(self, backend: InMemoryBackend):
        item = MemoryItem(id="upd_1", content="original", memory_type=MemoryType.EPISODIC)
        await backend.store(item)
        item.content = "updated"
        await backend.update(item)
        retrieved = await backend.get("upd_1")
        assert retrieved is not None
        assert retrieved.content == "updated"

    async def test_update_nonexistent_raises(self, backend: InMemoryBackend):
        item = MemoryItem(id="no_exist", content="ghost", memory_type=MemoryType.EPISODIC)
        with pytest.raises(MemoryNotFoundError):
            await backend.update(item)

    async def test_delete(self, backend: InMemoryBackend):
        item = MemoryItem(id="del_1", content="delete me", memory_type=MemoryType.EPISODIC)
        await backend.store(item)
        await backend.delete("del_1")
        assert await backend.get("del_1") is None

    async def test_delete_nonexistent_is_idempotent(self, backend: InMemoryBackend):
        await backend.delete("does_not_exist")  # should not raise


class TestBatch:
    async def test_batch_store(self, backend: InMemoryBackend):
        items = [
            MemoryItem(id=f"batch_{i}", content=f"item {i}", memory_type=MemoryType.EPISODIC)
            for i in range(5)
        ]
        await backend.batch_store(items)
        for item in items:
            retrieved = await backend.get(item.id)
            assert retrieved is not None

    async def test_batch_store_with_duplicate(self, backend: InMemoryBackend):
        items = [
            MemoryItem(id="dup", content="first", memory_type=MemoryType.EPISODIC),
            MemoryItem(id="dup", content="second", memory_type=MemoryType.EPISODIC),
        ]
        with pytest.raises(DuplicateMemoryError):
            await backend.batch_store(items)


class TestSearch:
    async def test_keyword_search(self, backend: InMemoryBackend):
        items = [
            MemoryItem(id="s1", content="Python is great", memory_type=MemoryType.EPISODIC, tags=["lang"]),
            MemoryItem(id="s2", content="Rust is fast", memory_type=MemoryType.EPISODIC, tags=["lang"]),
            MemoryItem(id="s3", content="JavaScript everywhere", memory_type=MemoryType.EPISODIC, tags=["lang"]),
        ]
        await backend.batch_store(items)
        results = await backend.search("Python")
        assert len(results) == 1
        assert results[0].id == "s1"

    async def test_search_by_type_filter(self, backend: InMemoryBackend):
        await backend.store(MemoryItem(id="ep_1", content="event", memory_type=MemoryType.EPISODIC))
        await backend.store(MemoryItem(id="sem_1", content="fact", memory_type=MemoryType.SEMANTIC))
        results = await backend.search("", memory_types=[MemoryType.SEMANTIC])
        assert len(results) == 1
        assert results[0].id == "sem_1"

    async def test_search_excludes_archived(self, backend: InMemoryBackend):
        from memore.memory.enums import ConsolidationStage
        item = MemoryItem(id="arch_1", content="old stuff", memory_type=MemoryType.EPISODIC,
                          consolidation_stage=ConsolidationStage.ARCHIVED)
        await backend.store(item)
        results = await backend.search("old")
        assert len(results) == 0


class TestAssociations:
    async def test_add_and_get_association(self, backend: InMemoryBackend):
        a = MemoryItem(id="a", content="A", memory_type=MemoryType.EPISODIC)
        b = MemoryItem(id="b", content="B", memory_type=MemoryType.EPISODIC)
        await backend.batch_store([a, b])
        await backend.add_association("a", "b", strength=0.8)
        associated = await backend.get_associated("a")
        assert "b" in associated
        assert associated["b"] == pytest.approx(0.8 * 0.85, rel=1e-3)

    async def test_association_chain(self, backend: InMemoryBackend):
        items = [
            MemoryItem(id="a", content="A", memory_type=MemoryType.EPISODIC),
            MemoryItem(id="b", content="B", memory_type=MemoryType.EPISODIC),
            MemoryItem(id="c", content="C", memory_type=MemoryType.EPISODIC),
        ]
        await backend.batch_store(items)
        await backend.add_association("a", "b", strength=0.9)
        await backend.add_association("b", "c", strength=0.8)
        associated = await backend.get_associated("a", max_depth=2)
        assert "b" in associated
        assert "c" in associated
        # C should have lower activation than B (two hops)
        assert associated["c"] < associated["b"]

    async def test_remove_association(self, backend: InMemoryBackend):
        a = MemoryItem(id="a", content="A", memory_type=MemoryType.EPISODIC)
        b = MemoryItem(id="b", content="B", memory_type=MemoryType.EPISODIC)
        await backend.batch_store([a, b])
        await backend.add_association("a", "b", strength=0.8)
        await backend.remove_association("a", "b")
        associated = await backend.get_associated("a")
        assert "b" not in associated


class TestStats:
    async def test_stats_empty(self, backend: InMemoryBackend):
        stats = await backend.stats()
        assert stats["total"] == 0

    async def test_stats_with_data(self, backend: InMemoryBackend):
        items = [
            MemoryItem(id="ep_1", content="event", memory_type=MemoryType.EPISODIC),
            MemoryItem(id="sem_1", content="fact", memory_type=MemoryType.SEMANTIC),
        ]
        await backend.batch_store(items)
        stats = await backend.stats()
        assert stats["total"] == 2
        assert stats["episodic"] == 1
        assert stats["semantic"] == 1
        assert stats["avg_importance"] > 0


class TestForgetting:
    async def test_get_decaying(self, backend: InMemoryBackend):
        from datetime import datetime, timedelta, timezone
        old_item = MemoryItem(
            id="old", content="very old", memory_type=MemoryType.EPISODIC,
            strength=0.1, decay_rate=10.0,
            last_rehearsed_at=datetime.now(timezone.utc) - timedelta(days=30),
        )
        fresh_item = MemoryItem(id="fresh", content="brand new", memory_type=MemoryType.EPISODIC)
        await backend.batch_store([old_item, fresh_item])
        decaying = await backend.get_decaying(threshold=0.5)
        assert any(i.id == "old" for i in decaying)
        assert not any(i.id == "fresh" for i in decaying)
