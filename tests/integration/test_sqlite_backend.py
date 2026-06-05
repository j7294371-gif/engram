"""Integration tests for the SQLite storage backend.

Tests the full StorageBackend contract against SQLite.
Uses ``:memory:`` for fast, isolated test databases.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from engram.exceptions import DuplicateMemoryError, MemoryNotFoundError
from engram.memory.enums import ConsolidationStage, MemoryType
from engram.memory.item import MemoryItem
from engram.storage.sqlite import SQLiteBackend


@pytest.fixture
async def backend() -> SQLiteBackend:
    b = SQLiteBackend(path=":memory:")
    await b.initialize()
    return b


@pytest.fixture
async def file_backend() -> SQLiteBackend:
    """Backend backed by a temporary file (for persistence tests)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    b = SQLiteBackend(path=tmp.name)
    await b.initialize()
    yield b
    await b.close()
    os.unlink(tmp.name)


class TestLifecycle:
    async def test_initialize_creates_tables(self, backend: SQLiteBackend):
        """Tables should exist after initialization."""
        import sqlite3
        conn = sqlite3.connect(":memory:")
        # This test uses the default :memory: backend
        stats = await backend.stats()
        assert stats["total"] == 0

    async def test_close_and_reopen(self, file_backend: SQLiteBackend):
        """Data should persist across close/reopen cycles."""
        item = MemoryItem(id="persist_1", content="persistent", memory_type=MemoryType.EPISODIC)
        await file_backend.store(item)
        await file_backend.close()

        # Reopen
        await file_backend.initialize()
        retrieved = await file_backend.get("persist_1")
        assert retrieved is not None
        assert retrieved.content == "persistent"


class TestCRUD:
    async def test_store_and_get(self, backend: SQLiteBackend):
        item = MemoryItem(id="sql_1", content="hello from SQLite", memory_type=MemoryType.EPISODIC)
        mid = await backend.store(item)
        assert mid == "sql_1"
        retrieved = await backend.get("sql_1")
        assert retrieved is not None
        assert retrieved.content == "hello from SQLite"

    async def test_store_duplicate_raises(self, backend: SQLiteBackend):
        item = MemoryItem(id="dup_1", content="first", memory_type=MemoryType.EPISODIC)
        await backend.store(item)
        with pytest.raises(DuplicateMemoryError):
            await backend.store(item)

    async def test_get_nonexistent(self, backend: SQLiteBackend):
        result = await backend.get("does_not_exist")
        assert result is None

    async def test_update(self, backend: SQLiteBackend):
        item = MemoryItem(id="upd_1", content="original", memory_type=MemoryType.EPISODIC)
        await backend.store(item)
        item.content = "updated"
        await backend.update(item)
        retrieved = await backend.get("upd_1")
        assert retrieved is not None
        assert retrieved.content == "updated"

    async def test_update_nonexistent_raises(self, backend: SQLiteBackend):
        item = MemoryItem(id="no_exist", content="ghost", memory_type=MemoryType.EPISODIC)
        with pytest.raises(MemoryNotFoundError):
            await backend.update(item)

    async def test_delete(self, backend: SQLiteBackend):
        item = MemoryItem(id="del_1", content="delete me", memory_type=MemoryType.EPISODIC)
        await backend.store(item)
        await backend.delete("del_1")
        assert await backend.get("del_1") is None

    async def test_delete_is_idempotent(self, backend: SQLiteBackend):
        await backend.delete("does_not_exist")  # should not raise


class TestBatch:
    async def test_batch_store(self, backend: SQLiteBackend):
        items = [
            MemoryItem(id=f"batch_{i}", content=f"item {i}", memory_type=MemoryType.EPISODIC)
            for i in range(5)
        ]
        await backend.batch_store(items)
        for item in items:
            retrieved = await backend.get(item.id)
            assert retrieved is not None

    async def test_batch_store_with_duplicate(self, backend: SQLiteBackend):
        items = [
            MemoryItem(id="dup", content="first", memory_type=MemoryType.EPISODIC),
            MemoryItem(id="dup", content="second", memory_type=MemoryType.EPISODIC),
        ]
        with pytest.raises(DuplicateMemoryError):
            await backend.batch_store(items)


class TestTags:
    async def test_tags_roundtrip(self, backend: SQLiteBackend):
        item = MemoryItem(
            id="tag_1", content="tagged item", memory_type=MemoryType.EPISODIC,
            tags=["important", "python", "test"],
        )
        await backend.store(item)
        retrieved = await backend.get("tag_1")
        assert retrieved is not None
        assert "important" in retrieved.tags
        assert "python" in retrieved.tags
        assert "test" in retrieved.tags

    async def test_update_tags(self, backend: SQLiteBackend):
        item = MemoryItem(
            id="tag_2", content="update tags", memory_type=MemoryType.EPISODIC,
            tags=["old"],
        )
        await backend.store(item)
        item.tags = ["new"]
        await backend.update(item)
        retrieved = await backend.get("tag_2")
        assert retrieved is not None
        assert "new" in retrieved.tags
        assert "old" not in retrieved.tags


class TestSearch:
    async def test_keyword_search(self, backend: SQLiteBackend):
        items = [
            MemoryItem(id="s1", content="Python is great for data science",
                      memory_type=MemoryType.EPISODIC),
            MemoryItem(id="s2", content="Rust is great for systems",
                      memory_type=MemoryType.EPISODIC),
        ]
        await backend.batch_store(items)
        results = await backend.search("Python")
        assert len(results) >= 1
        assert any(r.id == "s1" for r in results)

    async def test_search_with_type_filter(self, backend: SQLiteBackend):
        await backend.store(MemoryItem(id="ep_1", content="event", memory_type=MemoryType.EPISODIC))
        await backend.store(MemoryItem(id="sem_1", content="fact", memory_type=MemoryType.SEMANTIC))
        results = await backend.search("", memory_types=[MemoryType.SEMANTIC])
        assert len(results) >= 1
        assert results[0].id == "sem_1"

    async def test_search_excludes_archived(self, backend: SQLiteBackend):
        item = MemoryItem(
            id="arch_1", content="old stuff", memory_type=MemoryType.EPISODIC,
            consolidation_stage=ConsolidationStage.ARCHIVED,
        )
        await backend.store(item)
        results = await backend.search("old")
        assert len(results) == 0

    async def test_search_includes_archived_when_requested(self, backend: SQLiteBackend):
        item = MemoryItem(
            id="arch_2", content="forgotten memory", memory_type=MemoryType.EPISODIC,
            consolidation_stage=ConsolidationStage.ARCHIVED,
        )
        await backend.store(item)
        results = await backend.search("forgotten", include_archived=True)
        assert len(results) >= 1
        assert results[0].id == "arch_2"


class TestList:
    async def test_list_pagination(self, backend: SQLiteBackend):
        for i in range(10):
            await backend.store(
                MemoryItem(id=f"list_{i}", content=f"item {i}", memory_type=MemoryType.EPISODIC)
            )
        page1 = await backend.list(limit=3, offset=0)
        page2 = await backend.list(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0].id != page2[0].id

    async def test_list_by_type(self, backend: SQLiteBackend):
        await backend.store(MemoryItem(id="ep_1", content="event", memory_type=MemoryType.EPISODIC))
        await backend.store(MemoryItem(id="sem_1", content="fact", memory_type=MemoryType.SEMANTIC))
        results = await backend.list(memory_types=[MemoryType.SEMANTIC])
        assert len(results) == 1
        assert results[0].id == "sem_1"


class TestAssociations:
    async def test_add_and_get(self, backend: SQLiteBackend):
        a = MemoryItem(id="a", content="A", memory_type=MemoryType.EPISODIC)
        b = MemoryItem(id="b", content="B", memory_type=MemoryType.EPISODIC)
        await backend.batch_store([a, b])
        await backend.add_association("a", "b", strength=0.8)
        assoc = await backend.get_associated("a")
        assert "b" in assoc

    async def test_remove_association(self, backend: SQLiteBackend):
        a = MemoryItem(id="a", content="A", memory_type=MemoryType.EPISODIC)
        b = MemoryItem(id="b", content="B", memory_type=MemoryType.EPISODIC)
        await backend.batch_store([a, b])
        await backend.add_association("a", "b", strength=0.8)
        await backend.remove_association("a", "b")
        assoc = await backend.get_associated("a")
        assert "b" not in assoc

    async def test_association_chain(self, backend: SQLiteBackend):
        items = [
            MemoryItem(id="a", content="A", memory_type=MemoryType.EPISODIC),
            MemoryItem(id="b", content="B", memory_type=MemoryType.EPISODIC),
            MemoryItem(id="c", content="C", memory_type=MemoryType.EPISODIC),
        ]
        await backend.batch_store(items)
        await backend.add_association("a", "b", strength=0.9)
        await backend.add_association("b", "c", strength=0.8)
        assoc = await backend.get_associated("a", max_depth=2)
        assert "b" in assoc
        assert "c" in assoc
        # C should have lower activation (two hops with decay)
        assert assoc["c"] < assoc["b"]


class TestStats:
    async def test_stats_empty(self, backend: SQLiteBackend):
        stats = await backend.stats()
        assert stats["total"] == 0

    async def test_stats_with_data(self, backend: SQLiteBackend):
        await backend.store(MemoryItem(id="ep_1", content="event", memory_type=MemoryType.EPISODIC))
        await backend.store(MemoryItem(id="sem_1", content="fact", memory_type=MemoryType.SEMANTIC))
        stats = await backend.stats()
        assert stats["total"] == 2
        assert stats.get("episodic", 0) == 1
        assert stats.get("semantic", 0) == 1
        assert stats["avg_importance"] > 0


class TestForgetting:
    async def test_get_decaying(self, backend: SQLiteBackend):
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


class TestMetadata:
    async def test_metadata_roundtrip(self, backend: SQLiteBackend):
        item = MemoryItem(
            id="meta_1", content="with metadata", memory_type=MemoryType.EPISODIC,
            metadata={"source": "test", "version": 1, "tags": ["a", "b"]},
        )
        await backend.store(item)
        retrieved = await backend.get("meta_1")
        assert retrieved is not None
        assert retrieved.metadata["source"] == "test"
        assert retrieved.metadata["version"] == 1
        assert retrieved.metadata["tags"] == ["a", "b"]

    async def test_update_metadata(self, backend: SQLiteBackend):
        item = MemoryItem(id="meta_2", content="update meta", memory_type=MemoryType.EPISODIC)
        await backend.store(item)
        item.metadata = {"updated": True}
        await backend.update(item)
        retrieved = await backend.get("meta_2")
        assert retrieved is not None
        assert retrieved.metadata["updated"] is True


class TestClear:
    async def test_clear_removes_all(self, backend: SQLiteBackend):
        await backend.store(MemoryItem(id="c1", content="test", memory_type=MemoryType.EPISODIC))
        await backend.store(MemoryItem(id="c2", content="test2", memory_type=MemoryType.EPISODIC))
        await backend.clear()
        stats = await backend.stats()
        assert stats["total"] == 0
