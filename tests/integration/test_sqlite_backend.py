"""Integration tests for the SQLite storage backend.

Tests the full StorageBackend contract against SQLite.
Uses ``:memory:`` for fast, isolated test databases.
"""

from __future__ import annotations

import os
import tempfile
from typing import Generator

import pytest

from memore.exceptions import DuplicateMemoryError, MemoryNotFoundError
from memore.memory.enums import ConsolidationStage, MemoryType
from memore.memory.item import MemoryItem
from memore.storage.sqlite import SQLiteBackend


@pytest.fixture
def backend() -> SQLiteBackend:
    b = SQLiteBackend(path=":memory:")
    b.initialize()
    return b


@pytest.fixture
def file_backend() -> Generator[SQLiteBackend, None, None]:
    """Backend backed by a temporary file (for persistence tests)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    b = SQLiteBackend(path=tmp.name)
    b.initialize()
    yield b
    b.close()
    os.unlink(tmp.name)


class TestLifecycle:
    def test_initialize_creates_tables(self, backend: SQLiteBackend):
        """Tables should exist after initialization."""
        import sqlite3
        conn = sqlite3.connect(":memory:")
        # This test uses the default :memory: backend
        stats = backend.stats()
        assert stats["total"] == 0

    def test_close_and_reopen(self, file_backend: SQLiteBackend):
        """Data should persist across close/reopen cycles."""
        item = MemoryItem(id="persist_1", content="persistent", memory_type=MemoryType.EPISODIC)
        file_backend.store(item)
        file_backend.close()

        # Reopen
        file_backend.initialize()
        retrieved = file_backend.get("persist_1")
        assert retrieved is not None
        assert retrieved.content == "persistent"


class TestCRUD:
    def test_store_and_get(self, backend: SQLiteBackend):
        item = MemoryItem(id="sql_1", content="hello from SQLite", memory_type=MemoryType.EPISODIC)
        mid = backend.store(item)
        assert mid == "sql_1"
        retrieved = backend.get("sql_1")
        assert retrieved is not None
        assert retrieved.content == "hello from SQLite"

    def test_store_duplicate_raises(self, backend: SQLiteBackend):
        item = MemoryItem(id="dup_1", content="first", memory_type=MemoryType.EPISODIC)
        backend.store(item)
        with pytest.raises(DuplicateMemoryError):
            backend.store(item)

    def test_get_nonexistent(self, backend: SQLiteBackend):
        result = backend.get("does_not_exist")
        assert result is None

    def test_update(self, backend: SQLiteBackend):
        item = MemoryItem(id="upd_1", content="original", memory_type=MemoryType.EPISODIC)
        backend.store(item)
        item.content = "updated"
        backend.update(item)
        retrieved = backend.get("upd_1")
        assert retrieved is not None
        assert retrieved.content == "updated"

    def test_update_nonexistent_raises(self, backend: SQLiteBackend):
        item = MemoryItem(id="no_exist", content="ghost", memory_type=MemoryType.EPISODIC)
        with pytest.raises(MemoryNotFoundError):
            backend.update(item)

    def test_delete(self, backend: SQLiteBackend):
        item = MemoryItem(id="del_1", content="delete me", memory_type=MemoryType.EPISODIC)
        backend.store(item)
        backend.delete("del_1")
        assert backend.get("del_1") is None

    def test_delete_is_idempotent(self, backend: SQLiteBackend):
        backend.delete("does_not_exist")  # should not raise


class TestBatch:
    def test_batch_store(self, backend: SQLiteBackend):
        items = [
            MemoryItem(id=f"batch_{i}", content=f"item {i}", memory_type=MemoryType.EPISODIC)
            for i in range(5)
        ]
        backend.batch_store(items)
        for item in items:
            retrieved = backend.get(item.id)
            assert retrieved is not None

    def test_batch_store_with_duplicate(self, backend: SQLiteBackend):
        items = [
            MemoryItem(id="dup", content="first", memory_type=MemoryType.EPISODIC),
            MemoryItem(id="dup", content="second", memory_type=MemoryType.EPISODIC),
        ]
        with pytest.raises(DuplicateMemoryError):
            backend.batch_store(items)


class TestTags:
    def test_tags_roundtrip(self, backend: SQLiteBackend):
        item = MemoryItem(
            id="tag_1", content="tagged item", memory_type=MemoryType.EPISODIC,
            tags=["important", "python", "test"],
        )
        backend.store(item)
        retrieved = backend.get("tag_1")
        assert retrieved is not None
        assert "important" in retrieved.tags
        assert "python" in retrieved.tags
        assert "test" in retrieved.tags

    def test_update_tags(self, backend: SQLiteBackend):
        item = MemoryItem(
            id="tag_2", content="update tags", memory_type=MemoryType.EPISODIC,
            tags=["old"],
        )
        backend.store(item)
        item.tags = ["new"]
        backend.update(item)
        retrieved = backend.get("tag_2")
        assert retrieved is not None
        assert "new" in retrieved.tags
        assert "old" not in retrieved.tags


class TestSearch:
    def test_keyword_search(self, backend: SQLiteBackend):
        items = [
            MemoryItem(id="s1", content="Python is great for data science",
                      memory_type=MemoryType.EPISODIC),
            MemoryItem(id="s2", content="Rust is great for systems",
                      memory_type=MemoryType.EPISODIC),
        ]
        backend.batch_store(items)
        results = backend.search("Python")
        assert len(results) >= 1
        assert any(r.id == "s1" for r in results)

    def test_search_with_type_filter(self, backend: SQLiteBackend):
        backend.store(MemoryItem(id="ep_1", content="event", memory_type=MemoryType.EPISODIC))
        backend.store(MemoryItem(id="sem_1", content="fact", memory_type=MemoryType.SEMANTIC))
        results = backend.search("", memory_types=[MemoryType.SEMANTIC])
        assert len(results) >= 1
        assert results[0].id == "sem_1"

    def test_search_excludes_archived(self, backend: SQLiteBackend):
        item = MemoryItem(
            id="arch_1", content="old stuff", memory_type=MemoryType.EPISODIC,
            consolidation_stage=ConsolidationStage.ARCHIVED,
        )
        backend.store(item)
        results = backend.search("old")
        assert len(results) == 0

    def test_search_includes_archived_when_requested(self, backend: SQLiteBackend):
        item = MemoryItem(
            id="arch_2", content="forgotten memory", memory_type=MemoryType.EPISODIC,
            consolidation_stage=ConsolidationStage.ARCHIVED,
        )
        backend.store(item)
        results = backend.search("forgotten", include_archived=True)
        assert len(results) >= 1
        assert results[0].id == "arch_2"


class TestList:
    def test_list_pagination(self, backend: SQLiteBackend):
        for i in range(10):
            backend.store(
                MemoryItem(id=f"list_{i}", content=f"item {i}", memory_type=MemoryType.EPISODIC)
            )
        page1 = backend.list(limit=3, offset=0)
        page2 = backend.list(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0].id != page2[0].id

    def test_list_by_type(self, backend: SQLiteBackend):
        backend.store(MemoryItem(id="ep_1", content="event", memory_type=MemoryType.EPISODIC))
        backend.store(MemoryItem(id="sem_1", content="fact", memory_type=MemoryType.SEMANTIC))
        results = backend.list(memory_types=[MemoryType.SEMANTIC])
        assert len(results) == 1
        assert results[0].id == "sem_1"


class TestAssociations:
    def test_add_and_get(self, backend: SQLiteBackend):
        a = MemoryItem(id="a", content="A", memory_type=MemoryType.EPISODIC)
        b = MemoryItem(id="b", content="B", memory_type=MemoryType.EPISODIC)
        backend.batch_store([a, b])
        backend.add_association("a", "b", strength=0.8)
        assoc = backend.get_associated("a")
        assert "b" in assoc

    def test_remove_association(self, backend: SQLiteBackend):
        a = MemoryItem(id="a", content="A", memory_type=MemoryType.EPISODIC)
        b = MemoryItem(id="b", content="B", memory_type=MemoryType.EPISODIC)
        backend.batch_store([a, b])
        backend.add_association("a", "b", strength=0.8)
        backend.remove_association("a", "b")
        assoc = backend.get_associated("a")
        assert "b" not in assoc

    def test_association_chain(self, backend: SQLiteBackend):
        items = [
            MemoryItem(id="a", content="A", memory_type=MemoryType.EPISODIC),
            MemoryItem(id="b", content="B", memory_type=MemoryType.EPISODIC),
            MemoryItem(id="c", content="C", memory_type=MemoryType.EPISODIC),
        ]
        backend.batch_store(items)
        backend.add_association("a", "b", strength=0.9)
        backend.add_association("b", "c", strength=0.8)
        assoc = backend.get_associated("a", max_depth=2)
        assert "b" in assoc
        assert "c" in assoc
        # C should have lower activation (two hops with decay)
        assert assoc["c"] < assoc["b"]


class TestStats:
    def test_stats_empty(self, backend: SQLiteBackend):
        stats = backend.stats()
        assert stats["total"] == 0

    def test_stats_with_data(self, backend: SQLiteBackend):
        backend.store(MemoryItem(id="ep_1", content="event", memory_type=MemoryType.EPISODIC))
        backend.store(MemoryItem(id="sem_1", content="fact", memory_type=MemoryType.SEMANTIC))
        stats = backend.stats()
        assert stats["total"] == 2
        assert stats.get("episodic", 0) == 1
        assert stats.get("semantic", 0) == 1
        assert stats["avg_importance"] > 0


class TestForgetting:
    def test_get_decaying(self, backend: SQLiteBackend):
        from datetime import datetime, timedelta, timezone
        old_item = MemoryItem(
            id="old", content="very old", memory_type=MemoryType.EPISODIC,
            strength=0.1, decay_rate=10.0,
            last_rehearsed_at=datetime.now(timezone.utc) - timedelta(days=30),
        )
        fresh_item = MemoryItem(id="fresh", content="brand new", memory_type=MemoryType.EPISODIC)
        backend.batch_store([old_item, fresh_item])
        decaying = backend.get_decaying(threshold=0.5)
        assert any(i.id == "old" for i in decaying)


class TestMetadata:
    def test_metadata_roundtrip(self, backend: SQLiteBackend):
        item = MemoryItem(
            id="meta_1", content="with metadata", memory_type=MemoryType.EPISODIC,
            metadata={"source": "test", "version": 1, "tags": ["a", "b"]},
        )
        backend.store(item)
        retrieved = backend.get("meta_1")
        assert retrieved is not None
        assert retrieved.metadata["source"] == "test"
        assert retrieved.metadata["version"] == 1
        assert retrieved.metadata["tags"] == ["a", "b"]

    def test_update_metadata(self, backend: SQLiteBackend):
        item = MemoryItem(id="meta_2", content="update meta", memory_type=MemoryType.EPISODIC)
        backend.store(item)
        item.metadata = {"updated": True}
        backend.update(item)
        retrieved = backend.get("meta_2")
        assert retrieved is not None
        assert retrieved.metadata["updated"] is True


class TestClear:
    def test_clear_removes_all(self, backend: SQLiteBackend):
        backend.store(MemoryItem(id="c1", content="test", memory_type=MemoryType.EPISODIC))
        backend.store(MemoryItem(id="c2", content="test2", memory_type=MemoryType.EPISODIC))
        backend.clear()
        stats = backend.stats()
        assert stats["total"] == 0
