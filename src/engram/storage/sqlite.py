"""SQLite storage backend — persistent memory with FTS5 full-text search.

Default production backend. Zero additional dependencies beyond Python's
built-in sqlite3 module. Uses FTS5 for keyword search and a JSON-based
approach for flexible metadata queries.

Schema:
  - memories: core table with all MemoryItem fields
  - tags: memory_id → tag mapping (many-to-many)
  - associations: directed graph with strength
  - abstraction_sources: provenance tracking for consolidation
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Collection, Dict, Iterator, List, Optional, Set

from engram.exceptions import DuplicateMemoryError, MemoryNotFoundError
from engram.memory.enums import ConsolidationStage, MemoryType
from engram.memory.item import MemoryItem
from engram.storage.base import StorageBackend


class SQLiteBackend(StorageBackend):
    """Persistent memory storage using SQLite with FTS5 full-text search.

    Thread-safe (uses a per-thread connection pool with WAL mode).
    """

    def __init__(self, path: str = ":memory:") -> None:
        self._path = path
        self._lock = threading.Lock()
        self._local = threading.local()
        self._initialized = False

    # ── Connection management ───────────────────────────────────

    @property
    def _conn(self) -> sqlite3.Connection:
        """Get or create a thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._path, check_same_thread=False)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        """Transaction context manager."""
        conn = self._conn
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    # ── Lifecycle ────────────────────────────────────────────────

    async def initialize(self) -> None:
        if self._initialized:
            return
        with self._lock:
            with self._tx() as cur:
                cur.executescript(_SCHEMA_SQL)
            self._initialized = True

    async def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
        self._initialized = False

    # ── CRUD ─────────────────────────────────────────────────────

    async def store(self, item: MemoryItem) -> str:
        with self._lock:
            with self._tx() as cur:
                # Check for duplicate
                cur.execute("SELECT 1 FROM memories WHERE id = ?", (item.id,))
                if cur.fetchone() is not None:
                    raise DuplicateMemoryError(f"Memory {item.id!r} already exists.")

                self._insert_item(cur, item)
                self._insert_tags(cur, item.id, item.tags)
                self._insert_associations(cur, item.id, item.associations)
            return item.id

    async def batch_store(self, items: List[MemoryItem]) -> None:
        with self._lock:
            with self._tx() as cur:
                for item in items:
                    cur.execute("SELECT 1 FROM memories WHERE id = ?", (item.id,))
                    if cur.fetchone() is not None:
                        raise DuplicateMemoryError(f"Memory {item.id!r} already exists.")
                    self._insert_item(cur, item)
                    self._insert_tags(cur, item.id, item.tags)
                    self._insert_associations(cur, item.id, item.associations)

    async def get(self, memory_id: str) -> Optional[MemoryItem]:
        with self._lock:
            with self._tx() as cur:
                cur.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
                row = cur.fetchone()
                if row is None:
                    return None
                return self._row_to_item(cur, row)

    async def update(self, item: MemoryItem) -> None:
        with self._lock:
            with self._tx() as cur:
                cur.execute("SELECT 1 FROM memories WHERE id = ?", (item.id,))
                if cur.fetchone() is None:
                    raise MemoryNotFoundError(f"Memory {item.id!r} not found.")

                cur.execute(
                    """UPDATE memories SET
                        content=?, memory_type=?, memory_subtype=?,
                        last_accessed_at=?, access_count=?,
                        strength=?, decay_rate=?, last_rehearsed_at=?,
                        valence=?, arousal=?, importance=?,
                        attention_weight=?, consolidation_stage=?,
                        consolidated_at=?, promoted_from=?,
                        metadata_json=?, source=?
                    WHERE id=?""",
                    (
                        item.content, item.memory_type.value, item.memory_subtype,
                        _ts(item.last_accessed_at), item.access_count,
                        item.strength, item.decay_rate, _ts(item.last_rehearsed_at),
                        item.valence, item.arousal, item.importance,
                        item.attention_weight, item.consolidation_stage.value,
                        _ts(item.consolidated_at) if item.consolidated_at else None,
                        item.promoted_from,
                        json.dumps(item.metadata), item.source,
                        item.id,
                    ),
                )
                # Rebuild tags
                cur.execute("DELETE FROM tags WHERE memory_id = ?", (item.id,))
                self._insert_tags(cur, item.id, item.tags)
                # Rebuild associations
                cur.execute("DELETE FROM associations WHERE source_id = ?", (item.id,))
                self._insert_associations(cur, item.id, item.associations)

    async def delete(self, memory_id: str) -> None:
        with self._lock:
            with self._tx() as cur:
                cur.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                cur.execute("DELETE FROM tags WHERE memory_id = ?", (memory_id,))
                cur.execute("DELETE FROM associations WHERE source_id=? OR target_id=?",
                           (memory_id, memory_id))

    # ── Query ────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        memory_types: Optional[Collection[MemoryType]] = None,
        limit: int = 20,
        threshold: float = 0.0,
        include_archived: bool = False,
        **metadata_filters: Any,
    ) -> List[MemoryItem]:
        with self._lock:
            with self._tx() as cur:
                where_clauses: List[str] = []
                params: List[Any] = []

                # Keyword search via LIKE
                if query:
                    like_clauses = []
                    for term in query.strip().split():
                        like_clauses.append("(m.content LIKE ? OR m.id IN (SELECT memory_id FROM tags WHERE tag LIKE ?))")
                        params.extend([f"%{term}%", f"%{term}%"])
                    where_clauses.append(f"({' OR '.join(like_clauses)})")

                # Type filter
                if memory_types:
                    placeholders = ",".join("?" for _ in memory_types)
                    where_clauses.append(f"m.memory_type IN ({placeholders})")
                    params.extend(t.value for t in memory_types)

                # Archived filter
                if not include_archived:
                    where_clauses.append("m.consolidation_stage != 'archived'")

                # Build query
                where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
                sql = f"SELECT m.* FROM memories m WHERE {where_sql} ORDER BY m.importance DESC, m.created_at DESC LIMIT ?"
                params.append(limit)

                cur.execute(sql, params)
                return [self._row_to_item(cur, row) for row in cur.fetchall()]

    async def list(
        self,
        memory_types: Optional[Collection[MemoryType]] = None,
        importance_min: float = 0.0,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        include_archived: bool = False,
        **metadata_filters: Any,
    ) -> List[MemoryItem]:
        allowed_sorts = {"created_at", "last_accessed_at", "importance", "strength", "access_count"}
        if sort_by not in allowed_sorts:
            sort_by = "created_at"

        where_clauses: List[str] = ["m.importance >= ?"]
        params: List[Any] = [importance_min]

        if memory_types:
            placeholders = ",".join("?" for _ in memory_types)
            where_clauses.append(f"m.memory_type IN ({placeholders})")
            params.extend(t.value for t in memory_types)
        if not include_archived:
            where_clauses.append("m.consolidation_stage != 'archived'")

        where_sql = " AND ".join(where_clauses)
        order = "DESC" if sort_desc else "ASC"
        sql = f"SELECT m.* FROM memories m WHERE {where_sql} ORDER BY m.{sort_by} {order} LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._lock:
            with self._tx() as cur:
                cur.execute(sql, params)
                return [self._row_to_item(cur, row) for row in cur.fetchall()]

    # ── Associations ─────────────────────────────────────────────

    async def add_association(self, source_id: str, target_id: str, strength: float = 1.0) -> None:
        with self._lock:
            with self._tx() as cur:
                cur.execute(
                    """INSERT INTO associations (source_id, target_id, strength, created_at)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(source_id, target_id) DO UPDATE SET strength = ?""",
                    (source_id, target_id, strength, _ts(datetime.now(timezone.utc)), strength),
                )

    async def remove_association(self, source_id: str, target_id: str) -> None:
        with self._lock:
            with self._tx() as cur:
                cur.execute(
                    "DELETE FROM associations WHERE source_id=? AND target_id=?",
                    (source_id, target_id),
                )

    async def get_associated(
        self,
        memory_id: str,
        max_depth: int = 2,
        min_strength: float = 0.0,
    ) -> Dict[str, float]:
        with self._lock:
            with self._tx() as cur:
                # Use recursive CTE for BFS traversal
                cur.execute(
                    """WITH RECURSIVE assoc(id, strength, depth) AS (
                           SELECT target_id, strength, 1
                           FROM associations WHERE source_id = ? AND strength >= ?
                           UNION ALL
                           SELECT a.target_id, a.strength * assoc.strength * 0.85, assoc.depth + 1
                           FROM associations a JOIN assoc ON a.source_id = assoc.id
                           WHERE assoc.depth < ? AND a.strength >= ?
                       )
                       SELECT id, MAX(strength) as activation FROM assoc
                       GROUP BY id ORDER BY activation DESC""",
                    (memory_id, min_strength, max_depth, min_strength),
                )
                return {row["id"]: row["activation"] for row in cur.fetchall()}

    # ── Forgetting ───────────────────────────────────────────────

    async def get_decaying(self, threshold: float = 0.3, limit: int = 100) -> List[MemoryItem]:
        """Return memories with low retrieval probability.

        Retrieval probability P = strength * exp(-decay_rate * hours_since_rehearsal)
        A memory is "decaying" when P < threshold, which means:
          hours_since_rehearsal > -ln(threshold / strength) / decay_rate
        """
        with self._lock:
            with self._tx() as cur:
                # Use strftime to compute Unix timestamps in seconds
                cur.execute(
                    """SELECT m.* FROM memories m
                       WHERE m.strength > 0
                         AND m.consolidation_stage != 'archived'
                         AND (CAST(strftime('%s', 'now') AS REAL) - m.last_rehearsed_at) / 3600.0
                             > -ln(? / m.strength) / m.decay_rate
                       LIMIT ?""",
                    (threshold, limit),
                )
                return [self._row_to_item(cur, row) for row in cur.fetchall()]

    # ── Stats ────────────────────────────────────────────────────

    async def stats(self) -> Dict[str, Any]:
        with self._lock:
            with self._tx() as cur:
                cur.execute("SELECT COUNT(*) as total FROM memories")
                total = cur.fetchone()["total"]

                cur.execute(
                    "SELECT memory_type, COUNT(*) as cnt FROM memories GROUP BY memory_type"
                )
                counts = {row["memory_type"]: row["cnt"] for row in cur.fetchall()}

                cur.execute("SELECT AVG(importance) as avg FROM memories")
                avg_imp = cur.fetchone()["avg"] or 0.0

                cur.execute("SELECT COUNT(*) as cnt FROM associations")
                assoc_count = cur.fetchone()["cnt"]

                return {
                    "total": total,
                    **counts,
                    "avg_importance": avg_imp,
                    "associations": assoc_count,
                }

    async def clear(self) -> None:
        with self._lock:
            with self._tx() as cur:
                cur.execute("DELETE FROM memories")
                cur.execute("DELETE FROM tags")
                cur.execute("DELETE FROM associations")
                cur.execute("DELETE FROM abstraction_sources")

    # ── Internal helpers ─────────────────────────────────────────

    def _insert_item(self, cur: sqlite3.Cursor, item: MemoryItem) -> None:
        cur.execute(
            """INSERT INTO memories
               (id, content, memory_type, memory_subtype,
                created_at, last_accessed_at, access_count,
                strength, decay_rate, last_rehearsed_at,
                valence, arousal, importance,
                attention_weight, capacity_slot,
                consolidation_stage, consolidated_at, promoted_from,
                metadata_json, source)
               VALUES (?,?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?, ?,?,?, ?,?)""",
            (
                item.id, item.content, item.memory_type.value, item.memory_subtype,
                _ts(item.created_at), _ts(item.last_accessed_at), item.access_count,
                item.strength, item.decay_rate, _ts(item.last_rehearsed_at),
                item.valence, item.arousal, item.importance,
                item.attention_weight, item.capacity_slot,
                item.consolidation_stage.value,
                _ts(item.consolidated_at) if item.consolidated_at else None,
                item.promoted_from,
                json.dumps(item.metadata), item.source,
            ),
        )

    def _insert_tags(self, cur: sqlite3.Cursor, memory_id: str, tags: List[str]) -> None:
        for tag in tags:
            cur.execute(
                "INSERT OR IGNORE INTO tags (memory_id, tag) VALUES (?, ?)",
                (memory_id, tag),
            )

    def _insert_associations(
        self, cur: sqlite3.Cursor, memory_id: str, associations: Dict[str, float]
    ) -> None:
        now = _ts(datetime.now(timezone.utc))
        for target_id, strength in associations.items():
            cur.execute(
                "INSERT OR IGNORE INTO associations (source_id, target_id, strength, created_at) VALUES (?, ?, ?, ?)",
                (memory_id, target_id, strength, now),
            )

    def _row_to_item(self, cur: sqlite3.Cursor, row: sqlite3.Row) -> MemoryItem:
        """Convert a SQLite row to a MemoryItem, loading tags and associations."""
        tags = [
            r["tag"] for r in cur.execute(
                "SELECT tag FROM tags WHERE memory_id = ?", (row["id"],)
            )
        ]
        associations = {
            r["target_id"]: r["strength"] for r in cur.execute(
                "SELECT target_id, strength FROM associations WHERE source_id = ?",
                (row["id"],),
            )
        }

        return MemoryItem(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            memory_subtype=row["memory_subtype"],
            created_at=_from_ts(row["created_at"]),
            last_accessed_at=_from_ts(row["last_accessed_at"]),
            access_count=row["access_count"],
            strength=row["strength"],
            decay_rate=row["decay_rate"],
            last_rehearsed_at=_from_ts(row["last_rehearsed_at"]),
            valence=row["valence"],
            arousal=row["arousal"],
            importance=row["importance"],
            attention_weight=row["attention_weight"],
            consolidation_stage=ConsolidationStage(row["consolidation_stage"]),
            consolidated_at=_from_ts(row["consolidated_at"]) if row["consolidated_at"] else None,
            promoted_from=row["promoted_from"],
            tags=tags,
            associations=associations,
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
            source=row["source"],
        )


# ── Schema ──────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    memory_type     TEXT NOT NULL,
    memory_subtype  TEXT,
    created_at      REAL NOT NULL,
    last_accessed_at REAL NOT NULL,
    access_count    INTEGER DEFAULT 0,
    strength        REAL DEFAULT 1.0,
    decay_rate      REAL DEFAULT 0.1,
    last_rehearsed_at REAL NOT NULL,
    valence         REAL DEFAULT 0.0,
    arousal         REAL DEFAULT 0.0,
    importance      REAL DEFAULT 0.5,
    attention_weight REAL DEFAULT 0.0,
    capacity_slot   INTEGER,
    consolidation_stage TEXT DEFAULT 'raw',
    consolidated_at REAL,
    promoted_from   TEXT,
    metadata_json   TEXT DEFAULT '{}',
    source          TEXT
);

CREATE TABLE IF NOT EXISTS tags (
    memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    tag         TEXT NOT NULL,
    PRIMARY KEY (memory_id, tag)
);

CREATE TABLE IF NOT EXISTS associations (
    source_id    TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id    TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    strength     REAL DEFAULT 1.0,
    created_at   REAL NOT NULL,
    PRIMARY KEY (source_id, target_id)
);

CREATE TABLE IF NOT EXISTS abstraction_sources (
    abstract_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    source_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    PRIMARY KEY (abstract_id, source_id)
);

-- Full-text search is handled via SQL LIKE (faster, no sync needed)

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_stage ON memories(consolidation_stage);
CREATE INDEX IF NOT EXISTS idx_tags_memory ON tags(memory_id);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
CREATE INDEX IF NOT EXISTS idx_associations_source ON associations(source_id);
CREATE INDEX IF NOT EXISTS idx_associations_target ON associations(target_id);
"""


def _ts(dt: Optional[datetime]) -> float:
    """Convert datetime to Unix timestamp."""
    if dt is None:
        return 0.0
    return dt.timestamp()


def _from_ts(ts: float) -> datetime:
    """Convert Unix timestamp to UTC datetime."""
    return datetime.fromtimestamp(ts, tz=timezone.utc)
