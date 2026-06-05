"""Abstract base class for all memore storage backends."""

from __future__ import annotations

import builtins
from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Any

from memore.memory.enums import MemoryType
from memore.memory.item import MemoryItem


class StorageBackend(ABC):
    """Abstract interface for memory storage backends.

    All storage backends (SQLite, ChromaDB, Qdrant, pgvector, InMemory)
    must implement this interface. This enables pluggable storage
    without changing the AgentMemory API.
    """

    # ── Lifecycle ────────────────────────────────────────────────

    @abstractmethod
    def initialize(self) -> None:
        """Open connections, create tables/collections, run migrations."""

    @abstractmethod
    def close(self) -> None:
        """Close connections and release resources."""

    # ── CRUD ─────────────────────────────────────────────────────

    @abstractmethod
    def store(self, item: MemoryItem) -> str:
        """Insert a new memory item. Returns its ID.

        Raises ``DuplicateMemoryError`` if the ID already exists.
        """

    @abstractmethod
    def batch_store(self, items: builtins.list[MemoryItem]) -> None:
        """Optimized bulk insert of multiple items."""

    @abstractmethod
    def get(self, memory_id: str) -> MemoryItem | None:
        """Retrieve a single memory item by ID.

        Returns ``None`` if not found.
        """

    @abstractmethod
    def update(self, item: MemoryItem) -> None:
        """Full-overwrite update of an existing memory item.

        Raises ``MemoryNotFoundError`` if the item does not exist.
        """

    @abstractmethod
    def delete(self, memory_id: str) -> None:
        """Remove a memory item by ID. Idempotent (no-op if missing)."""

    # ── Query ────────────────────────────────────────────────────

    @abstractmethod
    def search(
        self,
        query: str,
        query_embedding: builtins.list[float] | None = None,
        memory_types: Collection[MemoryType] | None = None,
        limit: int = 20,
        threshold: float = 0.0,
        include_archived: bool = False,
        **metadata_filters: Any,
    ) -> builtins.list[MemoryItem]:
        """Multi-modal search combining semantic and keyword matching.

        Each backend interprets ``query_embedding`` differently:
        vector DBs use it directly for ANN; SQLite falls back to FTS5.
        """

    @abstractmethod
    def list(
        self,
        memory_types: Collection[MemoryType] | None = None,
        importance_min: float = 0.0,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        include_archived: bool = False,
        **metadata_filters: Any,
    ) -> builtins.list[MemoryItem]:
        """Structured listing with filters, pagination, and sorting."""

    # ── Associations (Graph) ─────────────────────────────────────

    @abstractmethod
    def add_association(
        self,
        source_id: str,
        target_id: str,
        strength: float = 1.0,
    ) -> None:
        """Create or strengthen a directed association between two memories."""

    @abstractmethod
    def remove_association(self, source_id: str, target_id: str) -> None:
        """Remove a specific directed association. Idempotent."""

    @abstractmethod
    def get_associated(
        self,
        memory_id: str,
        max_depth: int = 2,
        min_strength: float = 0.0,
    ) -> dict[str, float]:
        """BFS traversal of the association graph.

        Returns ``{memory_id: activation_strength}`` for all reachable
        memories within ``max_depth`` hops and above ``min_strength``.
        """

    # ── Forgetting ───────────────────────────────────────────────

    @abstractmethod
    def get_decaying(
        self,
        threshold: float = 0.3,
        limit: int = 100,
    ) -> builtins.list[MemoryItem]:
        """Return memories whose retrieval probability is below threshold."""

    # ─── Stats & admin ───────────────────────────────────────────

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return system statistics (counts per type, averages, etc.)."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all data. Use with extreme caution."""
