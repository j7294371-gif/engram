"""Abstract base class for all engram storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Collection, Dict, List, Optional

from engram.memory.item import MemoryItem
from engram.memory.enums import MemoryType


class StorageBackend(ABC):
    """Abstract interface for memory storage backends.

    All storage backends (SQLite, ChromaDB, Qdrant, pgvector, InMemory)
    must implement this interface. This enables pluggable storage
    without changing the AgentMemory API.
    """

    # ── Lifecycle ────────────────────────────────────────────────

    @abstractmethod
    async def initialize(self) -> None:
        """Open connections, create tables/collections, run migrations."""

    @abstractmethod
    async def close(self) -> None:
        """Close connections and release resources."""

    # ── CRUD ─────────────────────────────────────────────────────

    @abstractmethod
    async def store(self, item: MemoryItem) -> str:
        """Insert a new memory item. Returns its ID.

        Raises ``DuplicateMemoryError`` if the ID already exists.
        """

    @abstractmethod
    async def batch_store(self, items: List[MemoryItem]) -> None:
        """Optimized bulk insert of multiple items."""

    @abstractmethod
    async def get(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a single memory item by ID.

        Returns ``None`` if not found.
        """

    @abstractmethod
    async def update(self, item: MemoryItem) -> None:
        """Full-overwrite update of an existing memory item.

        Raises ``MemoryNotFoundError`` if the item does not exist.
        """

    @abstractmethod
    async def delete(self, memory_id: str) -> None:
        """Remove a memory item by ID. Idempotent (no-op if missing)."""

    # ── Query ────────────────────────────────────────────────────

    @abstractmethod
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
        """Multi-modal search combining semantic and keyword matching.

        Each backend interprets ``query_embedding`` differently:
        vector DBs use it directly for ANN; SQLite falls back to FTS5.
        """

    @abstractmethod
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
        """Structured listing with filters, pagination, and sorting."""

    # ── Associations (Graph) ─────────────────────────────────────

    @abstractmethod
    async def add_association(
        self,
        source_id: str,
        target_id: str,
        strength: float = 1.0,
    ) -> None:
        """Create or strengthen a directed association between two memories."""

    @abstractmethod
    async def remove_association(self, source_id: str, target_id: str) -> None:
        """Remove a specific directed association. Idempotent."""

    @abstractmethod
    async def get_associated(
        self,
        memory_id: str,
        max_depth: int = 2,
        min_strength: float = 0.0,
    ) -> Dict[str, float]:
        """BFS traversal of the association graph.

        Returns ``{memory_id: activation_strength}`` for all reachable
        memories within ``max_depth`` hops and above ``min_strength``.
        """

    # ── Forgetting ───────────────────────────────────────────────

    @abstractmethod
    async def get_decaying(
        self,
        threshold: float = 0.3,
        limit: int = 100,
    ) -> List[MemoryItem]:
        """Return memories whose retrieval probability is below threshold."""

    # ─── Stats & admin ───────────────────────────────────────────

    @abstractmethod
    async def stats(self) -> Dict[str, Any]:
        """Return system statistics (counts per type, averages, etc.)."""

    @abstractmethod
    async def clear(self) -> None:
        """Remove all data. Use with extreme caution."""
