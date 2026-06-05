"""Long-term memory store — episodic, semantic, and procedural.

Manages persistent storage, retrieval, and type-specific decay
for long-term memories. Delegates actual persistence to a
StorageBackend and adds pipeline-specific logic on top.
"""

from __future__ import annotations

import dataclasses
import secrets
import time
from collections.abc import Callable

from memore.memory.enums import ConsolidationStage, MemoryType
from memore.memory.item import MemoryItem
from memore.storage.base import StorageBackend


class LongTermMemory:
    """Orchestrates storage and retrieval of episodic, semantic,
    and procedural memories.

    Acts as a coordinator layer over the StorageBackend, adding
    type-specific routing, consolidation promotion hooks, and
    hybrid retrieval orchestration.
    """

    # Memory types managed by this store
    MANAGED_TYPES: set[MemoryType] = {
        MemoryType.EPISODIC,
        MemoryType.SEMANTIC,
        MemoryType.PROCEDURAL,
    }

    def __init__(
        self,
        backend: StorageBackend,
        on_promote: Callable[[MemoryItem], None] | None = None,
    ) -> None:
        self._backend = backend
        self._on_promote = on_promote

    # ── Store ────────────────────────────────────────────────────

    def store(self, item: MemoryItem) -> str:
        """Store a long-term memory item.

        If the memory type is not managed by long-term memory
        (e.g., sensory or working), it is silently rejected.
        """
        if item.memory_type not in self.MANAGED_TYPES:
            raise ValueError(
                f"LongTermMemory only manages {[t.value for t in self.MANAGED_TYPES]}, "
                f"got {item.memory_type.value!r}"
            )
        return self._backend.store(item)

    def batch_store(self, items: list[MemoryItem]) -> None:
        """Bulk store multiple long-term items."""
        valid = [i for i in items if i.memory_type in self.MANAGED_TYPES]
        if valid:
            self._backend.batch_store(valid)

    # ── Retrieve ─────────────────────────────────────────────────

    def get(self, memory_id: str) -> MemoryItem | None:
        """Retrieve a single memory by ID."""
        return self._backend.get(memory_id)

    def search(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 20,
        threshold: float = 0.0,
        mode: str = "hybrid",
        **kwargs,
    ) -> list[MemoryItem]:
        """Search across long-term memory stores."""
        # Normalise memory_types
        if memory_types is None:
            memory_types = [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]
        else:
            memory_types = [t for t in memory_types if t in self.MANAGED_TYPES]

        return self._backend.search(
            query=query,
            query_embedding=query_embedding,
            memory_types=memory_types,
            limit=limit,
            threshold=threshold,
            **kwargs,
        )

    # ── Type-specific helpers ────────────────────────────────────

    def store_episodic(self, item: MemoryItem) -> str:
        """Store an episodic (event/experience) memory."""
        copy = dataclasses.replace(item, memory_type=MemoryType.EPISODIC)
        return self.store(copy)

    def store_semantic(self, item: MemoryItem) -> str:
        """Store a semantic (fact/knowledge) memory."""
        copy = dataclasses.replace(item, memory_type=MemoryType.SEMANTIC)
        return self.store(copy)

    def store_procedural(self, item: MemoryItem) -> str:
        """Store a procedural (skill/pattern) memory."""
        copy = dataclasses.replace(item, memory_type=MemoryType.PROCEDURAL)
        return self.store(copy)

    # ── Consolidation support ────────────────────────────────────

    def promote_from_working(self, item: MemoryItem) -> str:
        """Promote a working memory item to episodic storage.

        Called during consolidation when a working memory item
        exceeds the importance threshold.
        """
        promoted = MemoryItem(
            id=f"ep_{int(time.time() * 1000):x}_{secrets.token_hex(6)}",
            content=item.content,
            memory_type=MemoryType.EPISODIC,
            consolidation_stage=ConsolidationStage.WORKING_PROMOTED,
            promoted_from=item.id,
            importance=item.recompute_importance(),
        )

        mid = self._backend.store(promoted)
        if self._on_promote:
            self._on_promote(promoted)
        return mid

    def update(self, item: MemoryItem) -> None:
        """Update an existing long-term memory item."""
        self._backend.update(item)

    def stats(self) -> dict:
        """Return storage statistics."""
        return self._backend.stats()
