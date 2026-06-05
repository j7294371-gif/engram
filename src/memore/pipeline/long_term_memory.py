"""Long-term memory store — episodic, semantic, and procedural.

Manages persistent storage, retrieval, and type-specific decay
for long-term memories. Delegates actual persistence to a
StorageBackend and adds pipeline-specific logic on top.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set

from memore.memory.enums import ConsolidationStage, MemoryType, RetrievalMode
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
    MANAGED_TYPES: Set[MemoryType] = {
        MemoryType.EPISODIC,
        MemoryType.SEMANTIC,
        MemoryType.PROCEDURAL,
    }

    def __init__(
        self,
        backend: StorageBackend,
        on_promote: Optional[Callable[[MemoryItem], None]] = None,
    ) -> None:
        self._backend = backend
        self._on_promote = on_promote

    # ── Store ────────────────────────────────────────────────────

    async def store(self, item: MemoryItem) -> str:
        """Store a long-term memory item.

        If the memory type is not managed by long-term memory
        (e.g., sensory or working), it is silently rejected.
        """
        if item.memory_type not in self.MANAGED_TYPES:
            raise ValueError(
                f"LongTermMemory only manages {[t.value for t in self.MANAGED_TYPES]}, "
                f"got {item.memory_type.value!r}"
            )
        return await self._backend.store(item)

    async def batch_store(self, items: List[MemoryItem]) -> None:
        """Bulk store multiple long-term items."""
        valid = [i for i in items if i.memory_type in self.MANAGED_TYPES]
        if valid:
            await self._backend.batch_store(valid)

    # ── Retrieve ─────────────────────────────────────────────────

    async def get(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a single memory by ID."""
        return await self._backend.get(memory_id)

    async def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 20,
        threshold: float = 0.0,
        mode: str = "hybrid",
        **kwargs,
    ) -> List[MemoryItem]:
        """Search across long-term memory stores."""
        # Normalise memory_types
        if memory_types is None:
            memory_types = [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]
        else:
            memory_types = [t for t in memory_types if t in self.MANAGED_TYPES]

        return await self._backend.search(
            query=query,
            query_embedding=query_embedding,
            memory_types=memory_types,
            limit=limit,
            threshold=threshold,
            **kwargs,
        )

    # ── Type-specific helpers ────────────────────────────────────

    async def store_episodic(self, item: MemoryItem) -> str:
        """Store an episodic (event/experience) memory."""
        item.memory_type = MemoryType.EPISODIC
        return await self.store(item)

    async def store_semantic(self, item: MemoryItem) -> str:
        """Store a semantic (fact/knowledge) memory."""
        item.memory_type = MemoryType.SEMANTIC
        return await self.store(item)

    async def store_procedural(self, item: MemoryItem) -> str:
        """Store a procedural (skill/pattern) memory."""
        item.memory_type = MemoryType.PROCEDURAL
        return await self.store(item)

    # ── Consolidation support ────────────────────────────────────

    async def promote_from_working(self, item: MemoryItem) -> str:
        """Promote a working memory item to episodic storage.

        Called during consolidation when a working memory item
        exceeds the importance threshold.
        """
        item.memory_type = MemoryType.EPISODIC
        item.consolidation_stage = ConsolidationStage.WORKING_PROMOTED
        item.promoted_from = item.id
        item.importance = item.recompute_importance()

        # Assign a new ID for the episodic copy
        import secrets, time

        item.id = f"ep_{int(time.time() * 1000):x}_{secrets.token_hex(6)}"

        mid = await self._backend.store(item)
        if self._on_promote:
            self._on_promote(item)
        return mid

    async def update(self, item: MemoryItem) -> None:
        """Update an existing long-term memory item."""
        await self._backend.update(item)

    async def stats(self) -> Dict:
        """Return storage statistics."""
        return await self._backend.stats()
