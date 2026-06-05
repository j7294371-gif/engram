"""In-memory storage backend — default for development and testing."""

from __future__ import annotations

import builtins
from collections import defaultdict
from collections.abc import Collection
from typing import Any

from memore.exceptions import DuplicateMemoryError, MemoryNotFoundError
from memore.memory.enums import ConsolidationStage, MemoryType
from memore.memory.item import MemoryItem
from memore.storage.base import StorageBackend


class InMemoryBackend(StorageBackend):
    """Stores all memory items in process memory using dicts.

    Perfect for development, testing, and prototyping. Not suitable
    for production use — data is lost on process restart.

    Uses three data structures internally:
    - ``_items``: ``{memory_id: MemoryItem}``
    - ``_associations``: ``{source_id: {target_id: strength}}``
    - ``_tags``: ``{tag: set[memory_id]}`` for fast tag filtering
    """

    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}
        self._associations: dict[str, dict[str, float]] = defaultdict(dict)
        self._tags: dict[str, set[str]] = defaultdict(set)
        self._initialized = False

    # ── Lifecycle ────────────────────────────────────────────────

    async def initialize(self) -> None:
        self._items.clear()
        self._associations.clear()
        self._tags.clear()
        self._initialized = True

    async def close(self) -> None:
        self._items.clear()
        self._associations.clear()
        self._tags.clear()
        self._initialized = False

    # ── CRUD ─────────────────────────────────────────────────────

    async def store(self, item: MemoryItem) -> str:
        if item.id in self._items:
            raise DuplicateMemoryError(f"Memory {item.id!r} already exists.")
        self._items[item.id] = item
        for tag in item.tags:
            self._tags[tag].add(item.id)
        return item.id

    async def batch_store(self, items: builtins.list[MemoryItem]) -> None:
        for item in items:
            await self.store(item)

    async def get(self, memory_id: str) -> MemoryItem | None:
        return self._items.get(memory_id)

    async def update(self, item: MemoryItem) -> None:
        if item.id not in self._items:
            raise MemoryNotFoundError(f"Memory {item.id!r} not found.")
        # Remove old tags, add new ones
        old = self._items[item.id]
        for tag in old.tags:
            self._tags[tag].discard(item.id)
        self._items[item.id] = item
        for tag in item.tags:
            self._tags[tag].add(item.id)

    async def delete(self, memory_id: str) -> None:
        item = self._items.pop(memory_id, None)
        if item is not None:
            for tag in item.tags:
                self._tags[tag].discard(memory_id)
            self._associations.pop(memory_id, None)
            # Remove associations pointing to this item
            for source in self._associations.values():
                source.pop(memory_id, None)

    # ── Query ────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        query_embedding: builtins.list[float] | None = None,
        memory_types: Collection[MemoryType] | None = None,
        limit: int = 20,
        threshold: float = 0.0,
        include_archived: bool = False,
        **metadata_filters: Any,
    ) -> builtins.list[MemoryItem]:
        """Search by keyword matching against content and tags.

        In the InMemory backend without embeddings, search performs
        case-insensitive substring matching on content, tags, and
        source fields as a basic keyword search.
        """
        query_lower = query.lower() if query else ""
        results: list[MemoryItem] = []

        for item in self._items.values():
            # Apply filters
            if memory_types and item.memory_type not in memory_types:
                continue
            if not include_archived and item.consolidation_stage == ConsolidationStage.ARCHIVED:
                continue
            if metadata_filters and not all(item.metadata.get(k) == v for k, v in metadata_filters.items()):
                continue

            # Keyword match
            if query_lower:
                if (
                    query_lower in item.content.lower()
                    or any(query_lower in t.lower() for t in item.tags)
                    or (item.source and query_lower in item.source.lower())
                ):
                    _keyword_score(item, query_lower)
                    results.append(item)
            else:
                results.append(item)

        # Sort by retrieval probability (importance for non-embedding backend)
        results.sort(key=lambda i: i.retrieval_probability(), reverse=True)
        return results[:limit]

    async def list(
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
        results: list[MemoryItem] = []
        for item in self._items.values():
            if memory_types and item.memory_type not in memory_types:
                continue
            if item.importance < importance_min:
                continue
            if not include_archived and item.consolidation_stage == ConsolidationStage.ARCHIVED:
                continue
            if metadata_filters and not all(item.metadata.get(k) == v for k, v in metadata_filters.items()):
                continue
            results.append(item)

        # Sort
        sort_key = {
            "created_at": lambda i: i.created_at,
            "last_accessed_at": lambda i: i.last_accessed_at,
            "importance": lambda i: i.importance,
            "strength": lambda i: i.strength,
            "access_count": lambda i: i.access_count,
        }.get(sort_by, lambda i: i.created_at)
        results.sort(key=sort_key, reverse=sort_desc)
        return results[offset : offset + limit]

    # ── Associations ─────────────────────────────────────────────

    async def add_association(self, source_id: str, target_id: str, strength: float = 1.0) -> None:
        if source_id not in self._items:
            raise MemoryNotFoundError(f"Source memory {source_id!r} not found.")
        if target_id not in self._items:
            raise MemoryNotFoundError(f"Target memory {target_id!r} not found.")
        self._associations[source_id][target_id] = strength

    async def remove_association(self, source_id: str, target_id: str) -> None:
        self._associations.get(source_id, {}).pop(target_id, None)

    async def get_associated(
        self,
        memory_id: str,
        max_depth: int = 2,
        min_strength: float = 0.0,
    ) -> dict[str, float]:
        """BFS spreading activation from the seed memory."""
        visited: dict[str, float] = {}
        queue: list[tuple[str, float, int]] = [(memory_id, 1.0, 0)]

        while queue:
            current_id, activation, depth = queue.pop(0)
            if depth > max_depth:
                continue
            if current_id in visited and visited[current_id] >= activation:
                continue
            visited[current_id] = activation

            if depth < max_depth:
                for neighbor, strength in self._associations.get(current_id, {}).items():
                    if strength < min_strength:
                        continue
                    next_activation = activation * strength * 0.85  # decay factor
                    queue.append((neighbor, next_activation, depth + 1))

        # Remove the seed itself from results
        visited.pop(memory_id, None)
        return visited

    # ── Forgetting ───────────────────────────────────────────────

    async def get_decaying(self, threshold: float = 0.3, limit: int = 100) -> builtins.list[MemoryItem]:
        decaying = []
        for item in self._items.values():
            if item.retrieval_probability() < threshold:
                decaying.append(item)
        decaying.sort(key=lambda i: i.retrieval_probability())
        return decaying[:limit]

    # ── Stats ────────────────────────────────────────────────────

    async def stats(self) -> dict[str, Any]:
        if not self._items:
            count_by_type = {t.value: 0 for t in MemoryType}
            return {"total": 0, **count_by_type, "avg_importance": 0.0}

        count_by_type: dict[str, int] = {t.value: 0 for t in MemoryType}
        total_importance = 0.0
        for item in self._items.values():
            count_by_type[item.memory_type.value] = count_by_type.get(item.memory_type.value, 0) + 1
            total_importance += item.importance

        return {
            "total": len(self._items),
            **count_by_type,
            "avg_importance": total_importance / len(self._items),
            "associations": sum(len(v) for v in self._associations.values()),
            "tags": len(self._tags),
        }

    async def clear(self) -> None:
        await self.initialize()


def _keyword_score(item: MemoryItem, query: str) -> float:
    """Simple keyword relevance score (term frequency heuristic)."""
    score = 0.0
    text = (item.content + " " + " ".join(item.tags)).lower()
    terms = query.strip().split()
    for term in terms:
        count = text.count(term)
        score += count / (len(text.split()) + 1)
    return score
