"""Working memory — bounded-capacity, attention-weighted context.

Models the biological working memory (Baddeley's model): a limited-
capacity workspace (~7±2 items) where information is actively
maintained and manipulated. Items have attention weights that
determine their prominence in the current context.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from memore.memory.item import MemoryItem


class WorkingMemory:
    """Bounded working memory with attention-weighted eviction.

    When capacity is exceeded, the item with the lowest attention
    weight is evicted. The ``on_evict`` callback is triggered,
    typically used to promote the item to long-term (episodic) storage.

    Attributes:
        capacity: Maximum number of items (Miller 7 ± 2).
    """

    def __init__(
        self,
        capacity: int = 7,
        on_evict: Callable[[MemoryItem], None] | None = None,
    ) -> None:
        self.capacity = max(1, capacity)
        self._on_evict = on_evict
        self._items: dict[str, MemoryItem] = {}
        self._lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────────

    def add(self, item: MemoryItem, attention_weight: float | None = None) -> None:
        """Add an item to working memory.

        If the item already exists, update it and boost attention.
        Otherwise, evict the lowest-attention item if at capacity.

        Args:
            item: The memory item to add.
            attention_weight: Attention weight [0, 1]; auto-computed
                if not provided (new items start at 0.5, boosted by 0.1).
        """
        with self._lock:
            if item.id in self._items:
                # Refresh existing — boost attention
                existing = self._items[item.id]
                existing.content = item.content
                existing.last_accessed_at = item.last_accessed_at
                existing.tags = list(set(existing.tags + item.tags))
                existing.attention_weight = min(
                    1.0, (attention_weight or existing.attention_weight) + 0.1
                )
                return

            weight = attention_weight if attention_weight is not None else 0.5
            item.attention_weight = weight

            if len(self._items) >= self.capacity:
                self._evict_lowest()

            self._items[item.id] = item

    def get(self, memory_id: str) -> MemoryItem | None:
        """Retrieve an item and boost its attention."""
        with self._lock:
            item = self._items.get(memory_id)
            if item is not None:
                item.attention_weight = min(1.0, item.attention_weight + 0.05)
                item.touch()
            return item

    def remove(self, memory_id: str) -> MemoryItem | None:
        """Explicitly remove an item from working memory."""
        with self._lock:
            return self._items.pop(memory_id, None)

    def get_context(self, window_size: int = 7) -> list[MemoryItem]:
        """Return items sorted by attention weight (highest first).

        Args:
            window_size: Maximum number of items to return.
        """
        with self._lock:
            sorted_items = sorted(
                self._items.values(),
                key=lambda i: i.attention_weight,
                reverse=True,
            )
            return sorted_items[:window_size]

    def focus(self, content: str) -> MemoryItem:
        """Add or refresh a high-attention item (foreground focus).

        Convenience method for the current task's focal point.
        """
        from memore.memory.enums import MemoryType

        item = MemoryItem(
            id=_make_id(),
            content=content,
            memory_type=MemoryType.WORKING,
            attention_weight=1.0,
        )
        self.add(item, attention_weight=item.attention_weight)
        return item

    def attend_to(self, memory_id: str, weight: float) -> None:
        """Manually set the attention weight of a specific item."""
        with self._lock:
            item = self._items.get(memory_id)
            if item is not None:
                item.attention_weight = max(0.0, min(1.0, weight))

    def size(self) -> int:
        """Current number of items in working memory."""
        with self._lock:
            return len(self._items)

    def is_full(self) -> bool:
        """Is working memory at or above capacity?"""
        return self.size() >= self.capacity

    def clear(self) -> None:
        """Remove all items."""
        with self._lock:
            self._items.clear()

    # ── Internal ─────────────────────────────────────────────────

    def _evict_lowest(self) -> MemoryItem | None:
        """Evict the item with the lowest attention weight."""
        if not self._items:
            return None
        lowest_id = min(self._items, key=lambda iid: self._items[iid].attention_weight)
        evicted = self._items.pop(lowest_id)
        if self._on_evict:
            self._on_evict(evicted)
        return evicted


def _make_id() -> str:
    """Generate a simple ULID-like ID for working memory items."""
    import secrets
    import time

    timestamp = int(time.time() * 1000)
    rand = secrets.token_hex(8)
    return f"wm_{timestamp:x}_{rand}"
