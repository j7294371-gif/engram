"""Sensory buffer — ultra-short-term, high-frequency ring buffer.

Mimics the biological sensory store: holds raw perceptual input for
a very short duration (seconds to tens of seconds), then auto-decays.
Items that are attended to may be promoted to working memory.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Callable, List, Optional

from engram.memory.item import MemoryItem


class SensoryBuffer:
    """Ring buffer for ultra-short-term sensory memory.

    Automatically evicts items by TTL. Optionally notifies a
    callback when an item's TTL expires (allowing promotion logic).

    Attributes:
        capacity: Maximum number of items.
        ttl_seconds: Time-to-live for each item.
    """

    def __init__(
        self,
        capacity: int = 50,
        ttl_seconds: float = 30.0,
        on_expire: Optional[Callable[[MemoryItem], None]] = None,
    ) -> None:
        self.capacity = max(1, capacity)
        self.ttl_seconds = max(0.1, ttl_seconds)
        self._on_expire = on_expire
        self._buffer: OrderedDict[str, tuple[MemoryItem, float]] = OrderedDict()
        self._lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────────

    def add(self, item: MemoryItem) -> None:
        """Insert an item into the sensory buffer.

        Evicts the oldest item if at capacity.
        """
        self._evict_expired()
        with self._lock:
            if len(self._buffer) >= self.capacity:
                self._buffer.popitem(last=False)
            self._buffer[item.id] = (item, time.monotonic())

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve an item by ID if it hasn't expired."""
        self._evict_expired()
        with self._lock:
            entry = self._buffer.get(memory_id)
            if entry is not None:
                return entry[0]
            return None

    def search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Basic keyword search over current buffer contents."""
        self._evict_expired()
        results: List[MemoryItem] = []
        query_lower = query.lower()
        with self._lock:
            for item, _ in self._buffer.values():
                if query_lower in item.content.lower():
                    results.append(item)
                    if len(results) >= limit:
                        break
        return results

    def all(self) -> List[MemoryItem]:
        """Return all non-expired items."""
        self._evict_expired()
        with self._lock:
            return [item for item, _ in self._buffer.values()]

    def size(self) -> int:
        """Current number of items in the buffer."""
        self._evict_expired()
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        """Remove all items immediately."""
        with self._lock:
            self._buffer.clear()

    # ── Internal ─────────────────────────────────────────────────

    def _evict_expired(self) -> None:
        """Remove items that have exceeded the TTL."""
        now = time.monotonic()
        expired_ids: List[str] = []
        with self._lock:
            for mid, (item, ts) in self._buffer.items():
                if now - ts > self.ttl_seconds:
                    expired_ids.append(mid)
            for mid in expired_ids:
                expired_item = self._buffer.pop(mid, None)
                if expired_item is not None and self._on_expire:
                    self._on_expire(expired_item[0])
