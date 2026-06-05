"""Sensory buffer — ultra-short-term, high-frequency ring buffer.

Mimics the biological sensory store: holds raw perceptual input for
a very short duration (seconds to tens of seconds), then auto-decays.
Items that are attended to may be promoted to working memory.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable

from memore.memory.item import MemoryItem


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
        on_expire: Callable[[MemoryItem], None] | None = None,
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
        with self._lock:
            self._evict_expired()
            if len(self._buffer) >= self.capacity:
                self._buffer.popitem(last=False)
            self._buffer[item.id] = (item, time.monotonic())

    def get(self, memory_id: str) -> MemoryItem | None:
        """Retrieve an item by ID if it hasn't expired."""
        with self._lock:
            self._evict_expired()
            entry = self._buffer.get(memory_id)
            if entry is not None:
                return entry[0]
            return None

    def search(self, query: str, limit: int = 10) -> list[MemoryItem]:
        """Basic keyword search over current buffer contents."""
        results: list[MemoryItem] = []
        query_lower = query.lower()
        with self._lock:
            self._evict_expired()
            for item, _ in self._buffer.values():
                if query_lower in item.content.lower():
                    results.append(item)
                    if len(results) >= limit:
                        break
        return results

    def all(self) -> list[MemoryItem]:
        """Return all non-expired items."""
        with self._lock:
            self._evict_expired()
            return [item for item, _ in self._buffer.values()]

    def size(self) -> int:
        """Current number of items in the buffer."""
        with self._lock:
            self._evict_expired()
            return len(self._buffer)

    def clear(self) -> None:
        """Remove all items immediately."""
        with self._lock:
            self._buffer.clear()

    # ── Internal ─────────────────────────────────────────────────

    def _evict_expired(self) -> None:
        """Remove items that have exceeded the TTL."""
        now = time.monotonic()
        expired: list[MemoryItem] = []
        # Collect expired items under the lock
        for mid, (item, ts) in list(self._buffer.items()):
            if now - ts > self.ttl_seconds:
                expired.append(item)
                self._buffer.pop(mid)
        # Fire callbacks outside the lock
        for item in expired:
            if self._on_expire:
                self._on_expire(item)
