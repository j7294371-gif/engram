"""No-op embedding provider — falls back to keyword search."""

from __future__ import annotations

from memore.embedding.base import EmbeddingProvider


class NoopEmbeddingProvider(EmbeddingProvider):
    """Disables vector embeddings. Search uses keyword matching.

    This is the default during MVP. Enables the system to work
    without any external embedding model or API key.
    """

    @property
    def dimension(self) -> int:
        return 0

    def embed(self, text: str) -> list[float]:
        return []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]
