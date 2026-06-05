"""Abstract interface for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class EmbeddingProvider(ABC):
    """Generates vector embeddings for memory items.

    Implementations may use local models (sentence-transformers),
    cloud APIs (OpenAI), or a no-op fallback.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension."""

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate an embedding vector for the given text."""

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of texts."""
