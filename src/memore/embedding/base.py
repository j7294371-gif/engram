"""Abstract interface for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


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
    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of texts."""

    def sync_embed(self, text: str) -> list[float]:
        """Synchronous embedding fallback.

        Default implementation delegates to embed().
        Providers with native sync support should override this.
        """
        return self.embed(text)  # embed() is now sync too!
