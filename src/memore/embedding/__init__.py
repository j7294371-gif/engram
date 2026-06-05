"""Embedding providers for semantic search."""

from memore.embedding.base import EmbeddingProvider
from memore.embedding.noop import NoopEmbeddingProvider

__all__ = ["EmbeddingProvider", "NoopEmbeddingProvider"]
