"""Embedding providers for semantic search."""

from engram.embedding.base import EmbeddingProvider
from engram.embedding.noop import NoopEmbeddingProvider

__all__ = ["EmbeddingProvider", "NoopEmbeddingProvider"]
