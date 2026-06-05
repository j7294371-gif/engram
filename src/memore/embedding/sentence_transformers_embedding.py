"""Sentence-Transformers embedding provider — local, free, no API key needed.

Uses the sentence-transformers library to generate embeddings locally
using models like all-MiniLM-L6-v2 (384-dim, fast, good quality).
"""

from __future__ import annotations

from typing import List, Optional

from memore.embedding.base import EmbeddingProvider


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Local embedding provider using sentence-transformers models.

    Default model is ``all-MiniLM-L6-v2`` (384 dimensions), which
    provides a good balance of speed and quality for memory search.

    Args:
        model_name: Name of a sentence-transformers model.
        device: Computation device ('cpu', 'cuda', 'mps').
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None

    @property
    def dimension(self) -> int:
        self._lazy_load()
        return self._model.get_sentence_embedding_dimension()

    async def embed(self, text: str) -> List[float]:
        self._lazy_load()
        return self._model.encode(text, normalize_embeddings=True).tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._lazy_load()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

    def sync_embed(self, text: str) -> List[float]:
        """Synchronous version for use in synchronous code paths."""
        self._lazy_load()
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def _lazy_load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name, device=self._device)
