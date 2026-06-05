"""OpenAI embedding provider — cloud-based embeddings via the OpenAI API.

Requires the ``openai`` Python package and an API key set via
the ``OPENAI_API_KEY`` environment variable or passed directly.
"""

from __future__ import annotations

import os

from memore.embedding.base import EmbeddingProvider
from memore.exceptions import EmbeddingError


class OpenAIEmbedding(EmbeddingProvider):
    """Cloud embedding provider using OpenAI's embedding models.

    Default model is ``text-embedding-3-small`` (1536 dimensions),
    which offers a good balance of quality and cost.

    Args:
        model: OpenAI embedding model name.
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        dimensions: Optional dimension override (for text-embedding-3 models).
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._dimensions = dimensions
        self._client = None

    @property
    def dimension(self) -> int:
        return self._dimensions or 1536

    async def embed(self, text: str) -> list[float]:
        client = self._get_client()
        kwargs = {"model": self._model, "input": text}
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions
        try:
            response = await client.embeddings.create(**kwargs)
            return response.data[0].embedding
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding failed: {e}") from e

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        kwargs = {"model": self._model, "input": texts}
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions
        try:
            response = await client.embeddings.create(**kwargs)
            # Sort by index to preserve order
            sorted_data = sorted(response.data, key=lambda d: d.index)
            return [d.embedding for d in sorted_data]
        except Exception as e:
            raise EmbeddingError(f"OpenAI batch embedding failed: {e}") from e

    def sync_embed(self, text: str) -> list[float]:
        """Synchronous fallback — uses openai's sync client."""
        import openai
        client = openai.OpenAI(api_key=self._api_key)
        kwargs = {"model": self._model, "input": text}
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions
        try:
            response = client.embeddings.create(**kwargs)
            return response.data[0].embedding
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding failed: {e}") from e

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.AsyncOpenAI(api_key=self._api_key)
        return self._client
