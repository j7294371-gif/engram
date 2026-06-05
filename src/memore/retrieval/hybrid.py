"""Hybrid ranker — fuses semantic similarity, spreading activation,
and importance scoring into a single relevance score.

The fusion formula:

    score = w_s * semantic + w_a * associative + w_i * importance

Where weights are configurable per query.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from memore.memory.item import MemoryItem

# Type alias for embedding similarity function
SimilarityFn = Callable[[MemoryItem, str | None, list[float] | None], float]


class HybridRanker:
    """Fuses multiple relevance signals into a composite score.

    Combines:
    - Semantic similarity (embedding cosine)
    - Associative activation (spreading activation from seed items)
    - Importance score (recency × frequency × emotional intensity)
    - Emotional congruence (mood bias)

    Args:
        semantic_weight: Weight for embedding similarity.
        associative_weight: Weight for spreading activation.
        importance_weight: Weight for importance score.
        emotional_weight: Weight for mood-congruence bias.
    """

    def __init__(
        self,
        semantic_weight: float = 0.4,
        associative_weight: float = 0.3,
        importance_weight: float = 0.2,
        emotional_weight: float = 0.1,
    ) -> None:
        self.semantic_weight = semantic_weight
        self.associative_weight = associative_weight
        self.importance_weight = importance_weight
        self.emotional_weight = emotional_weight

    def rank(
        self,
        items: list[MemoryItem],
        query: str | None = None,
        query_embedding: list[float] | None = None,
        activation_scores: dict[str, float] | None = None,
        mood_congruent: tuple[float, float] | None = None,
    ) -> list[tuple[MemoryItem, float]]:
        """Rank items by composite relevance score.

        Args:
            items: Memory items to rank.
            query: Original search query (used for retrieval probability).
            query_embedding: Query embedding for semantic similarity.
            activation_scores: Pre-computed spreading activation scores
                ``{memory_id: activation}``.
            mood_congruent: Optional (valence, arousal) for mood bias.

        Returns:
            Items sorted by descending relevance, each with its score.
        """
        scored: list[tuple[MemoryItem, float]] = []
        for item in items:
            score = self._compute_score(
                item, query, query_embedding, activation_scores, mood_congruent
            )
            scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _compute_score(
        self,
        item: MemoryItem,
        query: str | None = None,
        query_embedding: list[float] | None = None,
        activation_scores: dict[str, float] | None = None,
        mood_congruent: tuple[float, float] | None = None,
    ) -> float:
        """Compute composite relevance score for a single item."""
        score = 0.0

        # 1. Semantic similarity
        if query_embedding and item.embedding:
            sim = _cosine_similarity(query_embedding, item.embedding)
            score += self.semantic_weight * sim
        elif query:
            # Fallback: use simple keyword overlap score
            query_words = set(query.lower().split()) if query else set()
            content_words = set(item.content.lower().split())
            overlap = len(query_words & content_words) / max(len(query_words), 1) if query_words else 0
            score += self.semantic_weight * overlap

        # 2. Associative activation
        if activation_scores and item.id in activation_scores:
            score += self.associative_weight * activation_scores[item.id]

        # 3. Importance
        score += self.importance_weight * item.importance

        # 4. Emotional congruence
        if mood_congruent:
            valence, arousal = mood_congruent
            congruence = item.emotional_congruence(valence, arousal)
            score += self.emotional_weight * congruence

        return score


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
