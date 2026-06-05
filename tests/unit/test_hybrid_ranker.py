"""Tests for the HybridRanker — fusing semantic, associative, and importance signals."""

from __future__ import annotations

from memore.memory.enums import MemoryType
from memore.memory.item import MemoryItem
from memore.retrieval.hybrid import HybridRanker, _cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = [1.0, 0.0, 0.0]
        assert _cosine_similarity(a, a) == 1.0

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == -1.0

    def test_empty_returns_zero(self):
        assert _cosine_similarity([], [1.0]) == 0.0
        assert _cosine_similarity([1.0], []) == 0.0

    def test_different_lengths_returns_zero(self):
        assert _cosine_similarity([1.0, 0.0], [1.0]) == 0.0


class TestHybridRanker:
    def test_empty_items_returns_empty(self):
        ranker = HybridRanker()
        assert ranker.rank([]) == []

    def test_default_ranking_uses_importance(self):
        ranker = HybridRanker(importance_weight=1.0, semantic_weight=0.0, associative_weight=0.0)
        items = [
            MemoryItem(id="a", content="low", memory_type=MemoryType.EPISODIC, importance=0.3),
            MemoryItem(id="b", content="high", memory_type=MemoryType.EPISODIC, importance=0.9),
        ]
        ranked = ranker.rank(items)
        assert ranked[0][0].id == "b"
        assert ranked[1][0].id == "a"

    def test_associative_activation_boosts_score(self):
        ranker = HybridRanker(associative_weight=1.0, semantic_weight=0.0, importance_weight=0.0)
        items = [
            MemoryItem(id="a", content="item", memory_type=MemoryType.EPISODIC),
            MemoryItem(id="b", content="item", memory_type=MemoryType.EPISODIC),
        ]
        activation = {"a": 0.8, "b": 0.2}
        ranked = ranker.rank(items, activation_scores=activation)
        assert ranked[0][0].id == "a"
        assert ranked[1][0].id == "b"

    def test_semantic_similarity_uses_embedding(self):
        ranker = HybridRanker(semantic_weight=1.0, associative_weight=0.0, importance_weight=0.0)
        items = [
            MemoryItem(id="a", content="same", memory_type=MemoryType.EPISODIC, embedding=[1.0, 0.0]),
            MemoryItem(id="b", content="diff", memory_type=MemoryType.EPISODIC, embedding=[0.0, 1.0]),
        ]
        ranked = ranker.rank(items, query="test", query_embedding=[1.0, 0.0])
        assert ranked[0][0].id == "a"

    def test_mood_congruent_boosts_matching_memories(self):
        ranker = HybridRanker(emotional_weight=1.0, semantic_weight=0.0,
                               associative_weight=0.0, importance_weight=0.0)
        items = [
            MemoryItem(id="a", content="happy", memory_type=MemoryType.EPISODIC, valence=0.9, arousal=0.8),
            MemoryItem(id="b", content="sad", memory_type=MemoryType.EPISODIC, valence=-0.8, arousal=0.5),
        ]
        ranked = ranker.rank(items, mood_congruent=(0.9, 0.8))
        assert ranked[0][0].id == "a"

    def test_all_signals_combined(self):
        ranker = HybridRanker(semantic_weight=0.4, associative_weight=0.3,
                               importance_weight=0.2, emotional_weight=0.1)
        items = [
            MemoryItem(id="a", content="match", memory_type=MemoryType.EPISODIC,
                       embedding=[1.0, 0.0], importance=0.5, valence=0.5, arousal=0.5),
            MemoryItem(id="b", content="other", memory_type=MemoryType.EPISODIC,
                       embedding=[0.0, 1.0], importance=0.8, valence=-0.5, arousal=0.3),
        ]
        ranked = ranker.rank(
            items,
            query="test",
            query_embedding=[1.0, 0.0],
            activation_scores={"a": 0.6, "b": 0.3},
            mood_congruent=(0.5, 0.5),
        )
        # 'a' should rank higher due to matching embedding + high activation + mood congruence
        assert ranked[0][0].id == "a"
        # Scores should be within valid range
        for _, score in ranked:
            assert 0.0 <= score <= 1.0
