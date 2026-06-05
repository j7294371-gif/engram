"""Tests for the MemoryItem data model."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from memore.memory.enums import ConsolidationStage, MemoryType
from memore.memory.item import MemoryItem


class TestMemoryItemCreation:
    def test_default_fields(self):
        """Item should have sensible defaults."""
        item = MemoryItem(id="t1", content="hello", memory_type=MemoryType.EPISODIC)
        assert item.strength == 1.0
        assert item.importance == 0.5
        assert item.valence == 0.0
        assert item.arousal == 0.0
        assert item.consolidation_stage == ConsolidationStage.RAW
        assert item.tags == []
        assert item.associations == {}

    def test_decay_rate_defaults(self):
        """Each memory type should have an appropriate default decay rate."""
        sensory = MemoryItem(id="s", content="s", memory_type=MemoryType.SENSORY)
        working = MemoryItem(id="w", content="w", memory_type=MemoryType.WORKING)
        episodic = MemoryItem(id="e", content="e", memory_type=MemoryType.EPISODIC)
        semantic = MemoryItem(id="sem", content="sem", memory_type=MemoryType.SEMANTIC)
        procedural = MemoryItem(id="p", content="p", memory_type=MemoryType.PROCEDURAL)

        assert sensory.decay_rate == 10.0  # fastest decay
        assert working.decay_rate == 2.0
        assert episodic.decay_rate == 0.15
        assert semantic.decay_rate == 0.05  # slowest decay
        assert procedural.decay_rate == 0.08

    def test_valence_arousal_clamping(self):
        """Valence and arousal should be clamped to valid ranges."""
        item = MemoryItem(
            id="t", content="test", memory_type=MemoryType.EPISODIC,
            valence=5.0, arousal=-1.0,
        )
        assert item.valence == 1.0
        assert item.arousal == 0.0

        item2 = MemoryItem(
            id="t2", content="test", memory_type=MemoryType.EPISODIC,
            valence=-2.0, arousal=2.0,
        )
        assert item2.valence == -1.0
        assert item2.arousal == 1.0


class TestForgettingCurve:
    def test_retrieval_probability_decreases_over_time(self):
        """Retrieval probability should monotonically decrease."""
        item = MemoryItem(
            id="t", content="test", memory_type=MemoryType.EPISODIC,
            strength=1.0, decay_rate=0.1,
        )
        p1 = item.retrieval_probability(at_time=item.created_at + timedelta(hours=1))
        p2 = item.retrieval_probability(at_time=item.created_at + timedelta(hours=2))
        assert p1 > p2

    def test_retrieval_probability_at_zero_time(self):
        """At the moment of creation, probability should be strength."""
        item = MemoryItem(
            id="t", content="test", memory_type=MemoryType.EPISODIC,
            strength=0.8,
        )
        p = item.retrieval_probability(at_time=item.last_rehearsed_at)
        assert abs(p - 0.8) < 1e-6

    def test_is_forgotten_below_threshold(self):
        """Should return True when below threshold."""
        item = MemoryItem(
            id="t", content="test", memory_type=MemoryType.EPISODIC,
            strength=0.5, decay_rate=10.0,
        )
        assert item.is_forgotten(threshold=0.5, at_time=item.created_at + timedelta(hours=1))

    def test_is_forgotten_above_threshold(self):
        """Should return False when above threshold."""
        item = MemoryItem(
            id="t", content="test", memory_type=MemoryType.EPISODIC,
            strength=1.0, decay_rate=0.01,
        )
        assert not item.is_forgotten(threshold=0.5, at_time=item.created_at + timedelta(hours=1))


class TestRehearsal:
    def test_rehearsal_boosts_strength(self):
        """Rehearsal should boost strength."""
        item = MemoryItem(id="t", content="test", memory_type=MemoryType.EPISODIC, strength=0.5)
        item.rehearse(strength_boost=0.2)
        assert item.strength == pytest.approx(0.7, rel=1e-6)

    def test_rehearsal_resets_decay_clock(self):
        """Rehearsal should update last_rehearsed_at."""
        item = MemoryItem(id="t", content="test", memory_type=MemoryType.EPISODIC)
        old_time = item.last_rehearsed_at
        import time
        time.sleep(0.01)
        item.rehearse()
        assert item.last_rehearsed_at > old_time

    def test_rehearsal_increments_access_count(self):
        """Rehearsal should increment access count."""
        item = MemoryItem(id="t", content="test", memory_type=MemoryType.EPISODIC)
        assert item.access_count == 0
        item.rehearse()
        assert item.access_count == 1
        item.rehearse()
        assert item.access_count == 2

    def test_rehearsal_strength_capped_at_one(self):
        """Strength should not exceed 1.0 after many rehearsals."""
        item = MemoryItem(id="t", content="test", memory_type=MemoryType.EPISODIC, strength=0.9)
        for _ in range(5):
            item.rehearse(strength_boost=0.2)
        assert item.strength <= 1.0


class TestImportance:
    def test_recompute_importance(self):
        """Importance should be computable from factors."""
        item = MemoryItem(id="t", content="test", memory_type=MemoryType.EPISODIC, strength=0.8)
        score = item.recompute_importance()
        assert 0.0 <= score <= 1.0

    def test_high_emotional_memory_gets_boost(self):
        """Emotionally intense memories should score higher."""
        calm = MemoryItem(id="c", content="calm", memory_type=MemoryType.EPISODIC, valence=0.1, arousal=0.1)
        intense = MemoryItem(id="i", content="wow!", memory_type=MemoryType.EPISODIC, valence=0.9, arousal=0.8)
        assert intense.recompute_importance(emotional_weight=0.4) > calm.recompute_importance(emotional_weight=0.4)


class TestEmotionalCongruence:
    def test_same_emotion_high_congruence(self):
        """Same valence and arousal should give high congruence."""
        item = MemoryItem(id="t", content="happy", memory_type=MemoryType.EPISODIC, valence=0.8, arousal=0.7)
        score = item.emotional_congruence(valence=0.8, arousal=0.7)
        assert score > 0.9

    def test_opposite_emotion_low_congruence(self):
        """Opposite valence should give low congruence."""
        item = MemoryItem(id="t", content="happy", memory_type=MemoryType.EPISODIC, valence=0.8, arousal=0.7)
        score = item.emotional_congruence(valence=-0.8, arousal=0.3)
        assert score < 0.4


class TestSerialization:
    def test_to_dict_roundtrip(self):
        """to_dict and from_dict should be inverses."""
        original = MemoryItem(
            id="ser_001", content="test memory", memory_type=MemoryType.EPISODIC,
            tags=["a", "b"], valence=0.5, arousal=0.3, metadata={"key": "value"},
        )
        data = original.to_dict()
        restored = MemoryItem.from_dict(data)
        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.memory_type == original.memory_type
        assert restored.tags == original.tags
        assert restored.valence == original.valence
        assert restored.arousal == original.arousal
        assert restored.metadata == original.metadata



