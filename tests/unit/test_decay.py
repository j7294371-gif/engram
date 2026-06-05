"""Tests for the Ebbinghaus forgetting curve module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from engram.memory.decay import (
    hours_until_forgotten,
    optimal_rehearsal_intervals,
    retrieval_probability,
    strength_after_rehearsals,
)


class TestRetrievalProbability:
    def test_returns_one_at_zero_time(self):
        p = retrieval_probability(1.0, 0.1, datetime.now(timezone.utc))
        assert abs(p - 1.0) < 1e-6

    def test_decreases_with_time(self):
        now = datetime.now(timezone.utc)
        p1 = retrieval_probability(1.0, 0.1, now - timedelta(hours=1))
        p2 = retrieval_probability(1.0, 0.1, now - timedelta(hours=2))
        assert p1 > p2

    def test_higher_strength_means_higher_probability(self):
        now = datetime.now(timezone.utc)
        past = now - timedelta(hours=5)
        p_high = retrieval_probability(0.9, 0.1, past, now)
        p_low = retrieval_probability(0.5, 0.1, past, now)
        assert p_high > p_low

    def test_higher_decay_means_lower_probability(self):
        now = datetime.now(timezone.utc)
        past = now - timedelta(hours=5)
        p_fast = retrieval_probability(1.0, 0.5, past, now)
        p_slow = retrieval_probability(1.0, 0.1, past, now)
        assert p_fast < p_slow

    def test_never_negative(self):
        now = datetime.now(timezone.utc)
        past = now - timedelta(days=365)
        p = retrieval_probability(0.5, 0.1, past, now)
        assert p >= 0.0


class TestHoursUntilForgotten:
    def test_returns_positive(self):
        h = hours_until_forgotten(1.0, 0.1, threshold=0.05)
        assert h > 0

    def test_higher_strength_takes_longer(self):
        h_high = hours_until_forgotten(0.9, 0.1, 0.05)
        h_low = hours_until_forgotten(0.5, 0.1, 0.05)
        assert h_high > h_low

    def test_higher_decay_forgets_faster(self):
        h_fast = hours_until_forgotten(1.0, 0.5, 0.05)
        h_slow = hours_until_forgotten(1.0, 0.1, 0.05)
        assert h_fast < h_slow

    def test_zero_strength_returns_zero(self):
        h = hours_until_forgotten(0.0, 0.1, 0.05)
        assert h == 0.0


class TestStrengthAfterRehearsals:
    def test_increases_with_rehearsals(self):
        s = strength_after_rehearsals(0.5, 3, boost_per_rehearsal=0.1)
        assert s > 0.5

    def test_capped_at_max(self):
        s = strength_after_rehearsals(0.95, 10, boost_per_rehearsal=0.1)
        assert s <= 1.0


class TestOptimalRehearsalIntervals:
    def test_returns_correct_number(self):
        intervals = optimal_rehearsal_intervals(0.1, threshold=0.7, max_intervals=5)
        assert len(intervals) == 5

    def test_intervals_increase(self):
        intervals = optimal_rehearsal_intervals(0.1, threshold=0.7, max_intervals=4)
        for i in range(1, len(intervals)):
            assert intervals[i] > intervals[i - 1]
