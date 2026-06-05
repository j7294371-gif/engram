"""Tests for the EmotionalTagger — valence/arousal circumplex model."""

from __future__ import annotations

from engram.emotion.tag import EmotionalTagger


class TestCategorize:
    def test_neutral_at_origin(self):
        assert EmotionalTagger.categorize(0.0, 0.0) == "neutral"

    def test_high_positive_arousal_is_ecstasy(self):
        assert EmotionalTagger.categorize(0.9, 0.9) == "ecstasy"

    def test_low_positive_arousal_is_contentment(self):
        assert EmotionalTagger.categorize(0.5, 0.3) == "contentment"

    def test_negative_high_arousal_is_anger_or_fear(self):
        """Strong negative + high arousal → anger or fear."""
        cat = EmotionalTagger.categorize(-0.7, 0.9)
        assert cat in ("anger", "fear")


class TestEncodingBoost:
    def test_high_arousal_high_valence_boosts(self):
        boost = EmotionalTagger.encoding_boost(0.9, 0.9)
        assert boost > 0.5

    def test_neutral_emotion_no_boost(self):
        boost = EmotionalTagger.encoding_boost(0.0, 0.0)
        assert boost == 0.0

    def test_negative_high_arousal_also_boosts(self):
        """Negative emotions also boost encoding (biological mirror)."""
        boost = EmotionalTagger.encoding_boost(-0.9, 0.9)
        assert boost > 0.5


class TestClassification:
    def test_is_positive(self):
        assert EmotionalTagger.is_positive(0.8)

    def test_is_negative(self):
        assert EmotionalTagger.is_negative(-0.8)

    def test_neutral_not_positive_or_negative(self):
        assert not EmotionalTagger.is_positive(0.0)
        assert not EmotionalTagger.is_negative(0.0)

    def test_is_intense(self):
        assert EmotionalTagger.is_intense(0.9, 0.9, threshold=0.6)
        assert not EmotionalTagger.is_intense(0.2, 0.2, threshold=0.6)


class TestDescribe:
    def test_returns_string(self):
        desc = EmotionalTagger.describe(0.8, 0.6)
        assert isinstance(desc, str)
        assert len(desc) > 5
