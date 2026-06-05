"""Emotional tagging — valence/arousal circumplex model.

Maps emotions along two dimensions:
- Valence: pleasure/displeasure (-1 to +1)
- Arousal: activation/deactivation (0 to 1)

Categorizes into basic emotion families for higher-level reasoning.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple


# Emotion categories defined by (valence, arousal) centroids
EMOTION_CATEGORIES: Dict[str, Tuple[float, float]] = {
    "ecstasy": (0.9, 0.9),
    "joy": (0.8, 0.6),
    "serenity": (0.7, 0.2),
    "contentment": (0.5, 0.3),
    "neutral": (0.0, 0.0),
    "sadness": (-0.6, 0.3),
    "grief": (-0.9, 0.6),
    "anger": (-0.7, 0.9),
    "fear": (-0.6, 0.8),
    "anxiety": (-0.4, 0.7),
    "surprise": (0.2, 0.9),
    "disgust": (-0.7, 0.5),
    "boredom": (-0.3, 0.1),
    "excitement": (0.6, 0.9),
    "frustration": (-0.5, 0.6),
    "hope": (0.4, 0.5),
}


class EmotionalTagger:
    """Assigns and analyzes emotional tags on memories.

    Uses the circumplex model of emotion (Russell, 1980) to provide
    categorical labels from continuous valence/arousal dimensions.
    """

    @staticmethod
    def categorize(valence: float, arousal: float) -> str:
        """Map valence/arousal to the nearest emotion category.

        Args:
            valence: Emotional valence [-1, 1].
            arousal: Emotional arousal [0, 1].

        Returns:
            The nearest emotion category label.
        """
        best_label = "neutral"
        best_dist = float("inf")
        for label, (v, a) in EMOTION_CATEGORIES.items():
            dist = math.sqrt((valence - v) ** 2 + (arousal - a) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_label = label
        return best_label

    @staticmethod
    def encoding_boost(valence: float, arousal: float) -> float:
        """Compute encoding strength boost from emotional intensity.

        Highly arousing experiences (positive or negative) are
        encoded more strongly — mirrors the biological effect of
        emotional arousal on memory consolidation.

        Returns:
            Boost factor in [0, 1].
        """
        intensity = abs(valence) * arousal
        return min(1.0, intensity * 1.5)

    @staticmethod
    def is_positive(valence: float) -> bool:
        """Is this emotion positive?"""
        return valence > 0.2

    @staticmethod
    def is_negative(valence: float) -> bool:
        """Is this emotion negative?"""
        return valence < -0.2

    @staticmethod
    def is_intense(valence: float, arousal: float, threshold: float = 0.6) -> bool:
        """Is this emotion intense enough to affect encoding?"""
        return abs(valence) * arousal > threshold

    @staticmethod
    def describe(valence: float, arousal: float) -> str:
        """Human-readable description of an emotional state."""
        category = EmotionalTagger.categorize(valence, arousal)
        intensity = "strongly" if abs(valence) * arousal > 0.5 else "moderately"
        return f"{intensity} {category} (v={valence:.1f}, a={arousal:.1f})"
