"""Multi-factor importance scoring for memory items."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional


def compute_importance(
    strength: float,
    access_count: int,
    last_accessed_at: datetime,
    valence: float,
    arousal: float,
    attention_weight: float,
    *,
    recency_weight: float = 0.2,
    frequency_weight: float = 0.15,
    emotional_weight: float = 0.2,
    attention_weight_factor: float = 0.15,
    base_weight: float = 0.3,
) -> float:
    """Multi-factor importance score in [0, 1].

    Combines five factors:
    1. Base encoding strength
    2. Recency (exponential decay over days)
    3. Access frequency (capped at 10 accesses)
    4. Emotional intensity (|valence| × arousal)
    5. Current attention allocation

    Returns:
        Importance score in [0, 1].
    """
    now = datetime.now(timezone.utc)

    # Recency score: exponential decay over 24h
    recency = math.exp(-(now - last_accessed_at).total_seconds() / 86400.0)

    # Frequency score: saturating at 10 accesses
    frequency = min(access_count / 10.0, 1.0)

    # Emotional intensity
    emotional = abs(valence) * arousal

    return (
        base_weight * _clamp(strength)
        + recency_weight * recency
        + frequency_weight * frequency
        + emotional_weight * emotional
        + attention_weight_factor * _clamp(attention_weight)
    )


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
