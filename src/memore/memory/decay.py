"""Ebbinghaus forgetting curve — mathematical model of memory decay."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Optional


def retrieval_probability(
    strength: float,
    decay_rate: float,
    last_rehearsed_at: datetime,
    at_time: Optional[datetime] = None,
) -> float:
    """Ebbinghaus forgetting curve.

    ``P = S × exp(−d × Δt)``

    Where:
    - P = probability of retrieval
    - S = encoding strength (0 to 1)
    - d = decay rate (per hour)
    - Δt = elapsed time since last rehearsal (hours)

    Args:
        strength: Baseline encoding strength [0, 1].
        decay_rate: Per-hour decay constant.
        last_rehearsed_at: Timestamp of last rehearsal/access.
        at_time: Query time. Defaults to now.

    Returns:
        Retrieval probability in [0, 1].
    """
    if at_time is None:
        at_time = datetime.now(timezone.utc)
    delta_hours = (at_time - last_rehearsed_at).total_seconds() / 3600.0
    return strength * math.exp(-decay_rate * delta_hours)


def hours_until_forgotten(
    strength: float,
    decay_rate: float,
    threshold: float = 0.05,
) -> float:
    """Calculate how many hours until a memory falls below threshold.

    ``t = −ln(threshold / S) / d``

    Args:
        strength: Encoding strength.
        decay_rate: Per-hour decay constant.
        threshold: Retrieval probability threshold for "forgotten".

    Returns:
        Hours until the memory is considered forgotten.
    """
    if strength <= 0 or decay_rate <= 0:
        return 0.0
    ratio = threshold / strength
    if ratio <= 0:
        return float("inf")
    return -math.log(ratio) / decay_rate


def strength_after_rehearsals(
    initial_strength: float,
    num_rehearsals: int,
    boost_per_rehearsal: float = 0.1,
    max_strength: float = 1.0,
) -> float:
    """Compute strength after multiple rehearsals with diminishing returns.

    Each rehearsal adds ``boost_per_rehearsal`` but the effect
    diminishes logarithmically to model biological saturation.
    """
    boost = boost_per_rehearsal * math.log(1 + num_rehearsals)
    return min(max_strength, initial_strength + boost)


def optimal_rehearsal_intervals(
    decay_rate: float,
    threshold: float = 0.7,
    max_intervals: int = 5,
) -> list[float]:
    """Compute spaced-repetition rehearsal intervals (hours).

    Based on the idea that each rehearsal should happen when
    retrieval probability drops to ``threshold``. Returns
    increasingly spaced intervals, mirroring the Ebbinghaus
    spacing effect.
    """
    intervals: list[float] = []
    current_time = 0.0
    for i in range(max_intervals):
        # Each interval is longer than the last
        time_to_decay = -math.log(threshold) / decay_rate
        current_time += time_to_decay * (1 + i * 0.5)
        intervals.append(current_time)
        # Decay rate effectively decreases after each rehearsal
        decay_rate *= 0.85
    return intervals
