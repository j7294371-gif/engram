"""The MemoryItem dataclass — the fundamental unit of memory in memore."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from memore.memory.enums import ConsolidationStage, MemoryType


def _default_decay_rate(memory_type: MemoryType) -> float:
    """Return a biologically-plausible default decay rate per memory type.

    Semantic and procedural memories decay slower (more stable),
    sensory decays the fastest.
    """
    return {
        MemoryType.SENSORY: 10.0,
        MemoryType.WORKING: 2.0,
        MemoryType.EPISODIC: 0.15,
        MemoryType.SEMANTIC: 0.05,
        MemoryType.PROCEDURAL: 0.08,
    }[memory_type]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class MemoryItem:
    """A single atomic memory unit in the memore system.

    Every piece of information stored by an agent is encapsulated as
    a MemoryItem, with metadata supporting the full biomimetic pipeline:
    encoding, forgetting, consolidation, emotional tagging, and
    associative retrieval.

    Attributes:
        id: ULID-based unique identifier (time-sortable).
        content: The text payload of this memory.
        memory_type: Pipeline stage classification (sensory/working/episodic/...).
        memory_subtype: Optional finer-grained type (e.g., "skill", "routine").
        created_at: Immutable creation timestamp.
        last_accessed_at: Updated on every retrieval.
        access_count: Total number of retrievals.
        strength: Baseline encoding strength in [0, 1].
        decay_rate: Per-hour decay constant (Ebbinghaus parameter).
        last_rehearsed_at: Most recent rehearsal or access timestamp.
        valence: Emotional valence in [-1, 1] (negative to positive).
        arousal: Emotional arousal in [0, 1] (calm to intense).
        importance: Multi-factor importance score in [0, 1].
        attention_weight: Current working-memory attention allocation.
        consolidation_stage: Position in the consolidation pipeline.
        promoted_from: ID of source memory when promoted between stages.
        abstraction_source_ids: IDs of source memories for abstractions.
        associations: Map of {target_memory_id: association_strength}.
        embedding: Optional vector embedding for semantic search.
        embedding_model: Identifier of the embedding model used.
        tags: User-defined categorical labels.
        metadata: Arbitrary key-value store for extensibility.
        source: Identifier of the agent or module that created this memory.
    """

    # Identity
    id: str
    content: str
    memory_type: MemoryType
    memory_subtype: str | None = None

    # Temporal
    created_at: datetime = field(default_factory=_now_utc)
    last_accessed_at: datetime = field(default_factory=_now_utc)
    access_count: int = 0

    # Forgetting curve (Ebbinghaus)
    strength: float = 1.0
    decay_rate: float | None = None  # set by __post_init__
    last_rehearsed_at: datetime = field(default_factory=_now_utc)

    # Emotional
    valence: float = 0.0
    arousal: float = 0.0

    # Importance & attention
    importance: float = 0.5
    attention_weight: float = 0.0
    capacity_slot: int | None = None

    # Associations
    associations: dict[str, float] = field(default_factory=dict)

    # Consolidation
    consolidation_stage: ConsolidationStage = ConsolidationStage.RAW
    consolidated_at: datetime | None = None
    promoted_from: str | None = None
    abstraction_source_ids: list[str] = field(default_factory=list)

    # Embedding
    embedding: list[float] | None = None
    embedding_model: str | None = None

    # Metadata
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str | None = None

    def __post_init__(self) -> None:
        """Apply defaults and validate constraints after initialization."""
        if self.decay_rate is None:
            self.decay_rate = _default_decay_rate(self.memory_type)
        # Clamp valence and arousal
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.strength = max(0.0, min(1.0, self.strength))
        self.importance = max(0.0, min(1.0, self.importance))

    # ── Forgetting curve (Ebbinghaus) ──────────────────────────

    def retrieval_probability(self, at_time: datetime | None = None) -> float:
        """Ebbinghaus forgetting curve: P = S × exp(−d × Δt)

        Returns the probability that this memory can be retrieved
        after elapsed time since last rehearsal.
        """
        delta = ((at_time or _now_utc()) - self.last_rehearsed_at).total_seconds()
        hours = delta / 3600.0
        return self.strength * math.exp(-self.decay_rate * hours)

    def is_forgotten(self, threshold: float = 0.05, at_time: datetime | None = None) -> bool:
        """Has this memory decayed below the forgetting threshold?"""
        return self.retrieval_probability(at_time) < threshold

    # ── Rehearsal ───────────────────────────────────────────────

    def rehearse(self, strength_boost: float = 0.1) -> None:
        """Rehearse this memory: reset decay clock and boost strength.

        Mirrors the biological rehearsal effect — repeating or recalling
        information resets the forgetting curve and slightly
        strengthens the memory trace.
        """
        self.last_rehearsed_at = _now_utc()
        self.strength = min(1.0, self.strength + strength_boost)
        self.access_count += 1

    # ── Importance ──────────────────────────────────────────────

    def recompute_importance(
        self,
        recency_weight: float = 0.2,
        frequency_weight: float = 0.15,
        emotional_weight: float = 0.2,
        attention_weight: float = 0.15,
        base_weight: float = 0.3,
    ) -> float:
        """Multi-factor importance score.

        Combines recency, access frequency, emotional intensity,
        attention allocation, and base encoding strength.
        """
        now = _now_utc()
        recency = math.exp(-(now - self.last_accessed_at).total_seconds() / 86400.0)
        frequency = min(self.access_count / 10.0, 1.0)
        emotional = abs(self.valence) * self.arousal
        attention = self.attention_weight

        self.importance = (
            base_weight * self.strength
            + recency_weight * recency
            + frequency_weight * frequency
            + emotional_weight * emotional
            + attention_weight * attention
        )
        return self.importance

    # ── Emotional congruence ────────────────────────────────────

    def emotional_congruence(self, valence: float, arousal: float) -> float:
        """Compute mood-congruence score with a given emotional state.

        Higher scores indicate this memory is more congruent with
        the current emotional context (mood-congruent retrieval bias).
        """
        valence_sim = 1.0 - abs(self.valence - valence) / 2.0  # normalize [-1,1] diff
        arousal_sim = 1.0 - abs(self.arousal - arousal)
        return 0.6 * valence_sim + 0.4 * arousal_sim

    # ── Utility ─────────────────────────────────────────────────

    def touch(self) -> None:
        """Record an access to this memory."""
        self.last_accessed_at = _now_utc()
        self.access_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "memory_subtype": self.memory_subtype,
            "created_at": self.created_at.isoformat(),
            "last_accessed_at": self.last_accessed_at.isoformat(),
            "access_count": self.access_count,
            "strength": self.strength,
            "decay_rate": self.decay_rate,
            "last_rehearsed_at": self.last_rehearsed_at.isoformat(),
            "valence": self.valence,
            "arousal": self.arousal,
            "importance": self.importance,
            "attention_weight": self.attention_weight,
            "consolidation_stage": self.consolidation_stage.value,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryItem:
        """Deserialize from a dictionary produced by ``to_dict``."""
        from copy import deepcopy

        d = deepcopy(data)
        d["memory_type"] = MemoryType(d["memory_type"])
        d["consolidation_stage"] = ConsolidationStage(d.get("consolidation_stage", "raw"))
        for key in ("created_at", "last_accessed_at", "last_rehearsed_at"):
            if isinstance(d.get(key), str):
                d[key] = datetime.fromisoformat(d[key])
        d.pop("embedding", None)  # embeddings are handled separately
        return cls(**d)
