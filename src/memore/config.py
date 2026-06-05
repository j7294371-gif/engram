"""Global configuration for the memore memory system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Config:
    """Configuration for an AgentMemory instance.

    Provides sensible biomimetic defaults. Most users need only
    set ``backend`` and perhaps ``embedding_model``.
    """

    # ── Backend ──────────────────────────────────────────────────
    backend: str = "in_memory"
    backend_config: Dict[str, Any] = field(default_factory=dict)

    # ── Embedding ────────────────────────────────────────────────
    embedding_model: str = "none"
    embedding_dim: int = 384
    embedding_config: Dict[str, Any] = field(default_factory=dict)

    # ── Sensory Buffer ──────────────────────────────────────────
    sensory_buffer_size: int = 50
    sensory_decay_seconds: float = 30.0

    # ── Working Memory ──────────────────────────────────────────
    working_memory_capacity: int = 7  # Miller's law: 7 ± 2
    working_attention_threshold: float = 0.1

    # ── Forgetting Curve (Ebbinghaus) ───────────────────────────
    default_decay_rate: float = 0.1
    episodic_decay_rate: float = 0.15
    semantic_decay_rate: float = 0.05
    procedural_decay_rate: float = 0.08
    rehearsal_strength_boost: float = 0.1
    forgetting_threshold: float = 0.1      # 调高：低于10%概率 → 视为遗忘

    # ── Storage Quota & Pruning ─────────────────────────────────
    max_items: int = 10000                 # 硬上限，超过后自动裁剪
    prune_below_importance: float = 0.1    # 裁剪时移除重要性低于此值的记忆
    keep_at_most: int = 8000               # 裁剪后保留的数量
    auto_archive_on_recall: bool = True    # 检索时自动归档低概率记忆

    # ── Consolidation ───────────────────────────────────────────
    auto_consolidate: bool = True
    consolidation_interval_minutes: int = 60
    promotion_importance_threshold: float = 0.7
    abstraction_min_sources: int = 3
    operations_before_prune: int = 50      # 每N次操作触发一次自动整理

    # ── Retrieval ────────────────────────────────────────────────
    default_limit: int = 20
    similarity_threshold: float = 0.5
    spreading_activation_decay: float = 0.5
    spreading_max_depth: int = 3
    hybrid_semantic_weight: float = 0.5
    hybrid_associative_weight: float = 0.3
    hybrid_importance_weight: float = 0.2

    # ── Emotional ───────────────────────────────────────────────
    emotional_encoding_boost: float = 0.3
    mood_congruent_bias: float = 0.2

    # ── System ──────────────────────────────────────────────────
    log_level: str = "INFO"
