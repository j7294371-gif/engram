"""Core enums for the engram memory system."""

from __future__ import annotations

import enum


class MemoryType(enum.Enum):
    """Categorization of memory items along the biomimetic pipeline.

    Maps to the human memory hierarchy:
    - SENSORY: Ultra-short-term perceptual buffer (seconds)
    - WORKING: Active, task-focused context (limited capacity ~7±2)
    - EPISODIC: Autobiographical events and experiences
    - SEMANTIC: Abstracted facts, concepts, and general knowledge
    - PROCEDURAL: Skills, routines, and how-to patterns
    """

    SENSORY = "sensory"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class ConsolidationStage(enum.Enum):
    """Lifecycle stage of a memory item through the consolidation pipeline.

    Mirrors the process of memory stabilization in biological systems:
    - RAW: Freshly created, unprocessed
    - WORKING_PROMOTED: Elevated from working to episodic store
    - SEMANTIC_EXTRACTED: Abstracted into semantic knowledge
    - PROCEDURAL_EXTRACTED: Pattern-extracted into procedural skill
    - CONSOLIDATED: Fully stabilized
    - ARCHIVED: Below retrieval threshold (forgotten but recoverable)
    """

    RAW = "raw"
    WORKING_PROMOTED = "working_promoted"
    SEMANTIC_EXTRACTED = "semantic_extracted"
    PROCEDURAL_EXTRACTED = "procedural_extracted"
    CONSOLIDATED = "consolidated"
    ARCHIVED = "archived"


class RetrievalMode(enum.Enum):
    """Available retrieval strategies for searching memories."""

    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    ASSOCIATIVE = "associative"
    KEYWORD = "keyword"
