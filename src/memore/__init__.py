"""memore — biomimetic memory system for AI agents.

Engram provides AI agents with a human-like memory architecture:
sensory buffer → working memory → long-term memory (episodic,
semantic, procedural), with forgetting curves, sleep consolidation,
emotional tagging, and associative retrieval.

Core usage::

    from memore import AgentMemory

    memory = AgentMemory()
    memory.remember("User prefers Python over Java", memory_type="semantic")
    results = memory.recall("programming preferences")
"""

from memore.agent_memory import AgentMemory
from memore.config import Config
from memore.consolidation.sleep import ConsolidationReport
from memore.emotion.tag import EmotionalTagger
from memore.memory.enums import ConsolidationStage, MemoryType, RetrievalMode
from memore.memory.item import MemoryItem

__version__ = "0.1.0a1"
__all__ = [
    "AgentMemory",
    "Config",
    "ConsolidationReport",
    "EmotionalTagger",
    "MemoryItem",
    "MemoryType",
    "ConsolidationStage",
    "RetrievalMode",
]
