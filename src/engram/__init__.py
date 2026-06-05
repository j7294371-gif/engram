"""engram — biomimetic memory system for AI agents.

Engram provides AI agents with a human-like memory architecture:
sensory buffer → working memory → long-term memory (episodic,
semantic, procedural), with forgetting curves, sleep consolidation,
emotional tagging, and associative retrieval.

Core usage::

    from engram import AgentMemory

    memory = AgentMemory()
    memory.remember("User prefers Python over Java", memory_type="semantic")
    results = memory.recall("programming preferences")
"""

from engram.agent_memory import AgentMemory
from engram.config import Config
from engram.memory.item import MemoryItem
from engram.memory.enums import MemoryType, ConsolidationStage, RetrievalMode

__version__ = "0.1.0a1"
__all__ = [
    "AgentMemory",
    "Config",
    "MemoryItem",
    "MemoryType",
    "ConsolidationStage",
    "RetrievalMode",
]
