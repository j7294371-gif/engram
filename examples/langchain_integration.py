"""LangChain integration example — using Engram as a LangChain memory backend.

This example shows how to wrap Engram's AgentMemory as a custom
LangChain BaseMemory for use in conversational chains.

Run with: pip install langchain langchain-openai
          python examples/langchain_integration.py
"""

from __future__ import annotations

from typing import Any, Dict, List

from engram import AgentMemory


# ── Custom LangChain Memory Wrapper ──────────────────────────────

class EngramLangChainMemory:
    """A LangChain-compatible memory that uses Engram for storage.

    Wraps Engram's biomimetic memory system into LangChain's
    memory interface. Supports conversation history, entity
    extraction, and semantic recall.

    Usage::

        from langchain.memory import ConversationBufferMemory
        memory = EngramLangChainMemory(session_id="user_123")
        memory.chat_memory.add_user_message("Hello!")
        memory.chat_memory.add_ai_message("Hi there!")
    """

    def __init__(
        self,
        session_id: str = "default",
        engram_memory: AgentMemory | None = None,
    ) -> None:
        self.session_id = session_id
        self._memory = engram_memory or AgentMemory()
        self._turn_count = 0

    @property
    def chat_memory(self) -> "EngramChatMemory":
        """Return a chat-memory compatible interface."""
        return EngramChatMemory(self)

    @property
    def memory_variables(self) -> List[str]:
        return ["history", "context"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load conversation history and relevant context from Engram."""
        query = inputs.get("input", "")
        history = self._get_history()
        context = self._get_context(query)
        return {"history": history, "context": context}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save the current turn to Engram memory."""
        user_input = inputs.get("input", "")
        ai_output = outputs.get("output", "") or outputs.get("text", "")

        self._turn_count += 1

        # Store as episodic memory
        self._memory.remember(
            f"User ({self.session_id}): {user_input}",
            memory_type="episodic",
            tags=["conversation", "user"],
            metadata={"session": self.session_id, "turn": self._turn_count},
        )
        self._memory.remember(
            f"Assistant: {ai_output}",
            memory_type="episodic",
            tags=["conversation", "assistant"],
            metadata={"session": self.session_id, "turn": self._turn_count},
        )

        # Keep conversation in working memory for immediate context
        self._memory.focus(f"Turn {self._turn_count}: User said '{user_input[:50]}'")

        # Extract key facts as semantic memories
        self._extract_facts(user_input)

    def clear(self) -> None:
        """Clear all memories for this session."""
        self._memory.clear()
        self._turn_count = 0

    # ── Internal ────────────────────────────────────────────────

    def _get_history(self) -> str:
        """Format recent conversation history from working memory."""
        context = self._memory.get_context(window_size=10)
        lines = []
        for item in context:
            if item.memory_type.value == "episodic":
                lines.append(item.content)
        return "\n".join(lines[-6:])  # Last 3 turns

    def _get_context(self, query: str) -> str:
        """Retrieve semantically relevant context for the query."""
        results = self._memory.recall(
            query,
            memory_types=["episodic", "semantic"],
            limit=5,
            include_working=False,
        )
        return "\n".join(f"- {r.content}" for r in results)

    def _extract_facts(self, text: str) -> None:
        """Extract potential facts from user input as semantic memories."""
        # Simple heuristic: sentences with "is", "are", "was", "like", "prefer"
        indicators = [" is ", " are ", " was ", " like ", " prefer ", " hate ", " love "]
        for sentence in text.replace("!", ".").replace("?", ".").split("."):
            sentence = sentence.strip()
            if any(ind in sentence.lower() for ind in indicators) and len(sentence) > 10:
                self._memory.remember(
                    sentence,
                    memory_type="semantic",
                    tags=["fact", "extracted"],
                    metadata={"session": self.session_id},
                )


class EngramChatMemory:
    """Mimics LangChain's BaseChatMemory interface."""

    def __init__(self, parent: EngramLangChainMemory) -> None:
        self._parent = parent

    def add_user_message(self, message: str) -> None:
        self._parent.save_context({"input": message}, {"output": ""})

    def add_ai_message(self, message: str) -> None:
        # Store as the assistant part
        pass

    @property
    def messages(self) -> List[Dict[str, str]]:
        context = self._parent._memory.get_context(window_size=10)
        msgs = []
        for item in context:
            if "User" in item.content:
                msgs.append({"role": "user", "content": item.content})
            elif "Assistant" in item.content:
                msgs.append({"role": "assistant", "content": item.content})
        return msgs


# ── Demonstration ───────────────────────────────────────────────

def main():
    print("=== Engram + LangChain Integration Demo ===\n")

    # Create an Engram-backed memory
    memory = EngramLangChainMemory(session_id="demo_user")

    # Simulate a conversation
    turns = [
        "Hi! My name is Alice and I'm a Python developer.",
        "I really love using FastAPI for building APIs.",
        "Can you help me debug a performance issue?",
    ]

    for user_input in turns:
        ai_response = f"Sure Alice! Let me help you with that."
        memory.save_context({"input": user_input}, {"output": ai_response})
        print(f"User: {user_input}")
        print(f"AI: {ai_response}\n")

    # Show saved facts (semantic memories)
    print("--- Extracted Facts (Semantic Memory) ---")
    facts = memory._memory.recall("", memory_types=["semantic"], limit=10)
    for f in facts:
        print(f"  [{f.importance:.1f}] {f.content}")
    print()

    # Show context retrieval
    print("--- Context for 'Python FastAPI' ---")
    context = memory._memory.recall("Python FastAPI", limit=3)
    for c in context:
        print(f"  [{c.memory_type.value}] {c.content}")
    print()

    # Show stats
    print("--- System Stats ---")
    stats = memory._memory.stats()
    print(f"  Total memories: {stats['total']}")
    print(f"  Episodic: {stats.get('episodic', 0)}")
    print(f"  Semantic: {stats.get('semantic', 0)}")
    print()

    print("=== Demo Complete ===")


if __name__ == "__main__":
    main()
