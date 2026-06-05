"""Basic usage example for the engram memory system.

Run with: python examples/basic_usage.py
"""

from engram import AgentMemory


def main():
    # ── Create a memory system ────────────────────────────────
    memory = AgentMemory()
    print("[Engram] Memory system initialized\n")

    # ── Store memories ────────────────────────────────────────
    print("[Store] Storing memories...")
    memory.remember("Alice loves hiking in the mountains", memory_type="semantic")
    memory.remember("Alice works as a software engineer", memory_type="semantic")
    memory.remember("Alice just got promoted to senior engineer", memory_type="episodic",
                    emotional_valence=0.9, emotional_arousal=0.7)
    memory.remember("Alice is planning a trip to Japan next month", memory_type="episodic")

    # ── Focus working memory ──────────────────────────────────
    memory.focus("Current task: help Alice plan her Japan trip")
    memory.focus("Reminder: ask about hike preferences")

    # ── Retrieve ──────────────────────────────────────────────
    print("\n[Recall] Recalling 'hiking'...")
    results = memory.recall("hiking")
    for r in results:
        prob = r.retrieval_probability()
        print(f"  [{r.memory_type.value:>10}] (p={prob:.2f}) {r.content}")

    # ── Get context ───────────────────────────────────────────
    print("\n[Context] Working memory context:")
    context = memory.get_context(window_size=5)
    for item in context:
        print(f"  [attention={item.attention_weight:.1f}] {item.content}")

    # ── Stats ─────────────────────────────────────────────────
    print("\n[Stats] System statistics:")
    stats = memory.stats()
    print(f"  Total memories: {stats['total']}")
    print(f"  Episodic: {stats.get('episodic', 0)}")
    print(f"  Semantic: {stats.get('semantic', 0)}")
    print(f"  Working memory: {stats['working_memory_size']} items loaded")

    # ── Consolidate ───────────────────────────────────────────
    print("\n[Consolidate] Running consolidation...")
    report = memory.consolidate()
    print(f"  Promotions: {report['promotions']}, Archived: {report['archived']}")

    print("\n[Done] Example completed successfully!")


if __name__ == "__main__":
    main()
