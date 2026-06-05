# 🧠 Engram — Biomimetic Memory for AI Agents

[![PyPI version](https://img.shields.io/pypi/v/engram?color=blue)](https://pypi.org/project/engram/)
[![Python versions](https://img.shields.io/pypi/pyversions/engram)](https://pypi.org/project/engram/)
[![CI](https://github.com/engram-memory/engram/actions/workflows/ci.yml/badge.svg)](https://github.com/engram-memory/engram/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen)]()
[![PyPI Downloads](https://img.shields.io/pypi/dm/engram)](https://pypi.org/project/engram/)

**Engram** gives AI agents a memory system inspired by the human brain. Not just RAG — a full biomimetic pipeline with sensory buffers, working memory, episodic/semantic/procedural stores, Ebbinghaus forgetting curves, sleep consolidation, emotional tagging, and associative retrieval.

```python
from engram import AgentMemory

memory = AgentMemory()
memory.remember("User prefers Python over Java", memory_type="semantic")
results = memory.recall("programming preferences")
```

## 🔬 Why Engram?

Existing agent memory solutions are either too simple (flat RAG stores), too complex (tied to specific agent frameworks), or not truly open source.

| Feature | Engram | Mem0 | MemGPT | LangChain | Zep |
|---------|--------|------|--------|-----------|-----|
| 3-tier memory pipeline | ✅ | ❌ | ❌ | ❌ | ❌ |
| Forgetting curve (Ebbinghaus) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Sleep consolidation | ✅ | ❌ | ❌ | ❌ | ❌ |
| Emotional tagging | ✅ | ❌ | ❌ | ❌ | ❌ |
| Associative retrieval | ✅ | ❌ | ❌ | ❌ | ❌ |
| Pluggable backends | ✅ | ✅ | ❌ | ❌ | ❌ |
| Zero dependencies | ✅ | ❌ | ❌ | ❌ | ❌ |
| MIT licensed | ✅ | ❌ | ❌ | ✅ | ❌ |
| Framework-agnostic | ✅ | ✅ | ❌ | ❌ | ✅ |

## 🏗 Architecture

```
Sensory Buffer ──→ Working Memory ──→ Long-term Memory
  (ring buffer)      (7±2 capacity)      ├─ Episodic (events)
  TTL: ~30s         attention-weighted   ├─ Semantic (facts)
                                          └─ Procedural (skills)

Forgetting Curve ← Sleep Consolidation ← Associative Retrieval ↔ Emotion
```

## 🚀 Quick Start

```bash
pip install engram
```

```python
from engram import AgentMemory

# Create a memory system (default: in-memory backend)
memory = AgentMemory()

# Store memories
memory.remember("Alice loves hiking in the mountains", memory_type="semantic")

# Store with emotional context
memory.remember("Alice got a promotion!", memory_type="episodic",
                emotional_valence=0.9, emotional_arousal=0.7)

# Focus working memory
memory.focus("Current task: plan weekend hike")

# Retrieve
results = memory.recall("outdoor activities")
for r in results:
    print(f"[{r.memory_type.value}] {r.content} (relevance: {r.retrieval_probability():.2f})")

# Get current context for LLM prompt
context = memory.get_context(window_size=5)
for item in context:
    print(f"⚡ {item.content}")

# Consolidate (run periodically)
report = memory.consolidate()
print(f"Promoted: {report['promotions']}, Archived: {report['archived']}")
```

## 📦 Installation

```bash
# Core (zero dependencies)
pip install engram

# With SQLite backend
pip install engram[sqlite]

# With embedding support
pip install engram[embeddings]

# All backends and embeddings
pip install engram[all]

# Development
pip install engram[dev]
```

## 🎯 API Overview

### Tier 1: Core (3 lines)

```python
from engram import AgentMemory
memory = AgentMemory()
memory.remember("content", memory_type="episodic")
results = memory.recall("query")
```

### Tier 2: Standard

```python
memory.remember("content", emotional_valence=0.5, emotional_arousal=0.8)
memory.focus("current task")
context = memory.get_context(window_size=5)
memory.associate(id1, id2, strength=0.9)
results = memory.search("query", memory_types=["semantic", "episodic"])
memory.tag(id, "important", "review")
memory.tag_emotion(id, valence=0.7, arousal=0.3)
```

### Tier 3: Advanced

```python
report = memory.consolidate_sleep()          # Full sleep cycle
chain = memory.retrieve_associated(id, max_depth=3)
results = memory.recall("query", mood_congruent=(0.5, 0.3))
memory.rehearse(id)                          # Boost strength
stats = memory.stats()
```

## 🔌 MCP Server

Engram can run as a **Model Context Protocol (MCP) server**, making it available to any MCP-compatible client (Claude Code, Cursor, Windsurf, etc.).

```bash
# Start the MCP server
python -m engram.mcp_server.server

# Or install globally
pip install engram
engram-mcp
```

Configure in Claude Code's `settings.json`:
```json
{
  "mcpServers": {
    "engram": {
      "command": "python",
      "args": ["-m", "engram.mcp_server.server"]
    }
  }
}
```

**Available tools:** `remember`, `recall`, `search`, `forget`, `consolidate`, `stats`, `focus`, `get_context`

## 📊 Benchmark

```bash
python examples/benchmark.py
```

| Metric | Value |
|--------|-------|
| Write throughput (SQLite) | ~1,200/sec |
| Write throughput (Memory) | ~1,000/sec |
| Recall precision | 80% (500 items) |
| Avg query latency | 3.2ms |
| Consolidation (light) | 16ms for 500 items |

## 🔧 Backends

| Backend | Install | Best for |
|---------|---------|----------|
| InMemory | (built-in) | Testing, prototyping |
| SQLite | `pip install engram` | Default, embedded |
| ChromaDB | `pip install engram[chromadb]` | Lightweight vector search |
| Qdrant | `pip install engram[qdrant]` | Production vector search |
| pgvector | `pip install engram[pgvector]` | PostgreSQL users |

```python
# Use a specific backend
memory = AgentMemory(backend="chromadb")
```

## 📖 Documentation

Full documentation is available on [GitHub](https://github.com/j7294371-gif/engram).

- [Getting Started](https://github.com/j7294371-gif/engram#-quick-start)
- [Memory Pipeline Guide](src/engram/pipeline/)
- [Forgetting Curve](src/engram/memory/decay.py)
- [Consolidation](src/engram/consolidation/sleep.py)
- [Emotional Tagging](src/engram/emotion/tag.py)
- [Associative Retrieval](src/engram/retrieval/)
- [Backend Guide](src/engram/storage/)
- [API Reference](src/engram/agent_memory.py)
- [Examples](examples/)

## 🧪 Testing

```bash
pip install engram[dev]
pytest
```

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT — see [LICENSE](LICENSE).

---

*Engram: memory traces that last.*
