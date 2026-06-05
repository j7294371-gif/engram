"""Shared fixtures for memore tests."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Generator

import pytest

from memore.config import Config
from memore.memory.enums import MemoryType
from memore.memory.item import MemoryItem
from memore.storage.in_memory import InMemoryBackend


@pytest.fixture
def sample_memory() -> MemoryItem:
    """A basic episodic memory item for testing."""
    return MemoryItem(
        id="test_001",
        content="The user prefers Python over Java for backend development.",
        memory_type=MemoryType.EPISODIC,
        importance=0.8,
        tags=["python", "preference", "backend"],
        source="test-fixture",
    )


@pytest.fixture
def sample_semantic() -> MemoryItem:
    """A basic semantic memory item for testing."""
    return MemoryItem(
        id="sem_001",
        content="Python is a dynamically-typed programming language.",
        memory_type=MemoryType.SEMANTIC,
        importance=0.9,
        decay_rate=0.05,
        tags=["python", "fact"],
    )


@pytest.fixture
def sample_emotional() -> MemoryItem:
    """A memory with strong emotional valence."""
    return MemoryItem(
        id="emo_001",
        content="The user was extremely happy when the project succeeded!",
        memory_type=MemoryType.EPISODIC,
        valence=0.8,
        arousal=0.7,
        importance=0.9,
    )


@pytest.fixture
def empty_backend() -> InMemoryBackend:
    """A fresh in-memory backend."""
    return InMemoryBackend()


@pytest.fixture
def populated_backend(empty_backend: InMemoryBackend) -> Generator[InMemoryBackend, None, None]:
    """An in-memory backend pre-populated with test memories."""
    empty_backend.initialize()

    memories = [
        MemoryItem(id="pop_1", content="First memory", memory_type=MemoryType.EPISODIC, tags=["a"]),
        MemoryItem(id="pop_2", content="Second memory about Python", memory_type=MemoryType.EPISODIC, tags=["b"]),
        MemoryItem(id="pop_3", content="Third memory about Rust", memory_type=MemoryType.SEMANTIC, tags=["c"]),
        MemoryItem(id="pop_4", content="Fourth memory", memory_type=MemoryType.PROCEDURAL, tags=["d"]),
    ]
    empty_backend.batch_store(memories)
    yield empty_backend
    empty_backend.close()


@pytest.fixture
def default_config() -> Config:
    return Config()
