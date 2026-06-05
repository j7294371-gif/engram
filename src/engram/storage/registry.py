"""Backend registration system for pluggable storage."""

from __future__ import annotations

from typing import Dict, List, Type

from engram.exceptions import BackendNotFoundError
from engram.storage.base import StorageBackend

_registry: Dict[str, Type[StorageBackend]] = {}


def register_backend(name: str, backend_cls: Type[StorageBackend]) -> None:
    """Register a storage backend class under a canonical name."""
    _registry[name] = backend_cls


def get_backend(name: str) -> Type[StorageBackend]:
    """Look up a registered backend by name.

    Raises ``BackendNotFoundError`` if the name is unknown.
    """
    if name not in _registry:
        raise BackendNotFoundError(
            f"Unknown backend: {name!r}. "
            f"Available backends: {list_backends()}"
        )
    return _registry[name]


def list_backends() -> List[str]:
    """Return the list of registered backend names."""
    return list(_registry)
