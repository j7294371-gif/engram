"""Backend registration system for pluggable storage."""

from __future__ import annotations

from memore.exceptions import BackendNotFoundError
from memore.storage.base import StorageBackend

_registry: dict[str, type[StorageBackend]] = {}


def register_backend(name: str, backend_cls: type[StorageBackend]) -> None:
    """Register a storage backend class under a canonical name."""
    _registry[name] = backend_cls


def get_backend(name: str) -> type[StorageBackend]:
    """Look up a registered backend by name.

    Raises ``BackendNotFoundError`` if the name is unknown.
    """
    if name not in _registry:
        raise BackendNotFoundError(
            f"Unknown backend: {name!r}. "
            f"Available backends: {list_backends()}"
        )
    return _registry[name]


def list_backends() -> list[str]:
    """Return the list of registered backend names."""
    return list(_registry)
