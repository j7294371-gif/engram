"""Custom exceptions for the engram memory system."""


class EngramError(Exception):
    """Base exception for all engram errors."""


class MemoryNotFoundError(EngramError):
    """Raised when attempting to access a memory that does not exist."""


class DuplicateMemoryError(EngramError):
    """Raised when a memory with a duplicate ID is stored."""


class BackendError(EngramError):
    """Raised when a storage backend operation fails."""


class BackendNotFoundError(EngramError):
    """Raised when an unknown backend name is requested."""


class EmbeddingError(EngramError):
    """Raised when embedding generation fails."""


class ConsolidationError(EngramError):
    """Raised when a consolidation operation fails."""


class ConfigurationError(EngramError):
    """Raised when configuration validation fails."""
