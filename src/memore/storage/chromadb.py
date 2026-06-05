"""ChromaDB storage backend — lightweight vector database for semantic search.

Requires the ``chromadb`` package: ``pip install memore[chromadb]``
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Collection, Dict, List, Optional

from memore.exceptions import DuplicateMemoryError, MemoryNotFoundError
from memore.memory.enums import ConsolidationStage, MemoryType
from memore.memory.item import MemoryItem
from memore.storage.base import StorageBackend


class ChromaDBBackend(StorageBackend):
    """ChromaDB-based storage with native vector search.

    Uses ChromaDB collections for storage and embedding-based
    similarity search. Metadata filtering for type/tag queries.

    Args:
        path: ChromaDB persistence path.
        collection_name: Name of the ChromaDB collection.
    """

    def __init__(self, path: str = "./chroma_data", collection_name: str = "memore_memories") -> None:
        self._path = path
        self._collection_name = collection_name
        self._client = None
        self._collection = None

    async def initialize(self) -> None:
        import chromadb
        self._client = chromadb.PersistentClient(path=self._path)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def close(self) -> None:
        self._client = None
        self._collection = None

    async def store(self, item: MemoryItem) -> str:
        self._ensure_ready()
        # Check for existing
        existing = self._collection.get(ids=[item.id])
        if existing and existing["ids"]:
            raise DuplicateMemoryError(f"Memory {item.id!r} already exists.")

        metadata = {
            "memory_type": item.memory_type.value,
            "memory_subtype": item.memory_subtype or "",
            "created_at": item.created_at.timestamp(),
            "last_accessed_at": item.last_accessed_at.timestamp(),
            "access_count": item.access_count,
            "strength": item.strength,
            "decay_rate": item.decay_rate,
            "importance": item.importance,
            "valence": item.valence,
            "arousal": item.arousal,
            "consolidation_stage": item.consolidation_stage.value,
            "tags": ",".join(item.tags),
            "source": item.source or "",
        }
        self._collection.add(
            ids=[item.id],
            documents=[item.content],
            metadatas=[metadata],
            embeddings=[item.embedding] if item.embedding else None,
        )
        return item.id

    async def batch_store(self, items: List[MemoryItem]) -> None:
        self._ensure_ready()
        for item in items:
            existing = self._collection.get(ids=[item.id])
            if existing and existing["ids"]:
                raise DuplicateMemoryError(f"Memory {item.id!r} already exists.")
        ids = [i.id for i in items]
        docs = [i.content for i in items]
        metas = [{
            "memory_type": i.memory_type.value,
            "created_at": i.created_at.timestamp(),
            "importance": i.importance,
            "consolidation_stage": i.consolidation_stage.value,
            "tags": ",".join(i.tags),
        } for i in items]
        embeddings = [i.embedding for i in items if i.embedding] or None
        self._collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    async def get(self, memory_id: str) -> Optional[MemoryItem]:
        self._ensure_ready()
        result = self._collection.get(ids=[memory_id])
        if not result or not result["ids"]:
            return None
        return self._result_to_item(result, 0)

    async def update(self, item: MemoryItem) -> None:
        self._ensure_ready()
        existing = self._collection.get(ids=[item.id])
        if not existing or not existing["ids"]:
            raise MemoryNotFoundError(f"Memory {item.id!r} not found.")
        await self.delete(item.id)
        await self.store(item)

    async def delete(self, memory_id: str) -> None:
        self._ensure_ready()
        self._collection.delete(ids=[memory_id])

    async def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        memory_types: Optional[Collection[MemoryType]] = None,
        limit: int = 20,
        threshold: float = 0.0,
        include_archived: bool = False,
        **metadata_filters: Any,
    ) -> List[MemoryItem]:
        self._ensure_ready()
        where: Optional[Dict] = None
        clauses = []
        if memory_types:
            types = [t.value for t in memory_types]
            if len(types) == 1:
                clauses.append({"memory_type": types[0]})
            else:
                clauses.append({"memory_type": {"$in": types}})
        if not include_archived:
            clauses.append({"consolidation_stage": {"$ne": "archived"}})
        if clauses:
            where = {"$and": clauses} if len(clauses) > 1 else clauses[0]

        result = self._collection.query(
            query_texts=[query] if query else None,
            query_embeddings=[query_embedding] if query_embedding else None,
            n_results=limit,
            where=where,
        )
        if not result or not result["ids"]:
            return []
        return [
            self._result_to_item(result, i)
            for i in range(len(result["ids"][0]))
        ]

    async def list(
        self,
        memory_types: Optional[Collection[MemoryType]] = None,
        importance_min: float = 0.0,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        include_archived: bool = False,
        **metadata_filters: Any,
    ) -> List[MemoryItem]:
        self._ensure_ready()
        where: Optional[Dict] = None
        clauses = [{"importance": {"$gte": importance_min}}]
        if memory_types:
            types = [t.value for t in memory_types]
            clauses.append({"memory_type": {"$in": types}} if len(types) > 1 else {"memory_type": types[0]})
        if not include_archived:
            clauses.append({"consolidation_stage": {"$ne": "archived"}})
        where = {"$and": clauses}

        result = self._collection.get(
            where=where,
            limit=limit,
            offset=offset,
        )
        if not result or not result["ids"]:
            return []
        items = [self._result_to_item(result, i) for i in range(len(result["ids"]))]
        items.sort(key=lambda i: getattr(i, sort_by, i.created_at), reverse=sort_desc)
        return items

    async def add_association(self, source_id: str, target_id: str, strength: float = 1.0) -> None:
        pass  # ChromaDB doesn't natively support graph associations

    async def remove_association(self, source_id: str, target_id: str) -> None:
        pass

    async def get_associated(self, memory_id: str, max_depth: int = 2, min_strength: float = 0.0) -> Dict[str, float]:
        return {}

    async def get_decaying(self, threshold: float = 0.3, limit: int = 100) -> List[MemoryItem]:
        return []  # ChromaDB cannot compute forgetting curve

    async def stats(self) -> Dict[str, Any]:
        self._ensure_ready()
        count = self._collection.count()
        return {"total": count, "associations": 0}

    async def clear(self) -> None:
        self._ensure_ready()
        all_items = self._collection.get()
        if all_items and all_items["ids"]:
            self._collection.delete(ids=all_items["ids"])

    def _ensure_ready(self) -> None:
        if self._collection is None:
            raise RuntimeError("ChromaDBBackend not initialized. Call initialize() first.")

    def _result_to_item(self, result: Any, index: int) -> MemoryItem:
        metadata = result["metadatas"][index] if result["metadatas"] else {}
        return MemoryItem(
            id=result["ids"][index],
            content=result["documents"][index],
            memory_type=MemoryType(metadata.get("memory_type", "episodic")),
            created_at=datetime.fromtimestamp(metadata.get("created_at", 0), tz=timezone.utc),
            last_accessed_at=datetime.fromtimestamp(metadata.get("last_accessed_at", 0), tz=timezone.utc),
            access_count=metadata.get("access_count", 0),
            strength=metadata.get("strength", 1.0),
            decay_rate=metadata.get("decay_rate", 0.1),
            importance=metadata.get("importance", 0.5),
            valence=metadata.get("valence", 0.0),
            arousal=metadata.get("arousal", 0.0),
            consolidation_stage=ConsolidationStage(metadata.get("consolidation_stage", "raw")),
            tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
            source=metadata.get("source"),
        )
