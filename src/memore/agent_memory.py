"""AgentMemory — the primary interface to the memore memory system.

Combines sensory buffer, working memory, and long-term memory into
a unified biomimetic pipeline with forgetting, consolidation,
emotional tagging, and associative retrieval.
"""

from __future__ import annotations

import secrets
import time
from typing import Any

from memore.config import Config
from memore.consolidation.sleep import ConsolidationReport, SleepConsolidation
from memore.embedding import NoopEmbeddingProvider
from memore.embedding.base import EmbeddingProvider
from memore.memory.enums import ConsolidationStage, MemoryType
from memore.memory.item import MemoryItem
from memore.pipeline.long_term_memory import LongTermMemory
from memore.pipeline.sensory_buffer import SensoryBuffer
from memore.pipeline.working_memory import WorkingMemory
from memore.retrieval.hybrid import HybridRanker
from memore.storage import StorageBackend, get_backend, register_backend
from memore.storage.in_memory import InMemoryBackend


class AgentMemory:
    """Primary interface to the memore memory system.

    A unified facade over the three-tier biomimetic pipeline:
    sensory — working — long-term (episodic / semantic / procedural).

    Usage::

        from memore import AgentMemory

        memory = AgentMemory()
        mid = memory.remember("User prefers Python")
        results = memory.recall("programming preferences")
        ctx = memory.get_context()
    """

    def __init__(
        self,
        backend: str | StorageBackend | None = None,
        embedding: EmbeddingProvider | None = None,
        config: Config | None = None,
        auto_register: bool = True,
    ) -> None:
        if auto_register:
            _ensure_backends_registered()

        self._config = config or Config()

        # Resolve backend
        if backend is None:
            self._backend = InMemoryBackend()
        elif isinstance(backend, str):
            backend_cls = get_backend(backend)
            self._backend = backend_cls()
        else:
            self._backend = backend

        # Resolve embedding
        self._embedding = embedding or NoopEmbeddingProvider()

        # Pipeline stages
        self._ltm = LongTermMemory(backend=self._backend)
        self._sensory = SensoryBuffer(
            capacity=self._config.sensory_buffer_size,
            ttl_seconds=self._config.sensory_decay_seconds,
        )
        self._working = WorkingMemory(
            capacity=self._config.working_memory_capacity,
            on_evict=self._on_working_evict,
        )

        # Hybrid ranker (semantic + associative + importance fusion)
        self._ranker = HybridRanker(
            semantic_weight=self._config.hybrid_semantic_weight,
            associative_weight=self._config.hybrid_associative_weight,
            importance_weight=self._config.hybrid_importance_weight,
        )

        # Sleep consolidation engine
        self._consolidator = SleepConsolidation(
            backend=self._backend,
            promotion_importance_threshold=self._config.promotion_importance_threshold,
            abstraction_min_sources=self._config.abstraction_min_sources,
            forgetting_threshold=self._config.forgetting_threshold,
        )

        # Operation counter for auto-prune
        self._op_count = 0

    # ──────────────────────────────────────────────────────────────
    # Tier 1: Core API
    # ──────────────────────────────────────────────────────────────

    def remember(
        self,
        content: str,
        memory_type: str = "episodic",
        *,
        tags: list[str] | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
        emotional_valence: float | None = None,
        emotional_arousal: float | None = None,
        importance: float | None = None,
        associations: dict[str, float] | None = None,
    ) -> str:
        """Store a memory.

        Args:
            content: The text content of the memory.
            memory_type: One of "sensory", "working", "episodic",
                "semantic", or "procedural".
            tags: Optional list of categorical tags.
            source: Identifier for the agent/module that created this memory.
            metadata: Arbitrary key-value metadata.
            emotional_valence: Emotional valence [-1, 1].
            emotional_arousal: Emotional arousal [0, 1].
            importance: Manual importance score [0, 1]. Auto-computed if None.
            associations: Map of {memory_id: association_strength}.

        Returns:
            The ID of the stored memory.
        """
        mem_type = MemoryType(memory_type)
        mid = _make_id(mem_type)

        item = MemoryItem(
            id=mid,
            content=content,
            memory_type=mem_type,
            tags=tags or [],
            source=source,
            metadata=metadata or {},
            valence=emotional_valence if emotional_valence is not None else 0.0,
            arousal=emotional_arousal if emotional_arousal is not None else 0.0,
            importance=importance if importance is not None else 0.5,
            associations=associations or {},
            embedding=self._embedding.sync_embed(content),
        )

        # Route to the correct pipeline stage
        if mem_type == MemoryType.SENSORY:
            self._sensory.add(item)
        elif mem_type == MemoryType.WORKING:
            self._working.add(item)
        elif mem_type in LongTermMemory.MANAGED_TYPES:
            self._ltm.store(item)

        self._tick()
        return mid

    def recall(
        self,
        query: str | None = None,
        *,
        memory_types: str | list[str] | None = None,
        limit: int = 10,
        threshold: float = 0.1,
        include_sensory: bool = False,
        include_working: bool = True,
        include_archived: bool = False,
        mood_congruent: tuple[float, float] | None = None,
        **metadata_filters: Any,
    ) -> list[MemoryItem]:
        """Retrieve memories relevant to the query.

        Uses hybrid retrieval across all active pipeline stages,
        fusing semantic similarity, spreading activation, and
        importance scoring via the HybridRanker.

        Args:
            query: Natural language query. If None, returns recent items.
            memory_types: Filter by type(s).
            limit: Maximum number of results.
            threshold: Minimum retrieval probability threshold.
            include_sensory: Include sensory buffer in results.
            include_working: Include working memory in results.
            include_archived: Include archived (forgotten) memories.
            mood_congruent: Optional (valence, arousal) tuple to bias
                retrieval toward mood-congruent memories.
            **metadata_filters: Additional metadata key=value filters.

        Returns:
            Ranked list of memory items, sorted by relevance.
        """
        from memore.memory.enums import MemoryType as MT

        # Parse memory type filter
        type_filter = None
        if memory_types:
            if isinstance(memory_types, str):
                memory_types = [memory_types]
            type_filter = [MT(t) for t in memory_types]

        # Compute query embedding if we have a real embedding provider
        query_embedding: list[float] | None = None
        if query and not isinstance(self._embedding, NoopEmbeddingProvider):
            try:
                query_embedding = self._embedding.sync_embed(query)
            except Exception:
                query_embedding = None

        results: list[MemoryItem] = []

        # Collect from each active pipeline stage
        if include_sensory:
            sensory_results = self._sensory.search(query or "", limit=limit)
            results.extend(sensory_results)

        if include_working:
            working_items = self._working.get_context(window_size=limit)
            if query:
                working_items = [i for i in working_items if query.lower() in i.content.lower()]
            results.extend(working_items)

        # Long-term memory search (sync)
        try:
            ltm_results = self._ltm.search(
                query=query or "",
                query_embedding=query_embedding,
                memory_types=type_filter,
                limit=limit * 2,  # fetch more for re-ranking
                threshold=threshold,
                include_archived=include_archived,  # Bug #6: forward the filter
            )
        except Exception:
            ltm_results = []

        # Apply archived filter (safety net)
        if not include_archived:
            ltm_results = [
                i for i in ltm_results
                if i.consolidation_stage != ConsolidationStage.ARCHIVED
            ]

        # Merge all results
        seen_ids: set[str] = set()
        for i in results:
            seen_ids.add(i.id)
        for i in ltm_results:
            if i.id not in seen_ids:
                results.append(i)
                seen_ids.add(i.id)

        # Compute spreading activation scores from top results
        activation_scores: dict[str, float] | None = None
        if results and len(results) > 1:
            top_ids = [r.id for r in results[:min(3, len(results))]]
            try:
                activation_scores = {}
                for sid in top_ids:
                    assoc = self._backend.get_associated(sid, max_depth=2, min_strength=0.1)
                    for aid, activation in assoc.items():
                        activation_scores[aid] = max(
                            activation_scores.get(aid, 0.0), activation
                        )
            except Exception:
                activation_scores = None

        # Hybrid re-ranking
        ranked = self._ranker.rank(
            items=results,
            query=query,
            query_embedding=query_embedding,
            activation_scores=activation_scores,
            mood_congruent=mood_congruent,
        )

        # Deduplicate, touch, and auto-archive decaying memories
        final: list[MemoryItem] = []
        seen = set()
        for item, _ in ranked:
            if item.id not in seen:
                seen.add(item.id)

                # Auto-archive: if memory is below threshold, don't return it
                if self._config.auto_archive_on_recall and item.is_forgotten(self._config.forgetting_threshold):
                    # Bug #3: only archive LTM-managed types; skip working/sensory silently
                    if item.memory_type in LongTermMemory.MANAGED_TYPES:
                        item.consolidation_stage = ConsolidationStage.ARCHIVED
                        item.importance = 0.0
                        self._backend.update(item)
                    continue  # skip — this memory is too far gone

                item.touch()
                # Bug #7: persist touch for LTM-managed items
                if item.memory_type in LongTermMemory.MANAGED_TYPES:
                    self._backend.update(item)
                final.append(item)

        # Tick operation counter (triggers prune if threshold reached)
        self._tick()

        return final[:limit]

    def get_context(self, window_size: int = 7, *, include_sensory: bool = False) -> list[MemoryItem]:
        """Get the current working memory context for LLM consumption.

        Returns items ranked by attention weight. This is the data
        you would typically inject into an agent's system prompt.

        Args:
            window_size: Maximum number of items (default 7).
            include_sensory: Include recent sensory traces.

        Returns:
            Context items, highest attention first.
        """
        results = self._working.get_context(window_size=window_size)
        if include_sensory:
            sensory = self._sensory.all()
            results.extend(sensory[: max(0, window_size - len(results))])
        return results

    # ──────────────────────────────────────────────────────────────
    # Tier 2: Standard API
    # ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        memory_types: str | list[str] | None = None,
        limit: int = 20,
        **kwargs,
    ) -> list[MemoryItem]:
        """Unified search across all memory stores.

        Args:
            query: The search query.
            mode: One of "hybrid", "semantic", "associative", "keyword".
            memory_types: Filter by type(s).
            limit: Maximum results.

        Returns:
            Ranked memory items.
        """
        from memore.memory.enums import MemoryType as MT

        type_filter = None
        if memory_types:
            if isinstance(memory_types, str):
                memory_types = [memory_types]
            type_filter = [MT(t) for t in memory_types]

        try:
            return self._ltm.search(
                query=query,
                memory_types=type_filter,
                limit=limit,
                mode=mode,
                **kwargs,
            )
        except Exception:
            return []

    def get(self, memory_id: str) -> MemoryItem | None:
        """Retrieve a single memory by ID."""
        # Check pipeline stages first
        item = self._sensory.get(memory_id)
        if item is not None:
            return item
        item = self._working.get(memory_id)
        if item is not None:
            return item

        try:
            return self._backend.get(memory_id)
        except Exception:
            return None

    def forget(self, memory_id: str) -> None:
        """Explicitly archive a memory.

        Sets it to ARCHIVED stage so it's excluded from normal
        retrieval but can still be recovered with include_archived.
        """
        item = self._backend.get(memory_id)
        if item is not None:
            item.consolidation_stage = ConsolidationStage.ARCHIVED
            self._backend.update(item)

    def tag(self, memory_id: str, *tags: str) -> None:
        """Add tags to an existing memory."""
        item = self._backend.get(memory_id)
        if item is not None:
            existing = set(item.tags)
            existing.update(tags)
            item.tags = list(existing)
            self._backend.update(item)

    def tag_emotion(self, memory_id: str, valence: float, arousal: float) -> None:
        """Tag a memory with emotional dimensions.

        Args:
            memory_id: Target memory ID.
            valence: Emotional valence [-1, 1].
            arousal: Emotional arousal [0, 1].
        """
        item = self._backend.get(memory_id)
        if item is not None:
            item.valence = max(-1.0, min(1.0, valence))
            item.arousal = max(0.0, min(1.0, arousal))
            self._backend.update(item)

    # ──────────────────────────────────────────────────────────────
    # Tier 3: Advanced API
    # ──────────────────────────────────────────────────────────────

    def focus(self, content: str) -> str:
        """Add content to working memory with high attention.

        Sets the item as the current focal point (attention=1.0).

        Returns:
            The memory ID.
        """
        item = self._working.focus(content)
        return item.id

    def attend_to(self, memory_id: str, weight: float) -> None:
        """Manually adjust the attention weight of a working memory item."""
        self._working.attend_to(memory_id, weight)

    def associate(self, source_id: str, target_id: str, strength: float = 1.0) -> None:
        """Create an explicit association between two memories.

        This creates a directed edge in the association graph.
        """
        self._backend.add_association(source_id, target_id, strength)

    def retrieve_associated(
        self,
        memory_id: str,
        max_depth: int = 2,
        min_strength: float = 0.1,
        limit: int = 20,
    ) -> list[tuple[MemoryItem, float]]:
        """Spreading activation from a seed memory.

        Returns associated memories with their activation strengths.

        Args:
            memory_id: Seed memory ID.
            max_depth: Maximum hops in the association graph.
            min_strength: Minimum edge strength to traverse.
            limit: Maximum results.

        Returns:
            List of (MemoryItem, activation_strength) tuples.
        """
        try:
            associations = self._backend.get_associated(memory_id, max_depth, min_strength)
        except Exception:
            return []

        results: list[tuple[MemoryItem, float]] = []
        for aid, activation in associations.items():
            item = self._backend.get(aid)
            if item is not None:
                results.append((item, activation))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def rehearse(self, memory_id: str) -> None:
        """Rehearse a memory: resets decay clock, boosts strength.

        Mirrors the biological rehearsal effect.
        """
        item = self._backend.get(memory_id)
        if item is not None:
            item.rehearse(strength_boost=self._config.rehearsal_strength_boost)
            self._backend.update(item)

    def consolidate(self, *, force: bool = False) -> dict[str, Any]:
        """Run incremental consolidation.

        Actions:
        - Promote high-importance working items to episodic.
        - Archive memories below forgetting threshold.

        Args:
            force: Skip the interval check and consolidate immediately.

        Returns:
            A report dict with actions taken.
        """
        report: dict[str, Any] = {
            "promotions": 0,
            "archived": 0,
            "forgotten_count": 0,
        }

        # Promote important working memory items
        for item in self._working.get_context(window_size=self._config.working_memory_capacity):
            if item.importance >= self._config.promotion_importance_threshold:
                try:
                    self._ltm.promote_from_working(item)
                    report["promotions"] += 1  # Bug #9: only increment after success
                except Exception:
                    pass

        # Tag forgotten memories as archived
        try:
            decaying = self._backend.get_decaying(
                threshold=self._config.forgetting_threshold,
                limit=500,
            )
        except Exception:
            decaying = []

        # Bug #20: Exclude already-archived items from archival loop
        for item in decaying:
            if item.consolidation_stage == ConsolidationStage.ARCHIVED:
                continue
            item.consolidation_stage = ConsolidationStage.ARCHIVED
            item.importance = 0.0
            self._backend.update(item)
            report["archived"] += 1

        report["forgotten_count"] = len(decaying)
        return report

    def consolidate_sleep(self) -> ConsolidationReport:
        """Run a full sleep consolidation cycle.

        This is the advanced consolidation pipeline:
        1. Working -> Episodic promotion (high-importance items)
        2. Episodic -> Semantic abstraction (topic clustering)
        3. Procedural pattern extraction (repeated sequences)
        4. Forgetting curve archiving
        5. Strength reinforcement (frequently accessed memories)

        Returns:
            A ConsolidationReport with detailed actions taken.
        """
        try:
            working_items = self._working.get_context(
                window_size=self._config.working_memory_capacity
            )
            report = self._consolidator.run(working_items=working_items)
            return report
        except Exception:
            return ConsolidationReport(
                errors=["Could not run sleep consolidation in current event loop"]
            )

    def stats(self) -> dict[str, Any]:
        """Return system statistics."""
        try:
            backend_stats = self._backend.stats()
        except Exception:
            backend_stats = {"total": 0}
        return {
            **backend_stats,
            "sensory_buffer_size": self._sensory.size(),
            "working_memory_size": self._working.size(),
        }

    def clear(self) -> None:
        """Wipe all memories. Use with caution."""
        self._sensory.clear()
        self._working.clear()
        self._backend.clear()

    # ── Event hooks ──────────────────────────────────────────────

    def _tick(self) -> None:
        """Increment operation counter, auto-prune if threshold reached."""
        self._op_count += 1
        if self._config.auto_consolidate and self._op_count >= self._config.operations_before_prune:
            self._op_count = 0
            self._maybe_prune()

    def _maybe_prune(self) -> None:
        """Enforce max_items storage quota.

        When exceeded, archives low-importance memories until
        count drops to ``keep_at_most``.
        """
        try:
            stats = self._backend.stats()
            total = stats.get("total", 0)
            if total <= self._config.max_items:
                return

            # Fetch low-importance items and archive them
            to_archive = self._backend.list(
                sort_by="importance",
                sort_desc=False,
                limit=total,
                importance_min=0.0,
            )

            # Bug #19: exclude archived items from count so total can decrease
            non_archived = [
                i for i in to_archive
                if i.consolidation_stage != ConsolidationStage.ARCHIVED
            ]
            if len(non_archived) <= self._config.max_items:
                return

            target_archive = len(non_archived) - self._config.keep_at_most
            archived_count = 0
            for item in to_archive:
                if archived_count >= target_archive:
                    break
                if item.consolidation_stage != ConsolidationStage.ARCHIVED and item.importance <= self._config.prune_below_importance:
                    item.consolidation_stage = ConsolidationStage.ARCHIVED
                    self._backend.update(item)
                    archived_count += 1
        except Exception:
            pass

    def _on_working_evict(self, item: MemoryItem) -> None:
        """Called when an item is evicted from working memory.

        If auto-consolidation is enabled, the item may be promoted
        to episodic storage.
        """
        if not self._config.auto_consolidate:
            return
        if item.importance >= self._config.promotion_importance_threshold:
            self._ltm.promote_from_working(item)


# ── Helpers ──────────────────────────────────────────────────────

_BACKENDS_REGISTERED = False


def _ensure_backends_registered() -> None:
    """Register all built-in storage backends (idempotent)."""
    global _BACKENDS_REGISTERED
    if _BACKENDS_REGISTERED:
        return
    register_backend("in_memory", InMemoryBackend)

    # Optional backends — imported lazily
    _try_register("sqlite", "memore.storage.sqlite", "SQLiteBackend")
    _try_register("chromadb", "memore.storage.chromadb", "ChromaDBBackend")
    _try_register("qdrant", "memore.storage.qdrant", "QdrantBackend")
    _try_register("pgvector", "memore.storage.pgvector", "PGVectorBackend")

    _BACKENDS_REGISTERED = True


def _try_register(name: str, module_path: str, class_name: str) -> None:
    """Lazily import and register a backend if its dependencies are installed."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        register_backend(name, cls)
    except (ImportError, AttributeError):
        pass


def _make_id(memory_type: MemoryType) -> str:
    """Generate a type-prefixed ULID-like identifier."""
    prefix = memory_type.value[:2]
    timestamp = int(time.time() * 1000)
    rand = secrets.token_hex(8)
    return f"{prefix}_{timestamp:x}_{rand}"
