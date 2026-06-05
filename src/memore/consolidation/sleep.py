"""Sleep consolidation — offline memory stabilization and abstraction.

Mimics the biological sleep cycle where memories are:
1. Replayed and strengthened
2. Promoted from working to episodic storage
3. Abstracted from episodic to semantic knowledge
4. Pattern-extracted into procedural skills
5. Forgetting-curve pruned
"""

from __future__ import annotations

import builtins
import contextlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from memore.memory.enums import ConsolidationStage, MemoryType
from memore.memory.item import MemoryItem
from memore.storage.base import StorageBackend


@dataclass
class ConsolidationReport:
    """Detailed report of what happened during a consolidation cycle."""

    promotions: builtins.int = 0          # Working → Episodic
    abstractions: builtins.int = 0        # Episodic → Semantic
    patterns_extracted: builtins.int = 0  # Episodic → Procedural
    archived: builtins.int = 0            # Below forgetting threshold
    strengthened: builtins.int = 0        # Rehearsed during sleep
    merged: builtins.int = 0              # Duplicate memories merged
    purged: builtins.int = 0              # Archived memories permanently deleted
    compressed: builtins.int = 0          # Long content truncated
    duration_ms: builtins.float = 0.0     # Wall-clock time
    errors: builtins.list[builtins.str] = field(default_factory=list)

    def __str__(self) -> str:
        parts = [
            f"promotions={self.promotions}",
            f"abstractions={self.abstractions}",
            f"patterns={self.patterns_extracted}",
            f"archived={self.archived}",
            f"merged={self.merged}",
            f"purged={self.purged}",
            f"compressed={self.compressed}",
            f"strengthened={self.strengthened}",
            f"duration={self.duration_ms:.0f}ms",
        ]
        if self.errors:
            parts.append(f"errors={len(self.errors)}")
        return f"ConsolidationReport({', '.join(parts)})"


class SleepConsolidation:
    """Full offline consolidation engine.

    Orchestrates the multi-stage memory consolidation pipeline.
    Designed to be called periodically (e.g., every hour) or
    on-demand via ``consolidate_sleep()``.
    """

    def __init__(
        self,
        backend: StorageBackend,
        promotion_importance_threshold: float = 0.7,
        abstraction_min_sources: int = 3,
        forgetting_threshold: float = 0.05,
        rehearsal_boost: float = 0.05,
    ) -> None:
        self._backend = backend
        self._promotion_threshold = promotion_importance_threshold
        self._abstraction_min_sources = abstraction_min_sources
        self._forgetting_threshold = forgetting_threshold
        self._rehearsal_boost = rehearsal_boost

    def run(
        self,
        working_items: builtins.list[MemoryItem] | None = None,
    ) -> ConsolidationReport:
        """Run the full consolidation cycle.

        Args:
            working_items: Optional list of working memory items to
                consider for promotion. If None, only long-term
                memories are processed.

        Returns:
            A ConsolidationReport with actions taken.
        """
        start = time.monotonic()
        report = ConsolidationReport()

        # Stage 1: Promote high-importance working → episodic
        if working_items:
            self._stage1_promote_working(working_items, report)

        # Stage 2: Extract semantic abstractions from episodic clusters
        self._stage2_abstract_semantic(report)

        # Stage 3: Extract procedural patterns
        self._stage3_extract_procedural(report)

        # Stage 4: Apply forgetting curve
        self._stage4_forgetting_curve(report)

        # Stage 5: Merge near-duplicate memories
        self._stage5_merge_duplicates(report)

        # Stage 6: Purge archived memories
        self._stage6_purge_archived(report)

        # Stage 7: Compress very old memory content
        self._stage7_compress_content(report)

        # Stage 8: Strengthen frequently accessed memories
        self._stage8_strengthen(report)

        report.duration_ms = (time.monotonic() - start) * 1000
        return report

    # ── Stage 1: Working → Episodic promotion ───────────────────

    def _stage1_promote_working(
        self,
        working_items: builtins.list[MemoryItem],
        report: ConsolidationReport,
    ) -> None:
        """Promote high-importance working memory items to episodic."""
        for item in working_items:
            if item.importance >= self._promotion_threshold:
                try:
                    # Create an episodic copy
                    import secrets
                    new_id = f"ep_{int(time.time() * 1000):x}_{secrets.token_hex(6)}"
                    episodic = MemoryItem(
                        id=new_id,
                        content=item.content,
                        memory_type=MemoryType.EPISODIC,
                        memory_subtype=item.memory_subtype,
                        tags=list(item.tags),
                        metadata=dict(item.metadata),
                        source=item.source,
                        strength=min(1.0, item.strength * 1.05),  # slight consolidation boost
                        importance=item.importance,
                        valence=item.valence,
                        arousal=item.arousal,
                        consolidation_stage=ConsolidationStage.WORKING_PROMOTED,
                        promoted_from=item.id,
                    )
                    self._backend.store(episodic)
                    report.promotions += 1
                except Exception as e:
                    report.errors.append(f"promote({item.id}): {e}")

    # ── Stage 2: Episodic → Semantic abstraction ────────────────

    def _stage2_abstract_semantic(self, report: ConsolidationReport) -> None:
        """Find clusters of similar episodic memories and extract
        semantic abstractions."""
        try:
            # Get recent episodic memories with high importance
            episodic = self._backend.list(
                memory_types=[MemoryType.EPISODIC],
                importance_min=0.5,
                limit=100,
                sort_by="created_at",
                sort_desc=True,
            )

            if len(episodic) < self._abstraction_min_sources:
                return

            # Simple keyword-based clustering
            clusters = self._cluster_by_topic(episodic)

            for topic, members in clusters.items():
                if len(members) < self._abstraction_min_sources:
                    continue
                if not topic.strip():
                    continue

                try:
                    # Create a semantic abstraction from the cluster
                    source_ids = [m.id for m in members]

                    # Synthesize abstract content
                    common_tags = self._common_tags(members)
                    avg_importance = sum(m.importance for m in members) / len(members)
                    avg_valence = sum(m.valence for m in members) / len(members)
                    avg_arousal = sum(m.arousal for m in members) / len(members)

                    abstract_text = self._synthesize_abstraction(topic, members)

                    import secrets
                    sem_id = f"sem_{int(time.time() * 1000):x}_{secrets.token_hex(6)}"
                    semantic = MemoryItem(
                        id=sem_id,
                        content=abstract_text,
                        memory_type=MemoryType.SEMANTIC,
                        tags=list(common_tags),
                        source="consolidation",
                        strength=0.7,
                        importance=min(1.0, avg_importance * 1.1),
                        valence=avg_valence,
                        arousal=avg_arousal,
                        consolidation_stage=ConsolidationStage.SEMANTIC_EXTRACTED,
                        abstraction_source_ids=source_ids,
                    )
                    self._backend.store(semantic)
                    report.abstractions += 1

                    # Tag source items as semantically extracted
                    for m in members:
                        if m.consolidation_stage in (ConsolidationStage.RAW, ConsolidationStage.WORKING_PROMOTED):
                            m.consolidation_stage = ConsolidationStage.SEMANTIC_EXTRACTED
                            with contextlib.suppress(Exception):
                                self._backend.update(m)
                except Exception as e:
                    report.errors.append(f"abstract({topic}): {e}")

        except Exception as e:
            report.errors.append(f"stage2: {e}")

    # ── Stage 3: Procedural pattern extraction ──────────────────

    def _stage3_extract_procedural(self, report: ConsolidationReport) -> None:
        """Detect repeated patterns in episodic memories and
        extract procedural knowledge."""
        try:
            episodic = self._backend.list(
                memory_types=[MemoryType.EPISODIC],
                importance_min=0.3,
                limit=200,
            )

            if len(episodic) < 2:
                return

            # Detect patterns: repeated n-grams in content
            patterns = self._detect_procedural_patterns(episodic)

            for pattern_text, sources in patterns.items():
                if not pattern_text.strip():
                    continue

                try:
                    import secrets
                    proc_id = f"pr_{int(time.time() * 1000):x}_{secrets.token_hex(6)}"
                    procedural = MemoryItem(
                        id=proc_id,
                        content=pattern_text,
                        memory_type=MemoryType.PROCEDURAL,
                        tags=["pattern", "skill"],
                        source="consolidation",
                        strength=0.8,
                        importance=0.6,
                        consolidation_stage=ConsolidationStage.PROCEDURAL_EXTRACTED,
                        abstraction_source_ids=[s.id for s in sources],
                    )
                    self._backend.store(procedural)
                    report.patterns_extracted += 1
                except Exception as e:
                    report.errors.append(f"pattern extract: {e}")

        except Exception as e:
            report.errors.append(f"stage3: {e}")

    # ── Stage 4: Forgetting curve ───────────────────────────────

    def _stage4_forgetting_curve(self, report: ConsolidationReport) -> None:
        """Archive memories that have decayed below the forgetting
        threshold."""
        try:
            decaying = self._backend.get_decaying(
                threshold=self._forgetting_threshold,
                limit=500,
            )
            for item in decaying:
                item.consolidation_stage = ConsolidationStage.ARCHIVED
                item.importance = max(0.01, item.importance * 0.3)  # reduce but don't zero
                try:
                    self._backend.update(item)
                    report.archived += 1
                except Exception as e:
                    report.errors.append(f"archive({item.id}): {e}")
        except Exception as e:
            report.errors.append(f"stage4: {e}")

    # ── Stage 5: Merge near-duplicate memories ──────────────────

    def _stage5_merge_duplicates(self, report: ConsolidationReport) -> None:
        """Find and merge near-duplicate memories.

        If two episodic memories have very similar content, keep
        only the one with higher importance (compression ratio N→1).
        """
        try:
            items = self._backend.list(
                memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
                limit=200,
                sort_by="created_at",
                sort_desc=True,
            )

            seen_hashes: dict[str, list[MemoryItem]] = {}
            for item in items:
                # Create a simple content fingerprint: first 50 chars + keywords
                fingerprint = item.content[:50].lower().strip()
                if fingerprint not in seen_hashes:
                    seen_hashes[fingerprint] = []
                seen_hashes[fingerprint].append(item)

            for _fingerprint, duplicates in seen_hashes.items():
                if len(duplicates) < 2:
                    continue
                # Keep the one with highest importance
                duplicates.sort(key=lambda i: i.importance, reverse=True)
                keeper = duplicates[0]
                for dup in duplicates[1:]:
                    # Merge tags and metadata before deleting
                    merged_tags = list(set(keeper.tags) | set(dup.tags))
                    keeper.tags = merged_tags
                    keeper.metadata.update(dup.metadata)
                    self._backend.update(keeper)
                    # Transfer associations from duplicate to keeper
                    assoc = self._backend.get_associated(dup.id, max_depth=1)
                    for source_id, strength in assoc.items():
                        self._backend.add_association(keeper.id, source_id,
                                                             max(strength, 0.5))
                    # Delete the duplicate
                    self._backend.delete(dup.id)
                    report.merged += 1
        except Exception as e:
            report.errors.append(f"stage5(merge): {e}")

    # ── Stage 6: Purge archived memories ───────────────────────

    def _stage6_purge_archived(self, report: ConsolidationReport) -> None:
        """Permanently delete archived memories that haven't been
        accessed in a long time.

        Archived memories are "forgotten" — after 7+ days without
        access, they're permanently removed to free storage.
        """
        try:
            archived = self._backend.list(
                memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
                limit=500,
                sort_by="last_accessed_at",
                sort_desc=False,
            )
            from datetime import datetime, timedelta, timezone
            cutoff = datetime.now(timezone.utc) - timedelta(days=7)

            for item in archived:
                if item.consolidation_stage == ConsolidationStage.ARCHIVED and item.last_accessed_at < cutoff:
                        self._backend.delete(item.id)
                        report.purged += 1
        except Exception as e:
            report.errors.append(f"stage6(purge): {e}")

    # ── Stage 7: Compress old memory content ───────────────────

    def _stage7_compress_content(self, report: ConsolidationReport) -> None:
        """Truncate very long content from old, low-importance memories
        to save storage space."""
        try:
            items = self._backend.list(
                limit=200,
                sort_by="created_at",
                sort_desc=False,
            )
            for item in items:
                if item.importance < 0.3 and len(item.content) > 200:
                    item.content = item.content[:150] + "... [compressed]"
                    try:
                        self._backend.update(item)
                        report.compressed += 1
                    except Exception:
                        pass
        except Exception as e:
            report.errors.append(f"stage7(compress): {e}")

    # ── Stage 8: Strengthen ─────────────────────────────────────

    def _stage8_strengthen(self, report: ConsolidationReport) -> None:
        """Strengthen frequently accessed memories via simulated
        rehearsal during sleep."""
        try:
            # Get top-accessed memories
            frequent = self._backend.list(
                limit=50, sort_by="access_count", sort_desc=True
            )
            for item in frequent:
                if item.access_count > 3 and item.strength < 0.95:
                    item.strength = min(1.0, item.strength + self._rehearsal_boost)
                    item.last_rehearsed_at = datetime.now(timezone.utc)
                    try:
                        self._backend.update(item)
                        report.strengthened += 1
                    except Exception:
                        pass
        except Exception as e:
            report.errors.append(f"stage8: {e}")

    # ── Helper methods ──────────────────────────────────────────

    def _cluster_by_topic(self, items: builtins.list[MemoryItem]) -> dict[str, builtins.list[MemoryItem]]:
        """Simple keyword-overlap clustering.

        Groups items that share significant keyword overlap.
        Returns a dict of {topic_keyword: [items]}.
        """
        clusters: dict[str, builtins.list[MemoryItem]] = {}
        for item in items:
            # Extract meaningful keywords (words > 4 chars)
            words = set(
                w.lower().rstrip(".,!?;:")
                for w in item.content.split()
                if len(w) > 4
            )
            for tag in item.tags:
                words.add(tag.lower())

            for word in words:
                if word not in clusters:
                    clusters[word] = []
                clusters[word].append(item)

        return clusters

    def _common_tags(self, items: builtins.list[MemoryItem]) -> set[str]:
        """Find tags common to a majority of items."""
        if not items:
            return set()
        tag_counts: dict[str, int] = {}
        for item in items:
            for tag in item.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        threshold = len(items) // 2
        return {tag for tag, count in tag_counts.items() if count >= threshold}

    def _synthesize_abstraction(self, topic: str, members: builtins.list[MemoryItem]) -> str:
        """Synthesize an abstracted semantic fact from episodic memories.

        Creates a concise, generalized statement about the topic
        derived from multiple episodic instances.
        """
        # Collect unique statements about this topic
        statements = []
        for m in members:
            content_lower = m.content.lower()
            if topic.lower() in content_lower and len(m.content) < 200:
                statements.append(m.content)

        if not statements:
            # Fallback: generic abstraction
            return (
                f"Based on {len(members)} related experiences: "
                f"the topic '{topic}' appears frequently in memory."
            )

        # Create combined abstraction
        return (
            f"Synthesized from {len(statements)} related memories: "
            f"common patterns around '{topic}' include: "
            + "; ".join(statements[:5])
        )

    def _detect_procedural_patterns(
        self, items: builtins.list[MemoryItem]
    ) -> dict[str, builtins.list[MemoryItem]]:
        """Detect repeated action patterns across episodic memories.

        Looks for sequences like "X then Y" or repeated verb patterns.
        """
        patterns: dict[str, builtins.list[MemoryItem]] = {}

        # Simple approach: find common phrases across items
        phrase_map: dict[str, builtins.list[MemoryItem]] = {}
        for item in items:
            words = item.content.lower().split()
            # Extract 2-3 word phrases
            for i in range(len(words) - 1):
                for n in range(2, min(4, len(words) - i + 1)):
                    phrase = " ".join(words[i : i + n])
                    if len(phrase) > 10:  # meaningful phrase length
                        if phrase not in phrase_map:
                            phrase_map[phrase] = []
                        phrase_map[phrase].append(item)

        # Keep phrases that appear multiple times
        for phrase, sources in phrase_map.items():
            if len(sources) >= 2:
                patterns[f"Pattern: {phrase}"] = sources

        return patterns
