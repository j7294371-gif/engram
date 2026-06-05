"""Performance benchmark for engram memory system.

Measures:
- Write throughput (memories/sec)
- Recall precision/recall at various dataset sizes
- Consolidation speed
- Memory usage

Run with: python examples/benchmark.py
"""

from __future__ import annotations

import sys
import time

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from engram import AgentMemory


def benchmark_write_throughput(count: int = 1000) -> dict:
    """Measure how fast we can write memories."""
    m = AgentMemory()
    start = time.monotonic()
    for i in range(count):
        m.remember(f"Benchmark memory number {i} with some filler content", memory_type="episodic")
    elapsed = time.monotonic() - start
    return {"count": count, "elapsed_s": round(elapsed, 3), "per_second": round(count / elapsed, 1)}


def benchmark_recall(dataset_size: int = 500, queries: int = 20) -> dict:
    """Measure recall speed and basic precision."""
    m = AgentMemory()
    # Populate
    topics = {
        "python": "Python is a programming language for data science and web development",
        "rust": "Rust is a systems programming language focused on safety and performance",
        "javascript": "JavaScript runs in browsers and on servers via Node.js",
        "database": "SQLite is an embedded database, PostgreSQL is a server database",
        "machine_learning": "Machine learning uses algorithms to learn patterns from data",
    }
    for i in range(dataset_size):
        topic = list(topics.keys())[i % len(topics)]
        m.remember(f"{topics[topic]} — entry {i}", memory_type="semantic", tags=[topic])

    # Query each topic
    correct = 0
    total = 0
    times = []
    for topic, expected in topics.items():
        keyword = topic  # search for "python", "rust", etc.
        start = time.monotonic()
        results = m.recall(keyword, limit=5)
        elapsed = time.monotonic() - start
        times.append(elapsed)
        total += 1
        # Check if at least one result contains the expected keyword
        if any(keyword in r.content.lower() for r in results):
            correct += 1

    return {
        "dataset_size": dataset_size,
        "queries": queries,
        "precision": f"{correct}/{total}",
        "precision_pct": round(correct / total * 100, 1),
        "avg_query_ms": round(sum(times) / len(times) * 1000, 1),
        "p95_query_ms": round(sorted(times)[int(len(times) * 0.95)] * 1000, 1),
    }


def benchmark_consolidation(count: int = 200) -> dict:
    """Measure consolidation speed."""
    m = AgentMemory()
    for i in range(count):
        m.remember(
            f"Memory {i} for consolidation testing with varied content",
            memory_type="episodic",
            importance=0.3 + (i / count) * 0.7,
        )

    # Light consolidation
    start = time.monotonic()
    report = m.consolidate()
    light_ms = (time.monotonic() - start) * 1000

    # Sleep consolidation
    start = time.monotonic()
    sleep_report = m.consolidate_sleep()
    sleep_ms = (time.monotonic() - start) * 1000

    return {
        "dataset_size": count,
        "consolidate_ms": round(light_ms, 1),
        "consolidate_sleep_ms": round(sleep_ms, 1),
        "sleep_promotions": sleep_report.promotions,
        "sleep_abstractions": sleep_report.abstractions,
        "sleep_compressed": sleep_report.compressed,
    }


def benchmark_scale() -> dict:
    """Measure write speed at different dataset sizes."""
    results = {}
    for size in [100, 500, 1000]:
        r = benchmark_write_throughput(count=size)
        results[f"{size}_writes"] = r
    return results


def main():
    print("=" * 60)
    print("  Engram Memory System — Performance Benchmark")
    print("=" * 60)

    print("\n[1/4] Write throughput (SQLite backend)")
    m = AgentMemory(backend="sqlite")
    for count in [100, 500]:
        r = benchmark_write_throughput(count=count)
        print(f"  {r['count']} writes: {r['elapsed_s']}s ({r['per_second']}/s)")

    print("\n[2/4] Write throughput (InMemory backend)")
    for count in [100, 500, 1000]:
        r = benchmark_write_throughput(count=count)
        print(f"  {r['count']} writes: {r['elapsed_s']}s ({r['per_second']}/s)")

    print("\n[3/4] Recall precision & speed")
    r = benchmark_recall(dataset_size=500)
    print(f"  Dataset:     {r['dataset_size']} memories")
    print(f"  Precision:   {r['precision_pct']}% ({r['precision']})")
    print(f"  Avg query:   {r['avg_query_ms']}ms")
    print(f"  P95 query:   {r['p95_query_ms']}ms")

    print("\n[4/4] Consolidation speed")
    r = benchmark_consolidation(count=500)
    print(f"  Dataset:           {r['dataset_size']} memories")
    print(f"  Light consolidate: {r['consolidate_ms']}ms")
    print(f"  Sleep consolidate: {r['consolidate_sleep_ms']}ms")
    print(f"  Abstractions:      {r['sleep_abstractions']}")
    print(f"  Compressed:        {r['sleep_compressed']}")

    print("\n" + "=" * 60)
    print("  Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
