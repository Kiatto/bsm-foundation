"""
test_store.py — BSM Memory Store tests.

Tests: put, search, batch, persistence, vacuum, benchmark (10K < 100 ms).
"""

import sys
import time
import numpy as np
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bsm.memory.store.memory_store import MemoryStore

D = 256


def _random_binary(n=1):
    return np.where(np.random.randn(n, D) > 0, 1, -1).astype(np.int8)


def test_put_and_search():
    store = MemoryStore(state_dim=D)
    v1 = _random_binary()[0]
    store.put(v1, "hello")
    assert store.size() == 1
    results = store.search(v1, k=1)
    assert len(results) == 1
    assert results[0][0] == "hello"
    assert results[0][1] == 0  # exact match → distance 0
    print(f"  ✓ put & search: exact match distance={results[0][1]}")


def test_search_ranking():
    store = MemoryStore(state_dim=D)
    rng = np.random.RandomState(0)
    q = np.where(rng.randn(D) > 0, 1, -1).astype(np.int8)
    for i in range(10):
        v = np.where(rng.randn(D) > 0, 1, -1).astype(np.int8)
        store.put(v, f"item_{i}")
    results = store.search(q, k=3)
    assert len(results) == 3
    # Distances should be sorted ascending
    dists = [r[1] for r in results]
    assert all(dists[i] <= dists[i + 1] for i in range(len(dists) - 1)), "not sorted"
    print(f"  ✓ search ranking: dists={dists}")


def test_batch_put():
    store = MemoryStore(state_dim=D)
    N = 50
    binaries = _random_binary(N)
    values = [f"item_{i}" for i in range(N)]
    store.put_batch(binaries, values)
    assert store.size() == N
    print(f"  ✓ batch put: {N} entries")


def test_persistence():
    store = MemoryStore(state_dim=D)
    binaries = _random_binary(10)
    values = [f"persist_{i}" for i in range(10)]
    store.put_batch(binaries, values)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    store.save(path)
    loaded = MemoryStore.load(path)
    assert loaded.size() == 10
    # Check a search matches
    q = binaries[0]
    orig = store.search(q, k=1)
    loaded_r = loaded.search(q, k=1)
    assert orig[0][0] == loaded_r[0][0]
    assert orig[0][1] == loaded_r[0][1]
    print(f"  ✓ persistence: save/load roundtrip OK")
    Path(path).unlink(missing_ok=True)
    Path(path).with_suffix(".vals.jsonl").unlink(missing_ok=True)


def test_vacuum():
    store = MemoryStore(state_dim=D)
    rng = np.random.RandomState(1)
    t0 = time.time()
    for i in range(20):
        v = np.where(rng.randn(D) > 0, 1, -1).astype(np.int8)
        store.put(v, f"item_{i}",
                  meta={"ts": t0 + i * 10, "keep": i % 2 == 0})
    assert store.size() == 20
    removed = store.vacuum(lambda m: m["keep"])
    assert removed == 10
    assert store.size() == 10
    print(f"  ✓ vacuum: removed={removed}, remaining={store.size()}")


def test_vacuum_by_age():
    store = MemoryStore(state_dim=D)
    rng = np.random.RandomState(2)
    now = time.time()
    for i in range(10):
        v = np.where(rng.randn(D) > 0, 1, -1).astype(np.int8)
        ts = now - 7200 if i < 5 else now - 10  # first 5 are > 1 hour old
        store.put(v, f"item_{i}", meta={"ts": ts})
    assert store.size() == 10
    removed = store.vacuum_by_age(max_age_seconds=3600)
    assert removed == 5, f"expected 5 removed, got {removed}"
    assert store.size() == 5
    print(f"  ✓ vacuum_by_age: removed {removed}")


def test_benchmark_10k():
    """Target: 10K entries, search < 100 ms."""
    store = MemoryStore(state_dim=D)
    N = 10_000
    rng = np.random.RandomState(3)
    binaries = np.where(rng.randn(N, D) > 0, 1, -1).astype(np.int8)
    values = [f"item_{i}" for i in range(N)]
    t0 = time.perf_counter()
    store.put_batch(binaries, values)
    insert_time = (time.perf_counter() - t0) * 1000
    print(f"  insert {N} entries: {insert_time:.1f} ms")

    q = binaries[0]
    t0 = time.perf_counter()
    results = store.search(q, k=5)
    search_time = (time.perf_counter() - t0) * 1000
    assert len(results) == 5
    assert results[0][1] == 0  # exact match is first
    print(f"  benchmark: search {N} entries: {search_time:.1f} ms")
    assert search_time < 100, f"Search took {search_time:.1f} ms, expected < 100 ms"
    print(f"  ✓ benchmark: 10K entries search < 100 ms")


if __name__ == "__main__":
    print("\n=== BSM Memory Store Tests ===\n")
    test_put_and_search()
    test_search_ranking()
    test_batch_put()
    test_persistence()
    test_vacuum()
    test_vacuum_by_age()
    test_benchmark_10k()
    print("\n✓ All store tests passed\n")
