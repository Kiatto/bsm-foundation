"""
compare_with_faiss.py — BSM MemoryStore vs FAISS flat (L2) benchmark.

Compares:
  - Build time (indexing N vectors)
  - Search latency (p50, p99)
  - Memory footprint
  - Recall@1 (does FAISS find the same nearest neighbour as exact Hamming?)

Usage:
    python compare_with_faiss.py [--N 10000] [--D 256]

Requires: pip install faiss-cpu (or faiss)
"""

import sys, time, textwrap, argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bsm.memory.store.memory_store import MemoryStore

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def benchmark(N: int = 10_000, D: int = 256, k: int = 5, n_queries: int = 500):
    print(f"\n{'=' * 60}")
    print(f"BSM Store vs FAISS flat benchmark")
    print(f"{'=' * 60}")
    print(f"  N={N}, D={D}, k={k}, queries={n_queries}")
    print()

    # Generate data
    rng = np.random.RandomState(42)
    X = np.where(rng.randn(N, D) > 0, 1, -1).astype(np.int8)
    queries = np.where(rng.randn(n_queries, D) > 0, 1, -1).astype(np.int8)
    values = [f"item_{i}" for i in range(N)]

    # ---- BSM Store ----
    print("--- BSM MemoryStore ---")
    store = MemoryStore(state_dim=D)

    t0 = time.perf_counter()
    store.put_batch(X, values)
    build_time = (time.perf_counter() - t0) * 1000
    print(f"  Build: {build_time:.1f} ms")

    bsm_latencies = []
    for q in queries:
        t0 = time.perf_counter()
        store.search(q, k=k)
        bsm_latencies.append((time.perf_counter() - t0) * 1e6)
    bsm_lat = np.array(bsm_latencies)
    print(f"  Search: mean={bsm_lat.mean():.0f} µs, "
          f"p50={np.median(bsm_lat):.0f} µs, "
          f"p99={np.percentile(bsm_lat, 99):.0f} µs")

    # Estimate memory
    store_bytes = store.size() * store.n_uint64 * 8  # uint64
    val_bytes = sum(len(str(v)) for v in store._values)
    print(f"  Memory: ~{store_bytes/1024:.1f} KB (keys) + {val_bytes/1024:.1f} KB (values)")

    # ---- FAISS ----
    bsm_vs_faiss = {}

    if FAISS_AVAILABLE:
        print("\n--- FAISS (IndexFlatL2) ---")
        # FAISS needs float32; convert -1→0, 1→1 for L2 (or use binary index)
        X_f = X.astype(np.float32)
        q_f = queries.astype(np.float32)

        t0 = time.perf_counter()
        index = faiss.IndexFlatL2(D)
        index.add(X_f)
        faiss_build = (time.perf_counter() - t0) * 1000
        print(f"  Build: {faiss_build:.1f} ms")

        faiss_latencies = []
        for q in q_f:
            t0 = time.perf_counter()
            index.search(q.reshape(1, -1), k)
            faiss_latencies.append((time.perf_counter() - t0) * 1e6)
        faiss_lat = np.array(faiss_latencies)
        print(f"  Search: mean={faiss_lat.mean():.0f} µs, "
              f"p50={np.median(faiss_lat):.0f} µs, "
              f"p99={np.percentile(faiss_lat, 99):.0f} µs")

        # Recall@1: for each query, does FAISS top-1 match BSM top-1?
        matches = []
        for i, q in enumerate(queries):
            bsm_top = store.search(q, k=1)[0][1]  # distance
            faiss_dist, faiss_idx = index.search(q_f[i].reshape(1, -1), 1)
            # Convert FAISS L2 distance to approximate Hamming
            # (not directly comparable, but we can check if same item is #1)
            bsm_idx = store.search(q, k=N)  # get all
            bsm_top_value = bsm_idx[0][0]
            faiss_top_value = values[faiss_idx[0, 0]]
            matches.append(bsm_top_value == faiss_top_value)
        recall = np.mean(matches)
        print(f"  Recall@1 (BSM=FAISS top-1): {recall:.1%}")
        bsm_vs_faiss["recall@1"] = recall
        bsm_vs_faiss["faiss_mean_us"] = float(faiss_lat.mean())
        bsm_vs_faiss["faiss_p50_us"] = float(np.median(faiss_lat))
        bsm_vs_faiss["faiss_p99_us"] = float(np.percentile(faiss_lat, 99))

        # FAISS memory
        faiss_bytes = index.ntotal * D * 4  # float32
        print(f"  Memory: ~{faiss_bytes/1024:.1f} KB (vectors only)")

    # ---- Comparison summary ----
    print(f"\n{'=' * 60}")
    print("Comparison Summary")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<30} {'BSM Store':<16} {'FAISS Flat':<16}")
    print(f"  {'─' * 30} {'─' * 16} {'─' * 16}")
    print(f"  {'Build time (ms)':<30} {build_time:<16.1f} "
          f"{faiss_build if FAISS_AVAILABLE else 'N/A':<16}")
    print(f"  {'Search p50 (µs)':<30} {np.median(bsm_lat):<16.0f} "
          f"{bsm_vs_faiss.get('faiss_p50_us', -1):<16.0f}")
    print(f"  {'Search p99 (µs)':<30} {np.percentile(bsm_lat, 99):<16.0f} "
          f"{bsm_vs_faiss.get('faiss_p99_us', -1):<16.0f}")
    print(f"  {'Memory (KB)':<30} {(store_bytes+val_bytes)/1024:<16.1f} "
          f"{faiss_bytes/1024 if FAISS_AVAILABLE else 0:<16.1f}")

    store_metrics = {
        "N": N, "D": D, "k": k,
        "build_ms": build_time,
        "bsm_mean_us": float(bsm_lat.mean()),
        "bsm_p50_us": float(np.median(bsm_lat)),
        "bsm_p99_us": float(np.percentile(bsm_lat, 99)),
        "memory_kb": (store_bytes + val_bytes) / 1024,
    }
    if FAISS_AVAILABLE:
        store_metrics["faiss_build_ms"] = faiss_build
        store_metrics.update(bsm_vs_faiss)

    return store_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=10_000)
    parser.add_argument("--D", type=int, default=256)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--queries", type=int, default=500)
    args = parser.parse_args()

    if not FAISS_AVAILABLE:
        print("⚠ FAISS not installed.  Install with: pip install faiss-cpu")
        print("  Running BSM store benchmark only.\n")

    benchmark(N=args.N, D=args.D, k=args.k, n_queries=args.queries)
