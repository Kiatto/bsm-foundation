"""
bsm-bench CLI — Official BSM Foundation benchmark suite.

Usage:
    bsm-bench                      # full suite
    bsm-bench --quick              # reduced suite (N=500)
    bsm-bench --report json        # output as JSON
"""

import sys, time, json, argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bsm import BSM
from bsm.metrics.engine import compute_latency_stats


def _make_dataset(name: str, n: int, D: int, rng: np.random.RandomState):
    """Generate a synthetic benchmark dataset."""
    if name == "random":
        X = np.where(rng.randn(n, D) > 0, 1, -1).astype(np.int8)
        return X, [f"item_{i}" for i in range(n)]
    elif name == "gaussian_clusters":
        # 5 clusters in {0,1}^D
        centers = np.where(rng.randn(5, D) > 0, 1, -1).astype(np.int8)
        labels = rng.randint(0, 5, size=n)
        X = np.zeros((n, D), dtype=np.int8)
        for i in range(n):
            flip = rng.rand(D) < 0.2
            X[i] = np.where(flip, -centers[labels[i]], centers[labels[i]])
        return X, [f"cluster_{l}" for l in labels]
    else:
        raise ValueError(f"Unknown dataset: {name}")


def run_benchmark(n_train: int = 2000, n_test: int = 500,
                  D: int = 256, k: int = 5, seed: int = 42) -> dict:
    """Run BSM-Bench on synthetic data and return metrics."""
    rng = np.random.RandomState(seed)

    bsm = BSM(encoder="hash", state_dim=D)

    # Build
    t0 = time.perf_counter()
    X_train, labels = _make_dataset("random", n_train, D, rng)
    build_time_ms = (time.perf_counter() - t0) * 1000

    # Observe
    t0 = time.perf_counter()
    for i in range(n_train):
        bsm.observe(X_train[i], labels[i])
    observe_time_ms = (time.perf_counter() - t0) * 1000

    # Search
    X_test, _ = _make_dataset("random", n_test, D, rng)
    latencies = []
    for i in range(n_test):
        t0 = time.perf_counter()
        bsm.recall(X_test[i], k=k)
        latencies.append((time.perf_counter() - t0) * 1e6)

    lat_stats = compute_latency_stats(np.array(latencies))

    # Memory
    n_uint64 = max(1, (D + 63) // 64)
    keys_bytes = bsm._store.size() * n_uint64 * 8
    info = bsm.info()
    health = bsm.health()

    return {
        "dataset": {"n_train": n_train, "n_test": n_test, "D": D, "k": k},
        "build_time_ms": build_time_ms,
        "observe_time_ms": observe_time_ms,
        **lat_stats,
        "entries": bsm._store.size(),
        "memory_keys_kb": keys_bytes / 1024,
        "version": info["version"],
        "utilization": health["utilization"],
        "avg_hamming_radius": health["avg_hamming_radius"],
    }


def main():
    parser = argparse.ArgumentParser(description="BSM-Bench benchmark suite")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced suite (N=500)")
    parser.add_argument("--report", choices=["json", "md", "html"],
                        default="md", help="Output format")
    args = parser.parse_args()

    N = 500 if args.quick else 2000
    print(f"\nBSM-Bench (N={N}, report={args.report})\n")

    results = run_benchmark(n_train=N, n_test=N // 4)

    if args.report == "json":
        print(json.dumps(results, indent=2))
    else:
        print(f"  Build time:      {results['build_time_ms']:.1f} ms")
        print(f"  Observe time:    {results['observe_time_ms']:.1f} ms")
        print(f"  Search p50:      {results['latency_p50_us']:.0f} µs")
        print(f"  Search p99:      {results['latency_p99_us']:.0f} µs")
        print(f"  Entries:         {results['entries']}")
        print(f"  Memory (keys):   {results['memory_keys_kb']:.1f} KB")
        print(f"  Utilization:     {results['utilization']:.1%}")
        print(f"  Avg radius:      {results['avg_hamming_radius']:.1f}")
        print()

    # Save report
    out_path = Path(f"report.{args.report}")
    if args.report == "json":
        out_path.write_text(json.dumps(results, indent=2))
    elif args.report == "md":
        md = f"""# BSM-Bench Report

| Metric | Value |
|--------|-------|
| N_train | {results['dataset']['n_train']} |
| D | {results['dataset']['D']} |
| Build time | {results['build_time_ms']:.1f} ms |
| Search p50 | {results['latency_p50_us']:.0f} µs |
| Search p99 | {results['latency_p99_us']:.0f} µs |
| Memory (keys) | {results['memory_keys_kb']:.1f} KB |
"""
        out_path.write_text(md)

    print(f"  Report saved to {out_path}\n")


if __name__ == "__main__":
    main()
