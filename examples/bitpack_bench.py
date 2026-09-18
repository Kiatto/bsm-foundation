"""
bitpack_bench.py — Benchmark comparing Standard NumPy ABM vs Bitpacked SIMD ABM.

Reports raw latency, the speedup against the reference, and — because raw latency
is not comparable across dimensions — normalized figures: per fact, per candidate
bit scanned, and against the Law IV load (pressure). Only the normalized columns
can be compared between runs at different D or M.

Read the speedup column with the caveat printed below it: the reference is frozen
and carries a known inefficiency of its own.
"""

import time
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abm import Memory as StandardMemory, capacity, predicted_accuracy
from bsm.memory.bitpack import BitpackedMemory


def benchmark(dim: int = 2048, n_facts: int = 500):
    print(f"\n========================================================")
    print(f"  ABM PERFORMANCE BENCHMARK (D={dim}, {n_facts} facts)")
    print(f"========================================================\n")

    # Generate synthetic facts
    subjects = [f"entity_{i}" for i in range(100)]
    relations = [f"rel_{j}" for j in range(20)]
    objects = [f"target_{k}" for k in range(100)]

    rng = np.random.RandomState(42)
    facts = []
    for _ in range(n_facts):
        s = rng.choice(subjects)
        r = rng.choice(relations)
        o = rng.choice(objects)
        facts.append((s, r, o))

    # ----------------------------------------------------
    # 1. Standard ABM
    # ----------------------------------------------------
    std_mem = StandardMemory(dim=dim)
    t0 = time.perf_counter()
    for s, r, o in facts:
        std_mem.store(s, r, o)
    std_store_time = (time.perf_counter() - t0) * 1000  # ms

    # Pre-populate codebook for query benchmark
    for s in subjects:
        std_mem.items.add(s)
    for o in objects:
        std_mem.items.add(o)

    n_queries = 2000
    t0 = time.perf_counter()
    for i in range(n_queries):
        s, r, o = facts[i % n_facts]
        _ = std_mem.query(s, r)
    std_query_time = (time.perf_counter() - t0) / n_queries * 1e6  # μs per query

    # Member benchmark
    t0 = time.perf_counter()
    for i in range(n_queries):
        s, r, o = facts[i % n_facts]
        _ = std_mem.member(s, r, o)
    std_member_time = (time.perf_counter() - t0) / n_queries * 1e6  # μs per member check

    # ----------------------------------------------------
    # 2. Bitpacked SIMD ABM
    # ----------------------------------------------------
    bp_mem = BitpackedMemory(dim=dim)
    t0 = time.perf_counter()
    for s, r, o in facts:
        bp_mem.store(s, r, o)
    bp_store_time = (time.perf_counter() - t0) * 1000  # ms

    for s in subjects:
        bp_mem.items.add(s)
    for o in objects:
        bp_mem.items.add(o)

    t0 = time.perf_counter()
    for i in range(n_queries):
        s, r, o = facts[i % n_facts]
        _ = bp_mem.query(s, r)
    bp_query_time = (time.perf_counter() - t0) / n_queries * 1e6  # μs per query

    t0 = time.perf_counter()
    for i in range(n_queries):
        s, r, o = facts[i % n_facts]
        _ = bp_mem.member(s, r, o)
    bp_member_time = (time.perf_counter() - t0) / n_queries * 1e6  # μs per member check

    # ----------------------------------------------------
    # Memory Footprint
    # ----------------------------------------------------
    std_vector_bytes = dim * 1  # int8 -> 1 byte per dimension
    bp_vector_bytes = (dim // 64) * 8  # uint64 -> 8 bytes per 64 bits

    print(f"RESULTS:")
    print(f"  Store Time ({n_facts} facts):")
    print(f"    Standard ABM  : {std_store_time:.2f} ms")
    print(f"    Bitpacked ABM : {bp_store_time:.2f} ms (Speedup: {std_store_time/bp_store_time:.2f}x)")

    print(f"\n  Query Latency:")
    print(f"    Standard ABM  : {std_query_time:.2f} μs / query")
    print(f"    Bitpacked ABM : {bp_query_time:.2f} μs / query (Speedup: {std_query_time/bp_query_time:.2f}x)")

    print(f"\n  Member Oracle Latency:")
    print(f"    Standard ABM  : {std_member_time:.2f} μs / check")
    print(f"    Bitpacked ABM : {bp_member_time:.2f} μs / check (Speedup: {std_member_time/bp_member_time:.2f}x)")

    print(f"\n  Memory Consumption (per vector):")
    print(f"    Standard ABM  : {std_vector_bytes} Bytes")
    print(f"    Bitpacked ABM : {bp_vector_bytes} Bytes (RAM Savings: {(1 - bp_vector_bytes/std_vector_bytes)*100:.1f}%)")

    print("\n  CAVEAT on the speedup column:")
    print("    The comparison includes a known inefficiency of the frozen reference:")
    print("    reference/abm.py constructs a np.random.RandomState per codeword")
    print("    (~193 us each). Reseeding one generator instead is bit-identical and")
    print("    would close most of the store gap. Read these numbers as 'packed ABM")
    print("    vs the reference as it stands', not as the value of bitpacking.")

    codebook = len(bp_mem.items)
    scanned_bits = codebook * dim
    n_star = capacity(dim, max(codebook, 2))
    print("\n  NORMALIZED (comparable across D and M; the raw columns are not):")
    print("    per-bit cost falls as D grows: at small D the fixed per-call overhead")
    print("    dominates, so that regime is overhead-bound rather than bandwidth-bound.")
    print(f"    store            : {bp_store_time*1000/n_facts:8.2f} us / fact"
          f"   {bp_store_time*1e6/(n_facts*dim):8.3f} ns / fact-bit")
    print(f"    query            : {bp_query_time:8.2f} us / query"
          f"  {bp_query_time*1000/scanned_bits:8.3f} ps / candidate-bit"
          f"  ({scanned_bits/bp_query_time/1e3:.1f} Gbit/s)")
    print(f"    member           : {bp_member_time:8.2f} us / check"
          f"  {bp_member_time*1000/dim:8.3f} ns / bit")
    print(f"    codebook scanned : M={codebook} x D={dim} = {scanned_bits/1e6:.2f} Mbit per query")

    print("\n  REGIME (what the latency above was measured in):")
    print(f"    pressure N/N*    : {n_facts/n_star:8.2f}   (N*={n_star:.0f} facts at this D and M)")
    print(f"    predicted acc.   : {predicted_accuracy(n_facts, dim, max(codebook, 2)):8.2f}")
    if n_facts > n_star:
        print("    NOTE: past capacity — this measures throughput, not useful answers.")
    print(f"========================================================\n")


if __name__ == "__main__":
    # Sweep D so the normalized columns can be checked against each other: raw
    # latency scales with D and M, the per-bit figures should not.
    for d in (1024, 2048, 4096):
        benchmark(dim=d, n_facts=500)
