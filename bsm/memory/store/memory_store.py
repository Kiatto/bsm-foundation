"""
memory_store.py — BSM Memory Store.

Packs binary vectors into uint64 arrays for efficient Hamming distance
via POPCOUNT.  Provides insert, search (k-NN), persist, and vacuum.

Benchmark target:  N=10 000, D=256 → search < 100 ms (CPU).
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import List, Tuple, Any, Optional


class MemoryStore:
    """Binary vector store with linear-scan Hamming search.

    Each entry stores:
      - key:   D-bit binary vector packed as uint64 array
      - value: arbitrary Python object (the stored experience)
      - meta:  optional dict (timestamp, value, access_count, …)

    Search is exact (no LSH approximations).
    """

    def __init__(self, state_dim: int = 256):
        self.state_dim = state_dim
        self.n_uint64 = max(1, (state_dim + 63) // 64)
        self._keys: List[np.ndarray] = []      # list of uint64 arrays
        self._values: List[Any] = []
        self._meta: List[dict] = []

    # ------------------------------------------------------------------
    # Pack / unpack
    # ------------------------------------------------------------------

    def pack(self, binary: np.ndarray) -> np.ndarray:
        """Convert int8 {-1,+1} vector to uint64 array (packed bits)."""
        bools = (binary.ravel() > 0)
        out = np.zeros(self.n_uint64, dtype=np.uint64)
        for j, b in enumerate(bools):
            if b:
                out[j // 64] |= np.uint64(1) << np.uint64(j % 64)
        return out

    def pack_batch(self, binaries: np.ndarray) -> np.ndarray:
        """Convert (N, D) int8 array to (N, n_uint64) uint64 array.

        Uses vectorized bit-set per dimension (one pass over state_dim).
        """
        N = binaries.shape[0]
        bools = (binaries.reshape(N, -1) > 0)
        out = np.zeros((N, self.n_uint64), dtype=np.uint64)
        for j in range(self.state_dim):
            i = j // 64
            bit = j % 64
            out[:, i] |= bools[:, j].astype(np.uint64) << np.uint64(bit)
        return out

    def unpack(self, packed: np.ndarray) -> np.ndarray:
        """Unpack uint64 array back to int8 {-1,+1} vector."""
        bits = np.unpackbits(packed.view(np.uint8))[:self.state_dim]
        return np.where(bits > 0, 1, -1).astype(np.int8)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def put(self, binary: np.ndarray, value: Any, meta: Optional[dict] = None):
        """Insert one entry."""
        self._keys.append(self.pack(binary))
        self._values.append(value)
        self._meta.append(meta or {"ts": time.time()})

    def put_batch(self, binaries: np.ndarray, values: List[Any],
                  metas: Optional[List[dict]] = None):
        """Insert N entries at once (faster than loop)."""
        packed = self.pack_batch(binaries)
        if metas is None:
            metas = [{"ts": time.time()}] * len(values)
        for i in range(len(values)):
            self._keys.append(packed[i])
            self._values.append(values[i])
            self._meta.append(metas[i] if metas else {"ts": time.time()})

    def search(self, query: np.ndarray, k: int = 5
               ) -> List[Tuple[Any, int, dict]]:
        """Return k nearest neighbours as (value, hamming_dist, meta) sorted.

        Performs exact linear scan, O(N * D/64) POPCOUNT operations.
        """
        if not self._keys:
            return []
        q = self.pack(query)
        N = len(self._keys)
        dists = np.zeros(N, dtype=np.uint32)
        for i, key in enumerate(self._keys):
            xor = q ^ key
            d = 0
            for chunk in xor:
                d += int(chunk).bit_count()
            dists[i] = d
        idx = np.argpartition(dists, min(k, N) - 1)[:k]
        order = np.argsort(dists[idx])
        idx = idx[order]
        return [(self._values[i], int(dists[i]), self._meta[i]) for i in idx]

    def search_batch(self, queries: np.ndarray, k: int = 5
                     ) -> List[List[Tuple[Any, int, dict]]]:
        """Search multiple queries at once.  Returns list of results."""
        return [self.search(q, k) for q in queries]

    def size(self) -> int:
        return len(self._keys)

    def get(self, index: int) -> Tuple[np.ndarray, Any, dict]:
        """Retrieve entry by index."""
        return (self.unpack(self._keys[index]),
                self._values[index],
                self._meta[index])

    def clear(self):
        self._keys.clear()
        self._values.clear()
        self._meta.clear()

    # ------------------------------------------------------------------
    # Vacuum — remove entries by condition
    # ------------------------------------------------------------------

    def vacuum(self, keep_fn=None):
        """Remove entries where keep_fn(meta) is False.

        If keep_fn is None, does nothing (returns 0).
        """
        if keep_fn is None:
            return 0
        keep_indices = [i for i, m in enumerate(self._meta) if keep_fn(m)]
        removed = self.size() - len(keep_indices)
        self._keys = [self._keys[i] for i in keep_indices]
        self._values = [self._values[i] for i in keep_indices]
        self._meta = [self._meta[i] for i in keep_indices]
        return removed

    def vacuum_by_age(self, max_age_seconds: float):
        """Remove entries older than *max_age_seconds*."""
        now = time.time()
        return self.vacuum(lambda m: now - m.get("ts", 0) < max_age_seconds)

    def vacuum_by_count(self, max_entries: int):
        """Keep only the *max_entries* most recent entries."""
        if self.size() <= max_entries:
            return 0
        n_remove = self.size() - max_entries
        # Sort by timestamp (ascending), drop oldest
        pairs = sorted(enumerate(self._meta), key=lambda x: x[1].get("ts", 0))
        drop_indices = {p[0] for p in pairs[:n_remove]}
        keep = [i for i in range(self.size()) if i not in drop_indices]
        removed = self.size() - len(keep)
        self._keys = [self._keys[i] for i in keep]
        self._values = [self._values[i] for i in keep]
        self._meta = [self._meta[i] for i in keep]
        return removed

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save to .npz file."""
        path = str(path)
        keys_arr = np.array(self._keys) if self._keys else np.zeros((0, self.n_uint64), dtype=np.uint64)
        meta_json = json.dumps(self._meta)
        np.savez_compressed(
            path,
            keys=keys_arr,
            state_dim=self.state_dim,
            n_uint64=self.n_uint64,
            meta=meta_json,
        )
        # Save values separately as JSON lines (handles arbitrary Python objects)
        values_path = Path(path).with_suffix(".vals.jsonl")
        with open(values_path, "w") as f:
            for v in self._values:
                f.write(json.dumps({"v": v}) + "\n")

    @classmethod
    def load(cls, path: str):
        """Load from .npz file saved by .save()."""
        data = np.load(path, allow_pickle=True)
        store = cls(state_dim=int(data["state_dim"]))
        store.n_uint64 = int(data["n_uint64"])
        keys = data["keys"]
        store._keys = [keys[i] for i in range(len(keys))]
        store._meta = json.loads(str(data["meta"]))
        # Load values
        values_path = Path(path).with_suffix(".vals.jsonl")
        if values_path.exists():
            with open(values_path) as f:
                store._values = [json.loads(line)["v"] for line in f if line.strip()]
        return store

    # ------------------------------------------------------------------
    # Performance benchmark helper
    # ------------------------------------------------------------------

    def benchmark_search(self, n_queries: int = 100, k: int = 5) -> dict:
        """Run *n_queries* random queries and report timing."""
        if self.size() == 0:
            return {"error": "store is empty"}
        rng = np.random.RandomState(0)
        times = []
        for _ in range(n_queries):
            q = np.where(rng.randn(self.state_dim) > 0, 1, -1).astype(np.int8)
            t0 = time.perf_counter()
            self.search(q, k)
            times.append((time.perf_counter() - t0) * 1e6)
        times = np.array(times)
        return {
            "n_entries": self.size(),
            "state_dim": self.state_dim,
            "n_queries": n_queries,
            "k": k,
            "latency_mean_us": float(times.mean()),
            "latency_median_us": float(np.median(times)),
            "latency_p99_us": float(np.percentile(times, 99)),
            "latency_min_us": float(times.min()),
            "latency_max_us": float(times.max()),
        }

    def __repr__(self):
        return f"MemoryStore(D={self.state_dim}, entries={self.size()})"
