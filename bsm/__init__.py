"""
BSM Foundation — Geometric Memory Platform.

Single entry point:

    from bsm import BSM

    bsm = BSM(encoder="hash", state_dim=256)
    state = bsm.encode("hello")
    bsm.observe(state, {"msg": "hello"})
    results = bsm.recall(state)
    bsm.info()
    bsm.health()
"""

import time
import numpy as np
from collections import Counter

from bsm._version import __version__
from bsm.memory.encoder.bsm_encoder import HashEncoder, ProjectionEncoder, LearnedEncoder
from bsm.memory.store.memory_store import MemoryStore
from bsm.memory.router.bsm_router import BSMRouter

__all__ = ["BSM", "__version__"]


class BSM:
    """Unified entry point for the BSM Foundation memory platform.

    Usage:
        bsm = BSM(encoder="hash", state_dim=256)
        state = bsm.encode("text or embeddings")
        bsm.observe(state, payload)
        results = bsm.recall(state, k=5)
        prediction = bsm.predict(state)
        route, dist = bsm.route(state)
        bsm.info()
        bsm.health()
        bsm.save("memory.bsm-store.npz")
    """

    def __init__(self, encoder: str = "hash", state_dim: int = 256,
                 capacity: int = 100_000):
        self.state_dim = state_dim
        self.capacity = capacity

        # Internal encoder
        self._encoder = self._resolve_encoder(encoder, state_dim)

        # Universal projection matrix (for non-text inputs)
        self._projection = None

        # Core subsystems
        self._store = MemoryStore(state_dim=state_dim)
        self._router = BSMRouter(state_dim=state_dim)

        # Lifecycle
        self._creation_ts = time.time()
        self._observe_count = 0
        self._recall_count = 0

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def _resolve_encoder(self, spec, state_dim):
        if isinstance(spec, str):
            registry = {
                "hash": HashEncoder,
                "projection": ProjectionEncoder,
                "learned": LearnedEncoder,
            }
            cls = registry.get(spec.lower())
            if cls is None:
                raise ValueError(f"Unknown encoder '{spec}'. "
                                 f"Choose from: {list(registry.keys())}")
            if spec.lower() == "projection":
                return cls(state_dim=state_dim)
            return cls(state_dim=state_dim)
        return spec  # assume it's already an encoder instance

    def encode(self, data):
        """Convert data to binary state vector(s).

        Accepts:
            str              → uses the internal encoder
            list[str]        → batch encode via internal encoder
            np.ndarray       → applies random projection to D dimensions + sign
        """
        if isinstance(data, str):
            return self._encoder.encode(data)
        if isinstance(data, (list, tuple)) and all(isinstance(x, str) for x in data):
            return self._encoder.encode(data)
        arr = np.asarray(data, dtype=np.float32)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr.reshape(1, -1)
        if self._projection is None or self._projection.shape[0] != arr.shape[1]:
            rng = np.random.RandomState(42)
            self._projection = rng.randn(arr.shape[1], self.state_dim).astype(np.float32)
        raw = arr @ self._projection
        binary = np.where(raw >= 0, 1, -1).astype(np.int8)
        return binary[0] if squeeze else binary

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------

    def observe(self, state, payload, value: float = 1.0):
        """Store a (state, payload) pair."""
        self._store.put(state, payload, meta={"value": value, "ts": time.time()})
        self._observe_count += 1

    def recall(self, state, k: int = 5):
        """Retrieve k nearest neighbours by Hamming distance."""
        self._recall_count += 1
        return self._store.search(state, k=k)

    def predict(self, state):
        """Predict payload via majority vote among nearest neighbours."""
        results = self.recall(state, k=5)
        if not results:
            return None
        votes = Counter()
        for payload, dist, meta in results:
            votes[str(payload)] += 1
        return votes.most_common(1)[0][0]

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, state):
        """Classify state into a named route."""
        return self._router.route(state)

    def add_route(self, name, examples):
        """Build a prototype from examples and register a route."""
        if isinstance(examples, list) and all(isinstance(e, str) for e in examples):
            encodings = np.array([self._encoder.encode(e) for e in examples])
        else:
            encodings = np.asarray(examples)
        prototype = np.where(encodings.mean(axis=0) >= 0, 1, -1).astype(np.int8)
        self._router.add_route(name, prototype)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def sleep(self, forget_threshold: float = 0.3, max_age: float = 3600):
        """Consolidate memory: forget old/low-value entries."""
        n_before = self._store.size()
        self._store.vacuum_by_age(max_age)
        self._store.vacuum(lambda m: m.get("value", 1.0) >= forget_threshold)
        return n_before - self._store.size()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Persist memory store to disk."""
        self._store.save(path)

    def load(self, path: str):
        """Load memory store from disk."""
        self._store = MemoryStore.load(path)
        self.state_dim = self._store.state_dim
        return self

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def info(self) -> dict:
        """Return version and configuration."""
        return {
            "version": __version__,
            "encoder": getattr(self._encoder, "_name", "custom"),
            "state_dim": self.state_dim,
            "capacity": self.capacity,
            "entries": self._store.size(),
            "protocol": "1.0",
            "uptime_s": time.time() - self._creation_ts,
        }

    def health(self) -> dict:
        """Return memory health metrics."""
        n = self._store.size()
        if n == 0:
            return {"entries": 0, "status": "empty"}

        values = [m.get("value", 1.0) for m in self._store._meta]
        low_value_ratio = sum(1 for v in values if v < 0.3) / n

        avg_radius = 0.0
        if n >= 2:
            N = min(100, n)
            keys = self._store._keys[:N]
            dists = []
            for i in range(N):
                best = self.state_dim + 1
                for j in range(N):
                    if i == j:
                        continue
                    d = sum(int((keys[i] ^ keys[j])[c]).bit_count()
                            for c in range(self._store.n_uint64))
                    if d < best:
                        best = d
                if best <= self.state_dim:
                    dists.append(best)
            avg_radius = float(np.mean(dists)) if dists else 0.0

        entries = n
        util = entries / self.capacity

        return {
            "entries": entries,
            "capacity": self.capacity,
            "utilization": util,
            "low_value_ratio": low_value_ratio,
            "avg_hamming_radius": avg_radius,
            "sleep_suggested": low_value_ratio > 0.3 or util > 0.9,
        }

    def metrics(self, n_queries: int = 100) -> dict:
        """Run performance benchmark and return metrics."""
        return self._store.benchmark_search(n_queries=n_queries)

    def __repr__(self):
        return (f"BSM(version={__version__}, "
                f"encoder={getattr(self._encoder, '_name', 'custom')}, "
                f"D={self.state_dim}, entries={self._store.size()})")
