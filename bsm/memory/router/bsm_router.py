"""
bsm_router.py — BSM Router: route queries by binary vector similarity.

Maintains one prototype vector per route.  A query is routed to the nearest
prototype (k=1 in Hamming space).  Routes can be added or updated online.

Success metric:  accuracy > 80 % on weather-vs-math routing (D=256).
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional


class BSMRouter:
    """k-NN router in binary Hamming space (k=1).

    Each route has a prototype binary vector.  The router:
      1. Encodes a query to a binary vector.
      2. Finds the nearest prototype by Hamming distance.
      3. Returns the route name and distance.

    Online: can add, update, or remove routes at any time.
    """

    def __init__(self, state_dim: int = 256):
        self.state_dim = state_dim
        self._prototypes: Dict[str, np.ndarray] = {}   # name → int8 vector
        self._n_updates = 0

    # ------------------------------------------------------------------
    # Route management
    # ------------------------------------------------------------------

    def add_route(self, name: str, prototype: np.ndarray):
        """Add or update a route prototype."""
        if prototype.shape != (self.state_dim,):
            raise ValueError(f"prototype must be ({self.state_dim},), got {prototype.shape}")
        self._prototypes[name] = prototype.astype(np.int8).copy()
        self._n_updates += 1

    def remove_route(self, name: str):
        self._prototypes.pop(name, None)
        self._n_updates += 1

    def set_prototypes(self, prototypes: Dict[str, np.ndarray]):
        """Replace all prototypes at once."""
        self._prototypes = {}
        for name, proto in prototypes.items():
            self.add_route(name, proto)
        self._n_updates += 1

    def get_routes(self) -> List[str]:
        return list(self._prototypes.keys())

    def get_prototype(self, name: str) -> Optional[np.ndarray]:
        return self._prototypes.get(name)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, binary: np.ndarray) -> Tuple[str, int]:
        """Return (best_route_name, hamming_distance).

        Args:
            binary: (D,) int8 array in {-1, +1}

        Returns:
            (name, distance).  If no routes registered, returns ("_none_", -1).
        """
        if not self._prototypes:
            return ("_none_", -1)
        best_name = None
        best_dist = self.state_dim + 1
        for name, proto in self._prototypes.items():
            d = int(np.sum(binary != proto))
            if d < best_dist:
                best_dist = d
                best_name = name
        return (best_name, best_dist)

    def route_batch(self, binaries: np.ndarray) -> List[Tuple[str, int]]:
        """Route multiple queries at once."""
        return [self.route(b) for b in binaries]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, queries: np.ndarray, labels: List[str]
                 ) -> Dict[str, float]:
        """Evaluate routing accuracy.

        Args:
            queries: (N, D) int8 array of binary vectors
            labels:  length-N list of correct route names

        Returns:
            dict with accuracy, per-route breakdown, confusion, latency.
        """
        N = len(labels)
        correct = 0
        per_route = {name: {"correct": 0, "total": 0} for name in self._prototypes}
        per_route["_unknown_"] = {"correct": 0, "total": 0}
        confusion = {}

        t0 = time.perf_counter()
        for i in range(N):
            q = queries[i]
            label = labels[i]
            pred, dist = self.route(q)
            if pred == label:
                correct += 1
                if label in per_route:
                    per_route[label]["correct"] += 1
            else:
                key = f"{label}→{pred}"
                confusion[key] = confusion.get(key, 0) + 1
            if label in per_route:
                per_route[label]["total"] += 1
            else:
                per_route["_unknown_"]["total"] += 1
        latency_ms = (time.perf_counter() - t0) * 1000.0

        per_route_accuracy = {}
        for name, v in per_route.items():
            if v["total"] > 0:
                per_route_accuracy[name] = v["correct"] / v["total"]
            else:
                per_route_accuracy[name] = 0.0

        return {
            "accuracy": correct / N if N > 0 else 0.0,
            "correct": correct,
            "total": N,
            "per_route_accuracy": per_route_accuracy,
            "confusion": confusion,
            "latency_ms": latency_ms,
            "latency_per_query_us": latency_ms * 1000.0 / N if N > 0 else 0.0,
        }

    def build_prototypes(self, encodings: Dict[str, np.ndarray]):
        """Build prototypes as centroids of per-route encodings.

        Args:
            encodings: {route_name: (N_i, D) int8 array}
        """
        for name, vectors in encodings.items():
            if len(vectors) == 0:
                continue
            # Majority vote per bit
            centroid = np.where(vectors.mean(axis=0) >= 0, 1, -1).astype(np.int8)
            self.add_route(name, centroid)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        data = {
            "state_dim": self.state_dim,
        }
        for name, proto in self._prototypes.items():
            data[f"proto_{name}"] = proto
        np.savez_compressed(path, **data)

    def load(self, path: str):
        data = np.load(path, allow_pickle=True)
        self.state_dim = int(data["state_dim"])
        self._prototypes = {}
        for key in data.files:
            if key.startswith("proto_"):
                name = key[6:]
                self._prototypes[name] = data[key].astype(np.int8)
        return self

    # ------------------------------------------------------------------

    def __repr__(self):
        routes = ", ".join(self._prototypes.keys()) or "(empty)"
        return f"BSMRouter(D={self.state_dim}, routes=[{routes}])"
