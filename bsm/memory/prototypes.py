"""
prototypes.py — Consolidamento geometrico: prototipi majority-vote.

In spazio binario il centroide di un cluster è il voto di maggioranza
bit a bit — un'operazione quasi gratuita.  Questo modulo clusterizza i
ricordi di un BSM per vicinanza di Hamming (leader clustering) e crea
uno *stato prototipo* per cluster: un'astrazione che emerge dalla pura
geometria, senza euristiche testuali.

Il recall gerarchico cerca prima tra i prototipi, poi solo dentro i
cluster migliori: ricerca sublineare quando i cluster sono compatti.

Usage:
    idx = PrototypeIndex(bsm, radius_frac=0.35)
    stats = idx.build()
    results = idx.recall(bsm.encode("query"), k=5, n_probe=2)
"""

import numpy as np
from typing import List, Tuple, Any, Optional


class PrototypeIndex:
    """Indice gerarchico di prototipi majority-vote sopra un BSM.

    Ogni prototipo è il majority vote bit a bit dei membri del suo
    cluster.  I cluster sono formati per leader clustering: un ricordo
    entra nel primo cluster il cui prototipo dista meno di
    radius_frac * state_dim, altrimenti fonda un cluster nuovo.
    """

    def __init__(self, bsm, radius_frac: float = 0.35):
        self.bsm = bsm
        self.radius_frac = radius_frac
        self._centroids: List[np.ndarray] = []   # int8 {-1,+1}
        self._members: List[List[int]] = []      # indici nello store

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @staticmethod
    def _majority_vote(states: List[np.ndarray]) -> np.ndarray:
        """Centroide binario: segno della somma bit a bit (tie → +1)."""
        total = np.sum(np.stack(states).astype(np.int32), axis=0)
        return np.where(total >= 0, 1, -1).astype(np.int8)

    @staticmethod
    def _hamming(a: np.ndarray, b: np.ndarray) -> int:
        return int(np.count_nonzero(a != b))

    def build(self, refine_passes: int = 1) -> dict:
        """Costruisce i prototipi dai ricordi presenti nello store.

        Args:
            refine_passes: passate di riassegnazione dopo il primo
                clustering (i centroidi majority-vote si stabilizzano).

        Returns:
            dict con statistiche (ricordi, prototipi, compressione).
        """
        store = self.bsm._store
        n = store.size()
        states = [store.unpack(store._keys[i]) for i in range(n)]
        radius = self.radius_frac * self.bsm.state_dim

        # Leader clustering
        self._centroids, self._members = [], []
        for i, s in enumerate(states):
            placed = False
            for c_idx, centroid in enumerate(self._centroids):
                if self._hamming(s, centroid) <= radius:
                    self._members[c_idx].append(i)
                    placed = True
                    break
            if not placed:
                self._centroids.append(s.copy())
                self._members.append([i])

        # Refine: ricalcola i centroidi come majority vote e riassegna
        for _ in range(refine_passes):
            self._centroids = [
                self._majority_vote([states[i] for i in members])
                for members in self._members
            ]
            new_members = [[] for _ in self._centroids]
            for i, s in enumerate(states):
                dists = [self._hamming(s, c) for c in self._centroids]
                new_members[int(np.argmin(dists))].append(i)
            # Elimina cluster svuotati dalla riassegnazione
            keep = [j for j, m in enumerate(new_members) if m]
            self._centroids = [self._centroids[j] for j in keep]
            self._members = [new_members[j] for j in keep]

        self._centroids = [
            self._majority_vote([states[i] for i in members])
            for members in self._members
        ]
        return self.stats()

    # ------------------------------------------------------------------
    # Recall gerarchico
    # ------------------------------------------------------------------

    def recall(self, state: np.ndarray, k: int = 5,
               n_probe: int = 2) -> List[Tuple[Any, int, dict]]:
        """Recall in due stadi: prototipi → membri dei top n_probe cluster.

        Restituisce (value, hamming_dist, meta) come bsm.recall(), con
        distanze esatte rispetto ai ricordi originali (i prototipi
        servono solo a restringere la ricerca).
        """
        if not self._centroids:
            return []
        store = self.bsm._store

        proto_dists = [self._hamming(state, c) for c in self._centroids]
        probe = np.argsort(proto_dists)[:n_probe]

        candidates = []
        for c_idx in probe:
            for i in self._members[c_idx]:
                member_state = store.unpack(store._keys[i])
                d = self._hamming(state, member_state)
                candidates.append((d, i))
        candidates.sort(key=lambda x: x[0])
        return [(store._values[i], d, store._meta[i])
                for d, i in candidates[:k]]

    def prototype_of(self, state: np.ndarray) -> Optional[np.ndarray]:
        """Il prototipo (astrazione geometrica) più vicino a *state*."""
        if not self._centroids:
            return None
        dists = [self._hamming(state, c) for c in self._centroids]
        return self._centroids[int(np.argmin(dists))]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        n_memories = sum(len(m) for m in self._members)
        sizes = [len(m) for m in self._members]
        return {
            "memories": n_memories,
            "prototypes": len(self._centroids),
            "compression": (round(n_memories / len(self._centroids), 2)
                            if self._centroids else 0.0),
            "largest_cluster": max(sizes) if sizes else 0,
            "singletons": sum(1 for s in sizes if s == 1),
        }

    def __repr__(self):
        s = self.stats()
        return (f"PrototypeIndex(memories={s['memories']}, "
                f"prototypes={s['prototypes']}, "
                f"compression={s['compression']}x)")
