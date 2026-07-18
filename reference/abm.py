"""
abm.py — Algebraic Binary Memory: reference implementation.

FROZEN against FORMALISM.md v2.0. This file is the executable
demonstration of the theory, not a framework: numpy-only,
deterministic, < 500 lines. Applications live elsewhere.

Mapping to the formalism:
    bind, permute, bundle, cleanup   →  operators (Def. 2.1, axioms A1-A3)
    phi                              →  memory potential (Def. 0.1)
    confidence                       →  calibrated confidence (Law I)
    z_gumbel, predicted_accuracy,
    capacity                         →  capacity law (Law IV, Gumbel form)
    Memory.query                     →  elementary query (Def. 3.2)
    Memory.chain                     →  hop composition (Theorem 3.4: p^h)
    Memory.member                    →  algebraic truth oracle (Prop. 3.7)
    Memory.compile_pairs             →  sleep-time compilation (Calculus §5.2)
    compose                          →  Normal-Form composition (NF.1) with
                                        exact Bridge Elimination (Thm 2.10)

Everything is deterministic: item hypervectors derive from names.
"""

from math import erf, log, pi, sqrt
from typing import Any, List, Optional, Sequence, Tuple

import hashlib
import numpy as np

__version__ = "1.0.1"                       # frozen with FORMALISM v2.0
# 1.0.1: confidence() returns a plain Python float instead of
# np.float64 (public-API surface fix; numerically identical).


# ---------------------------------------------------------------------------
# Operators (axioms A1-A3)
# ---------------------------------------------------------------------------

def random_hv(name: str, dim: int) -> np.ndarray:
    """Deterministic item hypervector in {-1,+1}^dim (item memory seed)."""
    seed = int.from_bytes(hashlib.md5(name.lower().encode()).digest()[:4],
                          "little")
    rng = np.random.RandomState(seed)
    return np.where(rng.rand(dim) > 0.5, 1, -1).astype(np.int8)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A1 — binding: elementwise product (== XOR in {-1,+1}).
    Isometric involution: bind(bind(x, k), k) == x."""
    return (a.astype(np.int16) * b.astype(np.int16)).astype(np.int8)


def permute(x: np.ndarray, k: int = 1) -> np.ndarray:
    """Role/position marker rho (cyclic shift); a group homomorphism:
    permute(bind(x, y)) == bind(permute(x), permute(y))."""
    return np.roll(x, k)


def bundle(states: Sequence[np.ndarray],
           weights: Optional[Sequence[int]] = None) -> np.ndarray:
    """A2 — bundling: (weighted) bitwise majority vote.
    Variationally: the state maximizing sum_i w_i * phi(x, x_i).
    Deterministic tie-break for even totals."""
    ws = weights if weights is not None else [1] * len(states)
    total = np.zeros(states[0].shape[0], dtype=np.int64)
    for x, w in zip(states, ws):
        total += w * x.astype(np.int64)
    if sum(ws) % 2 == 0:
        total = total * 2 + random_hv("__tie__", states[0].shape[0])
    return np.where(total >= 0, 1, -1).astype(np.int8)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


# ---------------------------------------------------------------------------
# The potential and its statistics (Level 0 / Level 1)
# ---------------------------------------------------------------------------

def phi(x: np.ndarray, y: np.ndarray) -> float:
    """Memory potential (Def. 0.1): normalized z-score of x against the
    null distribution of y. 0 = chance; >0 = better than chance."""
    d = x.shape[0]
    return (d / 2 - hamming(x, y)) / sqrt(d)


def confidence(dist: float, dim: int, temperature: float = 8.0) -> float:
    """Calibrated confidence: logistic in the Hamming-null z-score.
    0.5 = indistinguishable from noise."""
    z = (dim / 2.0 - dist) / (sqrt(dim) / 2.0)
    return float(1.0 / (1.0 + np.exp(-z / temperature)))


def z_gumbel(m: int) -> float:
    """Second-order extreme-value threshold for the min of m null
    distances (the codebook noise floor)."""
    zm = sqrt(2 * log(m))
    return zm - (log(log(m)) + log(4 * pi)) / (2 * zm)


def predicted_accuracy(n_facts: int, dim: int, codebook: int) -> float:
    """Law IV, forward direction: theory-predicted single-query accuracy
    at load n_facts — the 'capacity contract'. No fitted parameters."""
    margin = sqrt(2 * dim / (pi * n_facts)) - z_gumbel(codebook)
    return 0.5 * (1 + erf(margin / sqrt(2)))


def capacity(dim: int, codebook: int, k: float = 0.92) -> float:
    """Law IV, inverse direction: the 50%-accuracy collapse load N*.
    k = 0.92 +/- 0.03 (measured); k = 1 is the theoretical constant."""
    return k * 2 * dim / (pi * z_gumbel(codebook) ** 2)


# ---------------------------------------------------------------------------
# Item memory (the trusted computing base)
# ---------------------------------------------------------------------------

class ItemMemory:
    """Codebook name <-> hypervector, with cleanup (A3: idempotent
    projection = argmax of phi over the codebook)."""

    def __init__(self, dim: int):
        self.dim = dim
        self._names: List[str] = []
        self._states: List[np.ndarray] = []
        self._index: dict = {}

    def add(self, name: str) -> np.ndarray:
        if name not in self._index:
            self._index[name] = len(self._names)
            self._names.append(name)
            self._states.append(random_hv(name, self.dim))
        return self._states[self._index[name]]

    def get(self, name: str) -> np.ndarray:
        return self._states[self._index[name]]

    def cleanup(self, noisy: np.ndarray,
                subset: Optional[Sequence[str]] = None) -> Tuple[str, int]:
        """(name, distance) of the nearest codeword. `subset` restricts
        to a typed sub-codebook (projection operator: capacity gain
        z_G(M)^2 / z_G(|S|)^2, verified within 5%)."""
        names = subset if subset is not None else self._names
        best, bd = "", self.dim + 1
        for name in names:
            d = hamming(noisy, self._states[self._index[name]])
            if d < bd:
                best, bd = name, d
        return best, bd

    def __len__(self):
        return len(self._names)


# ---------------------------------------------------------------------------
# Holographic memory (the machine)
# ---------------------------------------------------------------------------

class Memory:
    """A single D-bit holographic trace holding N facts.

    Contract (Law IV): expected single-query accuracy at the current
    load is `self.expected_accuracy()`; the collapse load is
    `capacity(dim, len(items))`. Both are computable before any query.
    """

    def __init__(self, dim: int = 2048,
                 items: Optional[ItemMemory] = None):
        self.dim = dim
        self.items = items or ItemMemory(dim)
        self._facts: List[np.ndarray] = []
        self._trace: Optional[np.ndarray] = None

    # -- write ----------------------------------------------------------

    def key(self, subject: str, relation: str) -> np.ndarray:
        return bind(self.items.add(subject),
                    permute(self.items.add(relation), 1))

    def fact_hv(self, s: str, r: str, o: str) -> np.ndarray:
        return bind(self.key(s, r), self.items.add(o))

    def store(self, s: str, r: str, o: str, weight: int = 1):
        for _ in range(weight):                    # Law VII: N_eff = sum w^2
            self._facts.append(self.fact_hv(s, r, o))
        self._trace = bundle(self._facts)

    # -- read (probabilistic steps, cost from Level 1) --------------------

    def query(self, s: str, r: str,
              subset: Optional[Sequence[str]] = None) -> Tuple[str, float]:
        """Elementary query: cleanup(T xor key). Returns (object name,
        calibrated confidence)."""
        noisy = bind(self._trace, self.key(s, r))
        name, dist = self.items.cleanup(noisy, subset)
        return name, confidence(dist, self.dim)

    def chain(self, start: str, relations: Sequence[str]) -> Tuple[str, float]:
        """Multi-hop with cleanup per hop. Theorem 3.4: success = p^h;
        the cleanup reset is constitutive (T xor T no-go), not optional."""
        node, conf = start, 1.0
        for r in relations:
            node, c = self.query(node, r)
            conf *= c
        return node, conf

    def member(self, s: str, r: str, o: str, z_min: float = 3.0) -> bool:
        """Algebraic truth oracle (Prop. 3.7): is the fact in the trace?
        A single Hamming distance against a single vector."""
        d = hamming(self.fact_hv(s, r, o), self._trace)
        return (self.dim / 2 - d) / (sqrt(self.dim) / 2) >= z_min

    # -- exact steps (free and certain, from A1) --------------------------

    @staticmethod
    def compose(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        """NF.1: one XOR generates the composed fact; the shared bridge
        is eliminated exactly (Theorem 2.10) and is unrecoverable from
        the result alone."""
        return bind(f1, f2)

    def compile_pairs(self, pairs: Sequence[Tuple[Tuple[str, str, str],
                                                  Tuple[str, str, str]]]
                      ) -> "Memory":
        """Sleep-time compilation (Calculus §5.2, verified): exact
        compositions consolidated into a second trace. Two-hop queries
        against it cost ONE cleanup: p(N2) instead of p^2."""
        compiled = Memory(self.dim, self.items)
        for (s1, r1, o1), (s2, r2, o2) in pairs:
            compiled._facts.append(self.compose(self.fact_hv(s1, r1, o1),
                                                self.fact_hv(s2, r2, o2)))
        compiled._trace = bundle(compiled._facts)
        return compiled

    def query_compiled(self, s: str, r1: str, r2: str) -> Tuple[str, float]:
        """Query a compiled trace with the static composed key
        c_s xor rho(r1) xor rho(r2)."""
        k = bind(self.items.add(s),
                 bind(permute(self.items.add(r1), 1),
                      permute(self.items.add(r2), 1)))
        name, dist = self.items.cleanup(bind(self._trace, k))
        return name, confidence(dist, self.dim)

    # -- introspection ----------------------------------------------------

    def expected_accuracy(self) -> float:
        """The contract: theory-predicted accuracy at the current load."""
        if not self._facts:
            return 1.0
        return predicted_accuracy(len(self._facts), self.dim,
                                  max(len(self.items), 2))

    def __repr__(self):
        return (f"Memory(D={self.dim}, facts={len(self._facts)}, "
                f"items={len(self.items)}, "
                f"expected_acc={self.expected_accuracy():.0%})")
