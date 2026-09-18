"""
bitpack.py — Bitpacked SIMD Acceleration for Algebraic Binary Memory (ABM).

Provides 64-bit packed bitset representations for hypervectors in {-1, +1}^D,
with 8x lower memory footprint (256 B vs 2048 B per vector at D=2048).

Algebraically equivalent to FORMALISM.md v2.0; test_bitpack.py pins bind, Hamming
and query/member equivalence against the dense reference implementation.

Measured against reference/abm.py at D=2048, 500 facts (examples/bitpack_bench.py):
store ~88x (median of 3 runs, 7% spread), query 12.5x (3% spread), member oracle
only stable to one significant figure — roughly 5-9x, 48% spread across runs, so
do not quote it more precisely than that. Caveat on the store figure: most of it
is that reference/abm.py constructs a np.random.RandomState per codeword, which
costs ~193 us; the same reseeding fix applies there and would close most of the
gap. Cleanup is an exhaustive vectorized scan (4.8 ms at M=100,000) — see
batch_hamming_packed for why no index beats it.
"""

from functools import lru_cache
from math import erf, exp, log, pi, sqrt
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import hashlib
import threading
import numpy as np



_rng_local = threading.local()


def _seeded_rng(seed: int) -> np.random.RandomState:
    """A RandomState reseeded to `seed`, reusing one generator per thread.

    Constructing np.random.RandomState(seed) costs ~193 us at every call, which
    dominated every operation touching a new entity; reseeding an existing
    generator costs 1.65 us and yields the identical bit stream. One generator per
    thread keeps that reuse safe under concurrent access.
    """
    rng = getattr(_rng_local, "rng", None)
    if rng is None:
        rng = _rng_local.rng = np.random.RandomState(0)
    rng.seed(seed)
    return rng


def random_packed_hv(name: str, dim: int) -> np.ndarray:
    """Generate a packed uint64 hypervector for a given name.

    Known limitation of the v1 codeword format, not a bug to fix here: the seed is
    32 bits wide, so distinct names collide with birthday probability — around a 69%
    chance of at least one collision by M = 100,000 entities, and two colliding
    names share a codeword and become indistinguishable. Widening the seed would
    change every codeword and invalidate every artifact and published measurement
    produced under the freeze. It belongs to a v2 of the format, if there is one.
    """
    if dim % 64 != 0:
        raise ValueError(f"Dimension {dim} must be a multiple of 64 for bitpacking.")
    seed = int.from_bytes(hashlib.md5(name.lower().encode()).digest()[:4], "little")
    return np.packbits(_seeded_rng(seed).rand(dim) > 0.5, bitorder="big").view(np.uint64)


def pack_vector(v: np.ndarray) -> np.ndarray:
    """Convert a dense vector {-1, +1}^D or {0, 1}^D to packed uint64 array."""
    dim = v.shape[0]
    if dim % 64 != 0:
        raise ValueError(f"Dimension {dim} must be a multiple of 64.")
    bits = np.where(v > 0, 1, 0).astype(np.uint8)
    return np.packbits(bits, bitorder="big").view(np.uint64)


def unpack_vector(packed: np.ndarray, dim: int) -> np.ndarray:
    """Convert packed uint64 array back to dense {-1, +1}^D int8 array."""
    bytes_arr = packed.view(np.uint8)
    bits = np.unpackbits(bytes_arr, bitorder="big")[:dim]
    return np.where(bits > 0, 1, -1).astype(np.int8)


def bind_packed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A1 — Bitwise XNOR binding on packed uint64 arrays.
    
    In {-1,+1} space, product maps to XNOR when +1 is bit 1 and -1 is bit 0:
    +1 * +1 = +1  <=> NOT(1 ^ 1) = 1
    -1 * -1 = +1  <=> NOT(0 ^ 0) = 1
    +1 * -1 = -1  <=> NOT(1 ^ 0) = 0
    -1 * +1 = -1  <=> NOT(0 ^ 1) = 0
    """
    return np.bitwise_not(np.bitwise_xor(a, b))



def permute_packed(packed: np.ndarray, k: int, dim: int) -> np.ndarray:
    """Rho — Cyclic permutation by k positions on packed uint64 vector.

    Operates directly on the packed bitstring (byte roll + intra-byte bit shift),
    so no dense round-trip is needed. Exactly equivalent to
    pack_vector(np.roll(unpack_vector(packed, dim), k)) for any k.
    """
    k %= dim
    if k == 0:
        return packed.copy()

    # Bits are laid out MSB-first inside each byte of the memory-order uint8 view,
    # so a positive np.roll on the dense vector is a right rotation of that bitstring.
    src = packed.view(np.uint8)
    byte_shift, bit_shift = divmod(k, 8)
    rolled = np.roll(src, byte_shift)
    if bit_shift:
        hi = rolled.astype(np.uint16) >> bit_shift
        lo = np.roll(rolled, 1).astype(np.uint16) << (8 - bit_shift)
        rolled = ((hi | lo) & 0xFF).astype(np.uint8)
    return np.ascontiguousarray(rolled).view(np.uint64)


def bundle_packed(packed_list: Sequence[np.ndarray], dim: int) -> np.ndarray:
    """A2 — Bitwise majority vote on a sequence of packed vectors."""
    if not packed_list:
        raise ValueError("Cannot bundle empty sequence.")
    if len(packed_list) == 1:
        return packed_list[0].copy()

    # Unpack to dense for majority voting
    sum_vector = np.zeros(dim, dtype=np.int64)
    for p in packed_list:
        sum_vector += unpack_vector(p, dim).astype(np.int64)

    if len(packed_list) % 2 == 0:
        tie = unpack_vector(random_packed_hv("__tie__", dim), dim).astype(np.int64)
        sum_vector = sum_vector * 2 + tie

    majority_bits = np.where(sum_vector >= 0, 1, 0).astype(np.uint8)
    return np.packbits(majority_bits, bitorder="big").view(np.uint64)


# Precomputed 256-entry lookup table for fast uint8 popcount (fallback path)
POPCNT_LUT = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)

# numpy >= 2.0 exposes a native per-element popcount, which operates directly on
# uint64 words instead of expanding each word into 8 LUT lookups.
_HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")


def popcount_packed(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Population count of packed uint64 words, summed along `axis`."""
    if _HAS_BITWISE_COUNT:
        return np.bitwise_count(x).sum(axis=axis, dtype=np.int32)
    return POPCNT_LUT[x.view(np.uint8)].sum(axis=axis, dtype=np.int32)


def hamming_packed(a: np.ndarray, b: np.ndarray) -> int:
    """Compute Hamming distance between two packed uint64 vectors.

    For a single pair, CPython's arbitrary-precision int.bit_count() over the raw
    bytes beats a numpy popcount-and-reduce (1.1 us vs 2.7 us at D=2048): one
    interpreter-level call instead of three array kernels on 32 elements. Use
    batch_hamming_packed when comparing against many vectors.

    int.bit_count() requires Python >= 3.10, which is the floor declared in
    pyproject.toml — no fallback, so a wrong floor fails loudly instead of silently
    degrading. numpy has no such guarantee: see popcount_packed for that fallback.
    """
    return int.from_bytes(np.bitwise_xor(a, b).tobytes(), "little").bit_count()


def batch_hamming_packed(codebook_matrix: np.ndarray, query: np.ndarray,
                         chunk: int = 4096) -> np.ndarray:
    """Compute Hamming distance from query to all rows in codebook_matrix (M, D/64).

    Returns an int32 array of shape (M,) containing exact Hamming distances.
    Processed in row blocks into preallocated buffers: the peak temporary is
    O(chunk * D/64) instead of O(M * D/64), which at M=100k, D=2048 is the
    difference between a 0.8 MB scratch buffer and a 25.6 MB one.

    This scan is exhaustive by design. Metric indexing (VP-tree, LSH) does not
    help here: a cleanup query sits at Hamming distance 0.45-0.48*D from its own
    codeword while non-targets sit at 0.50*D, so no partial-bit or
    triangle-inequality bound can prune. Measured in docs/falsification_report.md
    (addendum F4).
    """
    rows, words = codebook_matrix.shape
    if rows == 0:
        return np.empty(0, dtype=np.int32)

    # The per-word popcounts are reduced with a BLAS gemv against a vector of ones
    # rather than np.sum: measured 50 us vs 63 us at M=1001, D=2048. Every term is a
    # word popcount <= 64 and each row sums to <= D, so the float32 accumulation is
    # exact and the cast back to int32 is lossless. Below a few hundred rows the
    # gemv setup costs more than it saves, so np.sum stays.
    use_gemv = _HAS_BITWISE_COUNT and rows >= 256
    ones = np.ones(words, dtype=np.float32) if use_gemv else None
    out = np.empty(rows, dtype=np.int32)
    for start in range(0, rows, chunk):
        end = min(start + chunk, rows)
        xor = np.bitwise_xor(codebook_matrix[start:end], query)
        counts = np.bitwise_count(xor) if _HAS_BITWISE_COUNT else POPCNT_LUT[xor.view(np.uint8)]
        if use_gemv:
            out[start:end] = (counts.astype(np.float32) @ ones).astype(np.int32)
        else:
            out[start:end] = counts.sum(axis=-1, dtype=np.int32)
    return out



def z_gumbel(m: int) -> float:
    """Second-order extreme-value threshold for min of m null distances."""
    zm = sqrt(2 * log(m))
    return zm - (log(log(m)) + log(4 * pi)) / (2 * zm)


def confidence_packed(dist: float, dim: int, temperature: float = 8.0) -> float:
    """Calibrated confidence logistic in Hamming z-score."""
    z = (dim / 2.0 - dist) / (sqrt(dim) / 2.0)
    return 1.0 / (1.0 + exp(-z / temperature))


def predicted_accuracy(n_facts: int, dim: int, codebook: int) -> float:
    """Law IV forward: theory-predicted single-query accuracy."""
    margin = sqrt(2 * dim / (pi * n_facts)) - z_gumbel(codebook)
    return 0.5 * (1 + erf(margin / sqrt(2)))


def hamming_z(dist: float, dim: int) -> float:
    """Signed z-score of a Hamming distance under the null (unrelated vectors)."""
    return (dim / 2.0 - dist) / (sqrt(dim) / 2.0)


def _normal_quantile(p: float) -> float:
    """Phi^{-1}(p) by bisection on erf. Adequate for threshold computation."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if 0.5 * (1 + erf(mid / sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


@lru_cache(maxsize=4096)
def acceptance_z(candidates: int, alpha: float = 0.01) -> float:
    """Minimum z-score a cleanup hit must reach to be accepted.

    A cleanup reports the *minimum* distance over `candidates` codewords, so its
    z-score is a maximum of `candidates` draws and is biased upward even when the
    queried fact was never stored. Under the null the maximum z has CDF
    Phi(t)^candidates, so requiring z >= Phi^{-1}((1 - alpha)^(1/candidates))
    caps the false-accept rate at alpha regardless of codebook size.

    This is the correction that a fixed confidence threshold (e.g. 0.75) lacks:
    a fixed threshold's false-accept rate grows with the codebook.
    """
    m = max(int(candidates), 1)
    return _normal_quantile((1.0 - alpha) ** (1.0 / m))


class BitpackedItemMemory:
    """Codebook of item hypervectors stored in packed 64-bit format."""

    def __init__(self, dim: int = 2048):
        if dim % 64 != 0:
            raise ValueError(f"Dimension {dim} must be a multiple of 64.")
        self.dim = dim
        self._names: List[str] = []
        self._states: List[np.ndarray] = []  # list of uint64 arrays
        self._index: dict = {}
        self._matrix: Optional[np.ndarray] = None  # 2D array of shape (M, dim/64)
        self._perm_cache: dict = {}

    def add_permuted(self, name: str, k: int = 1) -> np.ndarray:
        """Rho^k applied to the codeword of `name`, memoized.

        Codewords are deterministic functions of the name, so the permuted form is
        immutable and safe to cache; callers must treat the result as read-only.
        """
        cached = self._perm_cache.get((name, k))
        if cached is None:
            cached = permute_packed(self.add(name), k, self.dim)
            self._perm_cache[(name, k)] = cached
        return cached

    def add(self, name: str) -> np.ndarray:
        if name not in self._index:
            self._index[name] = len(self._names)
            self._names.append(name)
            self._states.append(random_packed_hv(name, self.dim))
            self._matrix = None  # invalidate cache
        return self._states[self._index[name]]

    def get(self, name: str) -> np.ndarray:
        return self._states[self._index[name]]

    def _ensure_matrix(self):
        if self._matrix is None and self._states:
            self._matrix = np.vstack(self._states)

    def candidate_count(self, subset: Optional[Sequence[str]] = None) -> int:
        """Number of codewords a cleanup would compete over (needed to calibrate)."""
        if subset is None:
            return len(self._names)
        return sum(1 for n in subset if n in self._index)

    def cleanup(self, noisy: np.ndarray, subset: Optional[Sequence[str]] = None) -> Tuple[str, int]:
        """Find the nearest codeword to `noisy` by exhaustive vectorized Hamming scan.

        Returns (name, distance); ("", dim + 1) when there is nothing to compare against.
        """
        if subset is not None:
            names = [n for n in subset if n in self._index]
            if not names:
                return "", self.dim + 1
            sub_matrix = np.vstack([self._states[self._index[n]] for n in names])
            dists = batch_hamming_packed(sub_matrix, noisy)
            best_idx = int(np.argmin(dists))
            return names[best_idx], int(dists[best_idx])

        self._ensure_matrix()
        if self._matrix is None or len(self._names) == 0:
            return "", self.dim + 1

        dists = batch_hamming_packed(self._matrix, noisy)
        best_idx = int(np.argmin(dists))
        return self._names[best_idx], int(dists[best_idx])

    def save(self, dir_path: Union[str, Path]):
        """Save codebook items to disk."""
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        self._ensure_matrix()

        import json
        with open(path / "names.json", "w") as f:
            json.dump(self._names, f)

        if self._matrix is not None:
            np.save(path / "matrix.npy", self._matrix)

    @classmethod
    def load(cls, dir_path: Union[str, Path], mmap_mode: Optional[str] = None) -> "BitpackedItemMemory":
        """Load codebook items from disk with optional mmap_mode."""
        import json
        path = Path(dir_path)
        with open(path / "names.json", "r") as f:
            names = json.load(f)

        matrix_path = path / "matrix.npy"
        if matrix_path.exists():
            matrix = np.load(matrix_path, mmap_mode=mmap_mode)
            dim = matrix.shape[1] * 64
        else:
            dim = 2048
            matrix = None

        item_mem = cls(dim=dim)
        item_mem._names = names
        if matrix is not None:
            item_mem._matrix = matrix
            item_mem._states = [matrix[i] for i in range(matrix.shape[0])]
            item_mem._index = {name: i for i, name in enumerate(names)}
        return item_mem

    def __len__(self):
        return len(self._names)


class CodebookView:
    """A contiguous, cached subset of a codebook to restrict cleanup to.

    Used to hold the object-role codewords. Scope: this is an optimization for the
    *current query planner*, which only resolves (subject, relation) -> object. It
    is not a general property of ABM. A planner that also answered
    (object, relation) -> subject, or (?, relation, object), would need its own view
    per resolved position — restricting to objects would then silently drop valid
    answers. Whoever adds such a query shape must widen or bypass this view.
    """

    def __init__(self, items: BitpackedItemMemory, names: Optional[Sequence[str]] = None):
        self.items = items
        self._names: List[str] = []
        self._seen: set = set()
        self._matrix: Optional[np.ndarray] = None
        for name in names or ():
            self.register(name)

    def register(self, name: str):
        if name not in self._seen:
            self._seen.add(name)
            self._names.append(name)
            self._matrix = None

    @property
    def names(self) -> List[str]:
        return self._names

    def cleanup(self, noisy: np.ndarray) -> Tuple[str, int]:
        if not self._names:
            return "", self.items.dim + 1
        if self._matrix is None:
            self._matrix = np.vstack([self.items.get(n) for n in self._names])
        dists = batch_hamming_packed(self._matrix, noisy)
        best = int(np.argmin(dists))
        return self._names[best], int(dists[best])

    def __len__(self):
        return len(self._names)


class BitpackedMemory:
    """High-performance packed holographic memory trace."""

    def __init__(self, dim: int = 2048, items: Optional[BitpackedItemMemory] = None):
        if dim % 64 != 0:
            raise ValueError(f"Dimension {dim} must be a multiple of 64.")
        self.dim = dim
        # Explicit None check: BitpackedItemMemory defines __len__, so an empty
        # shared codebook is falsy and `items or ...` would silently discard it.
        self.items = items if items is not None else BitpackedItemMemory(dim)
        self._facts: List[np.ndarray] = []
        self._trace: Optional[np.ndarray] = None
        self._dirty: bool = False
        self._key_cache: dict = {}
        # Running per-position count of set bits over all stored facts (weights
        # included), kept in sync with _facts so the trace never re-bundles the whole
        # list. Counting ones rather than +-1 votes keeps the update a single uint8
        # add: 2.5 us per store versus 12.3 us for the dense +-1 round-trip.
        self._ones: Optional[np.ndarray] = None
        self._weight: int = 0
        # Codewords ever stored in object position — the only admissible answers.
        self._answers = CodebookView(self.items)

    def key(self, subject: str, relation: str) -> np.ndarray:
        """Bind(subject, Rho(relation)) — memoized, treat the result as read-only."""
        cached = self._key_cache.get((subject, relation))
        if cached is None:
            s_hv = self.items.add(subject)
            r_hv = self.items.add_permuted(relation, 1)
            cached = bind_packed(s_hv, r_hv)
            self._key_cache[(subject, relation)] = cached
        return cached

    def fact_hv(self, s: str, r: str, o: str) -> np.ndarray:
        return bind_packed(self.key(s, r), self.items.add(o))

    def _ensure_ones(self) -> np.ndarray:
        """Per-position count of set bits over all stored facts, built on demand."""
        if self._ones is None:
            if self._facts:
                packed = np.vstack(self._facts).view(np.uint8)
                self._ones = np.unpackbits(packed, axis=1).sum(axis=0, dtype=np.int32)
            else:
                self._ones = np.zeros(self.dim, dtype=np.int32)
            self._weight = len(self._facts)
        return self._ones

    def store_many(self, triples: Sequence[Tuple[str, str, str]], weight: int = 1):
        """Store a batch of (subject, relation, object) triples.

        One unpackbits over the whole batch plus one column sum, instead of a pair of
        array calls per fact: measured 1.6 us per fact versus 4.5 us for store() at
        D=2048. Produces the same trace and the same answers as calling store() on
        each triple; with weight > 1 the internal fact ordering differs, which the
        majority-vote bundle does not depend on.
        """
        if not triples:
            return
        facts = [self.fact_hv(s, r, o) for s, r, o in triples]
        for _ in range(weight):
            self._facts.extend(facts)
        if weight:
            if self._answers is not None:
                for _s, _r, o in triples:
                    self._answers.register(o)
            if self._ones is not None:
                batch = np.unpackbits(np.vstack(facts).view(np.uint8), axis=1)
                self._ones += batch.sum(axis=0, dtype=np.int32) * weight
                self._weight += weight * len(facts)
        self._dirty = True

    def store(self, s: str, r: str, o: str, weight: int = 1):
        fhv = self.fact_hv(s, r, o)
        for _ in range(weight):
            self._facts.append(fhv)
        if weight:
            if self._answers is not None:
                self._answers.register(o)
            if self._ones is not None:
                bits = np.unpackbits(fhv.view(np.uint8))
                if weight == 1:
                    np.add(self._ones, bits, out=self._ones, casting="unsafe")
                else:
                    self._ones += bits * weight
                self._weight += weight
        self._dirty = True

    @property
    def trace(self) -> Optional[np.ndarray]:
        """A2 — majority-vote bundle of all stored facts.

        Materialized from the running bit counter in O(D), independent of the number
        of stored facts. Bit-identical to bundle_packed(self._facts): with `ones`
        set bits out of `n` votes the +-1 sum is 2*ones - n, so the majority bit is
        ones > n/2, with the deterministic tie vector breaking ones == n/2 (only
        reachable when n is even).
        """
        if self._dirty and self._facts:
            ones = self._ensure_ones()
            n = self._weight
            if n % 2:
                bits = ones > (n >> 1)  # n odd: no exact tie is possible
            else:
                half = n >> 1
                bits = ones > half
                tied = ones == half
                if tied.any():
                    tie = np.unpackbits(random_packed_hv("__tie__", self.dim).view(np.uint8))
                    bits |= tied & (tie > 0)
            self._trace = np.packbits(bits, bitorder="big").view(np.uint64)
            self._dirty = False
        return self._trace

    def query_raw(self, s: str, r: str, subset: Optional[Sequence[str]] = None,
                  answers_only: bool = False) -> Tuple[str, int]:
        """Cleanup result as (name, Hamming distance), without any thresholding.

        answers_only restricts the cleanup to codewords stored in object position.
        """
        tr = self.trace
        if tr is None:
            return "", self.dim + 1
        noisy = bind_packed(tr, self.key(s, r))
        if answers_only and subset is None and self._answers is not None:
            return self._answers.cleanup(noisy)
        return self.items.cleanup(noisy, subset)

    def query(self, s: str, r: str, subset: Optional[Sequence[str]] = None,
              answers_only: bool = False) -> Tuple[str, float]:
        tr = self.trace
        if tr is None:
            return "", 0.5
        name, dist = self.query_raw(s, r, subset, answers_only=answers_only)
        return name, confidence_packed(dist, self.dim)

    def _candidate_count(self, subset: Optional[Sequence[str]], answers_only: bool) -> int:
        if answers_only and subset is None and self._answers is not None:
            return len(self._answers)
        return self.items.candidate_count(subset)

    def query_calibrated(self, s: str, r: str, subset: Optional[Sequence[str]] = None,
                         alpha: float = 0.01,
                         answers_only: bool = True) -> Tuple[Optional[str], float, float]:
        """Query with a false-accept rate capped at `alpha`.

        Returns (answer_or_None, confidence, z). The answer is None when the hit
        does not clear acceptance_z() for the number of competing codewords, i.e.
        when it is indistinguishable from the best of that many unrelated vectors.

        answers_only (default) competes only against codewords ever stored in object
        position. Under the current query planner — (subject, relation) -> object —
        subjects and relations are not admissible answers, so including them only
        adds null draws: it costs latency and raises the bar a true hit must clear.
        This is a planner-specific optimization, not a property of ABM: a planner
        resolving a different position must pass False or supply its own subset.
        """
        tr = self.trace
        if tr is None:
            return None, 0.5, 0.0
        name, dist = self.query_raw(s, r, subset, answers_only=answers_only)
        z = hamming_z(dist, self.dim)
        conf = confidence_packed(dist, self.dim)
        if not name or z < acceptance_z(self._candidate_count(subset, answers_only), alpha):
            return None, conf, z
        return name, conf, z

    def chain(self, start: str, relations: Sequence[str]) -> Tuple[str, float]:
        node, conf = start, 1.0
        for r in relations:
            node, c = self.query(node, r)
            conf *= c
        return node, conf

    def member(self, s: str, r: str, o: str, z_min: float = 3.0) -> bool:
        tr = self.trace
        if tr is None:
            return False
        d = hamming_packed(self.fact_hv(s, r, o), tr)
        return hamming_z(d, self.dim) >= z_min

    def expected_accuracy(self) -> float:
        if not self._facts:
            return 1.0
        return predicted_accuracy(len(self._facts), self.dim, max(len(self.items), 2))

    def save(self, dir_path: Union[str, Path]):
        """Save memory trace and codebook items to disk."""
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        self.items.save(path / "items")

        if self._facts:
            facts_mat = np.vstack(self._facts)
            np.save(path / "facts.npy", facts_mat)

        tr = self.trace
        if tr is not None:
            np.save(path / "trace.npy", tr)

        import json
        with open(path / "meta.json", "w") as f:
            # answer_names is not derivable from the packed facts, so it has to be
            # persisted: without it a reloaded memory would answer nothing under
            # answers_only.
            json.dump({"dim": self.dim, "facts_count": len(self._facts),
                       "answer_names": self._answers.names}, f)

    @classmethod
    def load(cls, dir_path: Union[str, Path], mmap_mode: Optional[str] = None) -> "BitpackedMemory":
        """Load memory trace from disk with zero-copy mmap support."""
        import json
        path = Path(dir_path)
        with open(path / "meta.json", "r") as f:
            meta = json.load(f)

        dim = meta.get("dim", 2048)
        items = BitpackedItemMemory.load(path / "items", mmap_mode=mmap_mode)

        mem = cls(dim=dim, items=items)
        # An artifact written before answer_names existed carries no object roles;
        # None means "unknown", which falls back to the whole codebook rather than
        # silently answering nothing.
        mem._answers = (CodebookView(items, meta["answer_names"])
                        if "answer_names" in meta else None)

        facts_path = path / "facts.npy"
        if facts_path.exists():
            facts_mat = np.load(facts_path, mmap_mode=mmap_mode)
            mem._facts = [facts_mat[i] for i in range(facts_mat.shape[0])]

        trace_path = path / "trace.npy"
        if trace_path.exists():
            mem._trace = np.load(trace_path, mmap_mode=mmap_mode)
            mem._dirty = False

        return mem

    def __repr__(self):
        return (f"BitpackedMemory(D={self.dim}, facts={len(self._facts)}, "
                f"items={len(self.items)}, expected_acc={self.expected_accuracy():.0%})")

