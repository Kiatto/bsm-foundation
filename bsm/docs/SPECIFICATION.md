# BSM Foundation Specification v1.0

> **Status:** DRAFT  
> **Last updated:** 2026-07-03  
> **Editors:** BSM Project  
> **License:** MIT

---

## Preamble

The Binary State Machine Foundation is a geometric memory platform.
It does not model language.  It models state transitions in a discrete
binary space.  Language is one domain among many.

This document defines the BSM Foundation contract.  Everything in it is
frozen for the 1.x release series.  Additions require a new minor version
or an RFC.

---

## Part I — Philosophy

### I.1  Principles

1.  **Geometry over statistics.**  Memory is a nearest-neighbour lookup in
    Hamming space, not a probability distribution over tokens.

2.  **Discreteness as feature, not bug.**  Binary states are immune to
    gradient noise, require no floating point arithmetic for retrieval,
    and trivially compose via bitwise operations.

3.  **No attention, no recurrence, no KV cache.**  BSM is feedforward at
    inference time.  Memory is content-addressable via Hamming distance,
    not via learned attention weights.

4.  **Memory is separate from computation.**  BSM provides a geometric
    store.  Any external process (LLM, planner, rule engine) may read
    and write.

5.  **Retrieval > Prediction.**  For any fixed-size memory budget,
    retrieving stored experiences dominates learned prediction in
    accuracy per byte.

6.  **Domains are distinguishable by their intrinsic geometry.**
    Participation ratio in Hamming space is an invariant of the data
    source, not of the architecture.

7.  **Small is a feature.**  Knowledge Density (bits of uncertainty
    reduced per MB of memory) is the primary efficiency metric.
    A 1 MB memory that achieves 13 % accuracy on 4096-class prediction
    is more valuable than a 100 MB model that achieves 15 %.

### I.2  Non-goals

BSM is NOT:

- A language model
- A neural network training framework
- A general-purpose vector database
- A replacement for attention
- A differentiable system

### I.3  When to use BSM

BSM is appropriate when:

- Memory footprint is constrained (< 10 MB for 100K entries at D=256)
- CPU-only inference is required
- Retrieval latency below 100 ms is acceptable
- The application benefits from deterministic, reproducible memory lookups

BSM is NOT appropriate when:

- Sub-millisecond retrieval on 10M+ entries is required (use FAISS)
- The data is naturally dense floating-point (use cosine similarity)
- Gradient information must flow through the memory operation

---

## Part II — Formal Definitions

### II.1  State

A **state** is a vector in the D-dimensional binary hypercube:

```
S ∈ {−1, +1}^D
```

Equivalently:

```
S ∈ {0, 1}^D
```

All implementations MUST accept the {−1, +1} encoding for arithmetic
operations and MAY convert to {0, 1} for storage.

The dimension D is a positive integer.  The RECOMMENDED default is
D = 256.  Valid values are multiples of 64.

### II.2  Memory

A **memory** is a set of entries:

```
M = {(S_i, P_i, metadata_i)}
```

where:

- `S_i` is a state (the key)
- `P_i` is an arbitrary payload (the value)
- `metadata_i` is a dictionary with at minimum:

  ```
  {
    "ts":  float,   # creation timestamp (Unix seconds)
    "access": int,  # access count
    "value": float  # utility / confidence in [0, 1]
  }
  ```

### II.3  Hamming Distance

The **Hamming distance** between two states is the number of positions
where they differ:

```
d_H(S_a, S_b) = |{j : S_a[j] ≠ S_b[j]}|
```

Equivalently, in {0, 1} encoding:

```
d_H(S_a, S_b) = popcount(S_a XOR S_b)
```

where popcount is the population count (number of 1 bits).

### II.4  Retrieval

**Retrieval** is the function:

```
R(S_q, M, k) = top_k_{(S, P, m) ∈ M} (−d_H(S_q, S))
```

returning the k entries with smallest Hamming distance to the query S_q.
Ties MAY be broken arbitrarily but MUST be deterministic for a given
implementation version.

### II.5  Sleep

**Sleep** is a function on memory:

```
sleep: M → M′
```

that consolidates, forgets, and reorganizes entries.  Sleep MUST:

1.  Remove entries with `value < forget_threshold` AND `age > min_age`.
2.  Rebuild any hash-based indices.
3.  Normalize values to [0, 1] range.

Sleep MAY perform additional consolidation (e.g. merging similar entries).

### II.6  Knowledge Density

**Knowledge Density** (KD) is the ratio of uncertainty reduction to
memory footprint:

```
KD = (H_prior − H_posterior) / V          [bits per byte]
```

where:

- `H_prior` is the entropy of the prediction task before memory
- `H_posterior` is the entropy after memory
- `V` is the total memory size in bytes (all storage, all overhead)

### II.7  Participation Ratio

The **participation ratio** of a set of states measures the effective
dimensionality of their distribution:

```
PR = (Σ λ_i)² / Σ λ_i²
```

where λ_i are the eigenvalues of the covariance matrix of the binary
states in {−1, +1} encoding.

PR ∈ [1, D].  Higher values indicate that more dimensions are utilized.

---

## Part III — Architecture

### III.1  Overview

```
┌─────────────────────────────────────────────┐
│            External Process                 │
│  (LLM, planner, rule engine, agent)         │
└─────────────────┬───────────────────────────┘
                  │
          state vector S
                  │
                  ▼
┌─────────────────────────────────────────────┐
│            BSM Foundation                   │
├─────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Encoder  │──▶  Memory  │──▶  Router  │  │
│  │          │  │  Store   │  │          │  │
│  └──────────┘  └────┬─────┘  └──────────┘  │
│                     │                       │
│              ┌──────▼──────┐                │
│              │  Lifecycle  │                │
│              │  (sleep,    │                │
│              │   dream,    │                │
│              │   reflect)  │                │
│              └─────────────┘                │
└─────────────────────────────────────────────┘
```

### III.2  Components

1.  **Encoder** — transforms external data into a binary state S.
2.  **Memory Store** — stores and retrieves (state, payload) pairs.
3.  **Router** — classifies a state into a named route.
4.  **Lifecycle** — consolidation, forgetting, reorganization.

### III.3  Data Flow

1.  External process produces data.
2.  Encoder converts data to state S.
3.  (Optional) Store `observe(S, payload)`.
4.  (Optional) Retrieve `recall(S, k)` → nearest neighbours.
5.  (Optional) Route `route(S)` → route name.
6.  (Periodic) `sleep()` consolidates memory.
7.  (On demand) `reflect()` reports metrics.

---

## Part IV — Component Specifications

### IV.1  Encoder

An Encoder is a callable that satisfies:

```python
def encode(data: Any) -> np.ndarray:
    """Returns (D,) int8 array in {-1, +1}."""
```

**Built-in encoders:**

| Name | Strategy | Training | Deterministic |
|------|----------|----------|---------------|
| HashEncoder | SimHash on hashed n-gram features | None | Yes (fixed seed) |
| ProjectionEncoder | SVD projection on corpus features | Fitted from corpus | Yes (fitted) |
| LearnedEncoder | MLP with contrastive loss | Trained | Yes (trained) |

**Implementing a custom encoder:**

1.  Subclass `BSMEncoder` or implement the protocol:
    - `encode(data) → np.ndarray[int8]` of shape `(D,)` or `(N, D)`
2.  Decorator `@register_encoder(name)` to make it discoverable.
3.  Encoder MUST be deterministic for a given `(implementation_version, seed)`.

### IV.2  Memory Store

The Memory Store is the core data structure.

**Storage format:** Binary states are packed into uint64 arrays.
For D = 256, each state occupies 4 × uint64 = 32 bytes.

**Search:** Exact linear scan over all entries using POPCOUNT-based
Hamming distance.

**Capacity:** RECOMMENDED soft limit is 10⁵ entries.  Performance
target for N = 10⁴, D = 256: search latency < 100 ms per query on
a single CPU core.

**Operations:**

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `put` | `(S, payload, meta?)` | Insert one entry |
| `put_batch` | `([S], [payload], [meta]?)` | Insert N entries |
| `search` | `(S, k=5) → [(payload, dist, meta)]` | k-NN by Hamming |
| `size` | `() → int` | Entry count |
| `get` | `(idx) → (S, payload, meta)` | Access by index |
| `clear` | `()` | Remove all entries |
| `vacuum` | `(keep_fn) → int` | Remove entries by predicate |
| `save` | `(path)` | Persist to disk |
| `load` | `(path) → MemoryStore` | Load from disk |

### IV.3  Router

The Router classifies a state into one of a fixed set of named routes
via nearest prototype in Hamming space (k = 1).

**Prototype construction:** For each route, the centroid (majority vote
per bit) over a set of example encodings.

**Evaluation:** Accuracy on held-out queries.  TARGET: > 80 % for
binary routing (e.g. weather vs math) at D = 256.

**Operations:**

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `add_route` | `(name, S)` | Add/update prototype |
| `remove_route` | `(name)` | Remove prototype |
| `route` | `(S) → (name, distance)` | Classify state |
| `route_batch` | `([S]) → [(name, distance)]` | Batch classify |
| `evaluate` | `([S], [labels]) → dict` | Accuracy report |
| `save` | `(path)` | Persist |
| `load` | `(path) → BSMRouter` | Load |

### IV.4  Lifecycle

The lifecycle manages memory health over time.

| Operation | Description | Frequency |
|-----------|-------------|-----------|
| `sleep()` | Consolidate, forget, re-index | Periodic (configurable) |
| `dream(n)` | Generate synthetic experience | Maintenance window |
| `reflect()` | Return health metrics | On demand |

**Sleep cycle specification:**

1.  Prune entries where `value < forget_threshold` AND
    `age > max_age` (default: value < 0.3, age > 3600 s).
2.  Normalize values to [0, 1] by dividing by max(value).
3.  Rebuild any hash indices.
4.  Return count of forgotten entries.

---

## Part V — Protocol

### V.1  Public API

The public API is the single entry point:

```python
from bsm import BSM

bsm = BSM(
    encoder="hash",       # encoder name or instance
    state_dim=256,        # D
    memory_capacity=100_000,
)

state = bsm.encode(data)            # Any → state vector
bsm.observe(state, payload)         # store
results = bsm.recall(state, k=5)    # retrieve
prediction = bsm.predict(state)     # nearest + vote
route, dist = bsm.route(state)      # classify
n_forgotten = bsm.sleep()           # consolidate
report = bsm.reflect()              # metrics
```

### V.2  Encoder Protocol

Any object conforming to the protocol is a valid encoder:

```python
class MyEncoder:
    def encode(self, data) -> np.ndarray:
        # Returns (D,) or (N, D) int8 in {-1, +1}
        ...
```

### V.3  Serialization Format

Memory Store serialization:

- **File:** `.npz` (NumPy compressed archive)
- **Contents:**
  - `keys`: `(N, n_uint64)` uint64 packed states
  - `state_dim`: int
  - `n_uint64`: int
  - `meta`: JSON string of metadata list
- **Values:** stored separately as `.vals.jsonl` (JSON lines)
- **Backward compatibility:** The loader MUST handle missing fields
  gracefully with defaults.

Router serialization:

- **File:** `.npz`
- **Contents:**
  - `state_dim`: int
  - `proto_{name}`: per-prototype state vector

### V.4  Versioning

BSM Foundation uses Semantic Versioning (MAJOR.MINOR.PATCH).

- **MAJOR:** Incompatible API or serialization format changes.
- **MINOR:** Backward-compatible feature additions.
- **PATCH:** Bug fixes, performance improvements.

The current version is 1.0.0-dev.

---

## Part VI — Memory Lifecycle

### VI.1  Stages

```
  observe
     │
     ▼
  ┌──────────┐
  │  Active  │
  │  Memory  │
  └────┬─────┘
       │ sleep()
       ▼
  ┌──────────┐
  │Consolid. │
  │+ Forget  │
  └────┬─────┘
       │ dream()
       ▼
  ┌──────────┐
  │  Dream   │
  │  Memory  │
  └──────────┘
```

### VI.2  Observation

Every `observe()` creates an entry with:

- Value = 1.0 (initial confidence)
- Timestamp = current time
- Access count = 0

### VI.3  Retrieval

Every `recall()` increments the access count of retrieved entries.
Retrieval does NOT modify the stored state.

### VI.4  Sleep

Sleep is a maintenance cycle that:

1.  **Forgets** low-value, old, unaccessed entries.
2.  **Normalizes** all values to [0, 1].
3.  **Rebuilds** data structures (indices, hash tables).

The forget policy:

```
keep = (value > threshold) OR (age < max_age) OR (access_rate > min_rate)
```

Default thresholds: `threshold=0.3`, `max_age=3600s`, `min_rate=0.001/s`.

### VI.5  Dream

Dream generates synthetic entries by:

1.  Selecting a random existing state.
2.  Adding Gaussian noise (scale = `noise_scale`).
3.  Re-binarizing via sign().
4.  Finding the nearest real experience.
5.  Storing the (dream_state, experience) with low initial value (0.3).

### VI.6  Reflect

Reflect returns:

```python
{
    "entries": int,           # current count
    "capacity": int,          # max capacity
    "usage_pct": float,       # entries / capacity
    "mean_value": float,      # over all entries
    "mean_access": float,
    "hit_rate": float,        # retrievals that found something
    "avg_latency_us": float,
    "memory_bytes": int,      # estimated RAM usage
}
```

---

## Part VII — Official Metrics

### VII.1  Core Metrics

| Metric | Symbol | Unit | Definition |
|--------|--------|------|------------|
| Accuracy | ACC | [0, 1] | Correct predictions / total predictions |
| Recall@k | R@k | [0, 1] | Fraction of queries where correct item is in top-k |
| Latency (p50) | L₅₀ | µs | Median retrieval time |
| Latency (p99) | L₉₉ | µs | 99th percentile retrieval time |
| Memory Footprint | V | bytes | Total storage (keys + values + overhead) |
| Knowledge Density | KD | bits/byte | Uncertainty reduction per byte |
| Forgetting Rate | FR | entries/hour | Average entries forgotten per sleep cycle |
| Sleep Gain | SG | [0, 1] | Accuracy after sleep / accuracy before sleep |
| Consolidation Gain | CG | [0, 1] | Accuracy after consolidation / accuracy before |
| Accuracy per KB | A/K | %/KB | ACC / (V / 1024) |

### VII.2  Geometric Metrics

| Metric | Symbol | Range | Definition |
|--------|--------|-------|------------|
| Participation Ratio | PR | [1, D] | Effective dimension |
| Zero-distance probability | P(d=0) | [0, 1] | Fraction of exact duplicate states |
| Gini coefficient | G | [0, 1] | Unevenness of bit usage (Mutual Information) |
| Bank ratio | BR | [1, ∞) | Decorrelation between feature banks |

### VII.3  Measurement Protocol

All metrics MUST be measured on:

- **Hardware:** Single CPU core, no GPU, no SIMD intrinsics beyond
  what Python's `int.bit_count()` provides.
- **Dataset:** The official BSM-Bench suite (Section VIII).
- **Seed:** Fixed across all measurements (default: 42).
- **Warm-up:** 10 queries before timing.
- **Repeats:** Minimum 3 runs, report mean.

---

## Part VIII — Benchmarks

### VIII.1  BSM-Bench

BSM-Bench is the official benchmark suite.  It consists of:

1.  **TinyStories** — 2K context windows from TinyStories.
2.  **WikiText** — 2K windows from WikiText-2.
3.  **Python Code** — 8K lines from public repositories.
4.  **DNA** — 2K segments from human genome.
5.  **Random** — Uniform random binary vectors (null baseline).

Each dataset provides:

- 2000 training entries, 500 test queries
- Fixed train/test split (seed 42)

### VIII.2  Benchmark Targets

| Metric | Target | Condition |
|--------|--------|-----------|
| Search latency | < 100 ms | N = 10⁴, D = 256, k = 5 |
| Build time | < 100 ms | N = 10⁴, D = 256 |
| Router accuracy | > 80 % | Weather vs math, D = 256 |
| Memory overhead | < 5× | N * D bits stored |
| Persistence save | < 1 s | N = 10⁴ |
| Sleep cycle | < 1 s | N = 10⁴ |

### VIII.3  Running

```bash
bsm-bench                    # full suite
bsm-bench --quick            # subset (N=1000)
bsm-bench --report html      # output format
```

Output: `report.json`, `report.md`, optionally `report.html`.

---

## Part IX — Extension Points

### IX.1  Custom Encoders

Encoders can be registered and discovered:

```python
from bsm.encoder import register_encoder

@register_encoder("my_encoder")
class MyEncoder:
    ...
```

### IX.2  Custom Metrics

Metrics can be added via the metrics registry:

```python
from bsm.metrics import register_metric

@register_metric("my_metric")
def my_metric(memory, **kwargs) -> float:
    ...
```

### IX.3  Custom Benchmarks

Benchmark datasets can be contributed:

```
bsm/bench/datasets/
    my_dataset/
        train.jsonl
        test.jsonl
        metadata.json
```

### IX.4  Forbidden Extensions

The following are NOT allowed in the Core repository:

- Neural network training loops
- GPU-specific code paths
- External vector database dependencies
- Differentiable approximations of Hamming distance
- Alternative distance functions in the retrieval path

---

## Part X — RFC Index

RFCs define extensions and modifications to this specification.

| RFC | Title | Status |
|-----|-------|--------|
| 0001 | Memory Protocol | Draft |
| 0002 | Encoder Interface | Draft |
| 0003 | Persistence Format | Draft |
| 0004 | Benchmark Specification | Draft |
| 0005 | Sleep Cycle | Draft |
| 0006 | Metric Definitions | Draft |

Each RFC in `bsm/docs/rfc/RFC-XXXX.md` follows this template:

```
Title:
Status: [Draft | Active | Superseded]
Date:
Summary:

Specification:

Rationale:

Compatibility:
```

---

## Appendices

### A.  Implementation Checklist for v1.0

- [ ] `bsm.BSM` single entry point
- [ ] Three built-in encoders
- [ ] Memory Store with POPCOUNT search
- [ ] Router with prototype evaluation
- [ ] Persistence (save/load)
- [ ] Sleep / Dream / Reflect lifecycle
- [ ] Metrics: ACC, R@k, L₅₀, L₉₉, V, KD, PR, G
- [ ] BSM-Bench: 5 datasets, 3 runs
- [ ] `pyproject.toml` with `bsm-bench` CLI
- [ ] `bsm-bench --report {json,md,html}`
- [ ] RFC 0001–0006 in `docs/rfc/`
- [ ] Test coverage > 90 %
- [ ] README with 5-line example

### B.  Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0-dev | 2026-07-03 | Initial specification draft |

---

> *This specification is the BSM Foundation contract.  Everything in it is frozen for the 1.x release series.  Changes require a new minor version or an RFC.*
