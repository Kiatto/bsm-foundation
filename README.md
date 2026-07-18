# ABM — Algebraic Binary Memory

**A deterministic algebraic memory runtime for compiled symbolic
knowledge. It complements LLMs rather than replacing them.**

The LLM is the compiler: it reads documents and emits facts. ABM is the
runtime: it stores them in a single holographic binary trace, composes
them algebraically, and answers multi-hop queries — with **accuracy,
capacity and reasoning depth contracted *before* deployment**, from a
quantitative theory with no fitted parameters.

```
Documents → LLM compiler → triples → ABM → Inspector → Memory Contract
                                       ↓
                              queries + explanations
```

## Quickstart

```bash
pip install abm-runtime
abm demo
```

Developer guide (no theory required): [docs/SDK.md](docs/SDK.md).
The runtime is two files, numpy-only, ~400 lines total.

```python
from abm import Memory
from inspector import stats, contract, report

mem = Memory(dim=8192)
mem.store("payment_service", "requires", "auth_service")
mem.store("auth_service", "writes_to", "session_store")

# elementary query: (answer, calibrated confidence — 0.5 = chance)
mem.query("payment_service", "requires")     # → ("auth_service", 0.99…)

# multi-hop reasoning
mem.chain("payment_service", ["requires", "writes_to"])  # → "session_store"

# algebraic truth oracle: one Hamming distance
mem.member("payment_service", "requires", "auth_service")  # → True

# the Memory Contract — computed from theory BEFORE any query
print(contract(mem, grounding=0.93))
```

Contract output for a memory at realistic load (the `edge` config of
[`examples/scale_bench.py`](examples/scale_bench.py), 250 facts):

```
MEMORY CONTRACT
  Capacity        <= 561 facts (D=8192, codebook=525)
  Expected accuracy >= 95% at current load (250 facts)
  Grounding       >= 93% (projected end-to-end 82%)
  Confidence      calibrated (0.5 = chance), margin 1.6σ
  Max depth (p50) =  13 hops
  Pressure        =  0.45
```

No vector database tells you *before deployment* that adding 300 more
facts will collapse accuracy, or that your bottleneck is the extractor
and not the memory. The Inspector does — every field is a formula of
the theory, not a statistic.

## Why trust the contract

Every claim is a law verified with confidence intervals, and every
falsified prediction is retained in the record:

- **Capacity law** N\* = k·2D/(π·z_G(M)²), k = 0.92 ± 0.03 — accuracy
  predicted within 4.2% at a fixed 1 KB budget, zero fitted parameters.
- **Hop composition** Acc(h) = p^h — proved, not fitted (mean dev 0.02).
- **Resource Composition Law** Acc = E_q[Pg] × Pr(N_eff) — your
  extractor's audited precision composes multiplicatively with the
  memory's predicted accuracy; corroborated against i.i.d., clustered
  and systematic error structures.
- **End-to-end pilot**: documents → compiler → 1000 queries; contract
  issued pre-query with a declared CI; theory error 1.7%.
- **2000 facts live in a 16 KB trace.** Deterministic: same inputs,
  same bits, same answers. Inverse queries come free (edge symmetry).

Full record: [the paper](docs/paper.md) ·
[FORMALISM v2.1](docs/FORMALISM.md) (normative, frozen) ·
[experiment reports](docs/) · raw results as JSON in the repo root.

## Repository layout

- [`reference/abm.py`](reference/abm.py) — the executable specification
  (frozen v1.0.0 against FORMALISM; changes require a version bump)
- [`reference/inspector.py`](reference/inspector.py) — the Memory
  Contract as an API: `stats()`, `contract()`, `report()`, `aliasing()`
- [`reference/test_abm.py`](reference/test_abm.py) — property tests
  against the axioms and theorems
- [`examples/`](examples/) — every experiment behind every number above,
  reproducible; `industrial_pilot.py --extraction your_llm_output.json`
  plugs in a real LLM compiler
- [`bsm/`](bsm/) — the research codebase the theory grew out of
  (encoders, RAG integration, reasoning engine)

## What ABM is not

Not an embedding store, not a vector-DB replacement, not "a better
RAG". It is a different category: a memory with a quantitative resource
theory and predictive contracts on its own behavior — the memory
equivalent of what space/time/bandwidth guarantees are for classical
systems.

## License

MIT
