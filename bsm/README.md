# BSM Foundation 1.0

Binary State Model Foundation — a geometric memory layer for AI systems.

## Structure

```
bsm/
├── manifesto.md          # 6 principles, definitions, scope
├── index.md              # experiment index & results
├── core/
│   └── memory_engine.py   # Geometric content-addressable memory
├── benchmark/
│   └── mib.py             # Memory Intelligence Benchmark
└── experiments/
    ├── phases_1_4.py      # BSM-X: weighted Hamming, ablation, etc.
    ├── mi_analysis.py     # MI distribution, Gini
    ├── pairwise_mi.py     # Pairwise MI analysis
    ├── shuffle_test.py    # Working bank invariance test
    ├── bank_probe.py      # Logistic regression bank probes
    ├── dynamics.py        # State dynamics (falsified)
    ├── scaling.py         # D=128..1024 dimension scaling
    ├── intrinsic_dim.py   # Participation Ratio estimation
    ├── pr_verify.py       # PR binary vs raw verification
    ├── cross_domain.py    # 5 domains: stories, wiki, code, DNA, random
    ├── kd_analysis.py     # Knowledge Density metric
    └── augmentation.py    # LSTM + BSM augmentation demo
```

## Core API

### MemoryEngine — geometric content-addressable memory

```python
from bsm.core.memory_engine import MemoryEngine

mem = MemoryEngine(state_dim=128, capacity=100000)
mem.observe(state, experience)   # store
mem.recall(state, top_k=4)       # retrieve
mem.predict(state)               # retrieve & vote
mem.plan(state, n_steps=5)       # simulate trajectory
mem.sleep()                      # forget + consolidate
mem.dream(n_steps=100)           # explore via random walks
mem.reflect()                    # diagnostics
```

### BSMAugment — LLM memory layer

```python
from bsm.core.augment import BSMAugment

# Wrap any LLM with BSM memory
aug = BSMAugment(llm_hidden_dim=768)
aug.create_memory(capacity=50000)

# Store observations during inference
aug.observe(llm_model, input_ids, target_token)

# Augment predictions with memory
prediction = aug.predict(llm_model, input_ids)
# → returns token, potentially overridden by memory recall
```

### LLMAdapter — convert LLM hidden states to binary states

```python
from bsm.core.llm_adapter import LLMAdapter

adapter = LLMAdapter(llm_hidden_dim=768, state_dim=128)
_, binary_state = adapter(llm_hidden_tensor)  # [B, 768] → [B, 128]
```

### MIB — Memory Intelligence Benchmark

```python
from bsm.benchmark.mib import MIB

bench = MIB(n_train=5000, n_test=500)
results = bench.evaluate(memory)
bench.report(results)
# → 7 metrics, composite MIB score
```

## Quick Reference

| Law | Finding | Value |
|-----|---------|-------|
| 1 | Semantic Hamming geometry | P(d=0) ≈ 0.75 |
| 2 | Retrieval > Decoder | 1.6× (text only) |
| 3 | Hierarchical MI | Gini = 0.34 |
| 4 | Decorrelated banks | ratio ≈ 1.2× |
| 5 | Local > Global | dynamics fail |
| 6 | Intrinsic dimension | PR ≈ 33±4 |

## Key Numbers

- **Decoder**: 138 KB → 5.70% accuracy
- **Decoder + GAM**: 992 KB → 13.03% accuracy
- **KD**: 1.23 bits/MB
- **LSTM + BSM**: 1.65 pp improvement over LSTM alone
- **Latency**: ~400 µs/query
- **Memory**: ~1 MB total for 50K-state GAM
- **Benchmark**: MIB — 7 dimensions, composite score

## Freeze

BSM Foundation 1.0 is frozen. No further modifications.
New research → BSM 2.x (separate branch/phase).
