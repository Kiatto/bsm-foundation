# ABM SDK — Developer Guide

No theory required. Five concepts, one CLI.

## Install

```bash
pip install abm-runtime
```

## The five concepts

```python
from abm import Memory, contract

mem = Memory(dim=8192)          # one memory = one binary trace

mem.store("payment_service", "requires", "auth_service")

answer, confidence = mem.query("payment_service", "requires")
# → ("auth_service", 0.99)     confidence: 0.5 = random guess

answer, confidence = mem.chain("payment_service",
                               ["requires", "writes_to"])
# multi-hop: follow relations step by step

mem.member("payment_service", "requires", "auth_service")
# → True/False: is this exact fact in memory?

print(contract(mem, grounding=0.93))
# the Memory Contract: how this memory WILL behave, computed
# before you run a single query
```

## The Memory Contract

Every ABM memory can state, in advance:

```
MEMORY CONTRACT
  Capacity        <= 944 facts (D=8192, codebook=74)
  Expected accuracy >= 96% at current load (180 facts)
  Grounding       >= 93% (projected end-to-end 89%)
  Confidence      calibrated (0.5 = chance), margin 3.1σ
  Max depth (p50) =  7 hops
  Pressure        =  0.19
```

- **Capacity / Pressure** — how full the memory is, and when it will
  start degrading. Pressure > 1 means overloaded: use a bigger `dim`.
- **Expected accuracy** — predicted single-query accuracy at the
  current load. Not a benchmark: a computation.
- **Grounding / projected end-to-end** — pass the audited precision of
  your extractor and get the end-to-end projection.
- **Max depth** — how many hops you can chain before accuracy drops
  below 50%.

If the contract says 88% and you measure 60%, file a bug — that is the
product promise.

## CLI

```bash
abm demo                                  # 30-second tour
abm inspect triples.json --grounding 0.9  # contract for your data
```

`triples.json` is a JSON array of `[subject, relation, object]` —
the natural output of an LLM extractor.

## Examples

- [examples/sdk/01_faq.py](../examples/sdk/01_faq.py) — store facts, ask questions
- [examples/sdk/02_knowledge_base.py](../examples/sdk/02_knowledge_base.py) — a KB with its contract
- [examples/sdk/03_llm_memory.py](../examples/sdk/03_llm_memory.py) — document → triples → ABM → queries

## Practical notes

- **Deterministic**: same facts in, same bits, same answers. Always.
- **Sizing**: don't guess `dim` — run `abm inspect` and use the
  `recommended_dimension` it prints.
- **Multi-valued relations** (one subject, many objects): `query`
  returns the strongest one; use `member` to test specific candidates.
- **Inverse queries are free**: a stored fact answers in both
  directions.
- **What ABM is not**: an embedding store or a fuzzy semantic search.
  Entities are exact names — normalize them at ingestion.
