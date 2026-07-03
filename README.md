# BSM Foundation

**Geometric Memory Platform**

BSM Foundation is a persistent, content-addressable memory that works in
Hamming space.  It is not a language model — it is a geometric structure
for storing and retrieving binary states.

```python
from bsm import BSM

bsm = BSM(encoder="hash", state_dim=256)

# Encode anything (text, embeddings, features)
state = bsm.encode("the cat sat on the mat")

# Store with a payload
bsm.observe(state, {"entity": "cat", "action": "sit"})

# Retrieve by similarity
results = bsm.recall(state, k=3)
for payload, dist, meta in results:
    print(f"  dist={dist}: {payload}")

# Inspect health
print(bsm.info())
print(bsm.health())

# Persist
bsm.save("memory.bsm-store.npz")
```

## Key properties

- **25× less memory than FAISS** at 92.5% recall@1
- **10K entries in < 50 MB**, search in < 100 ms on CPU
- **Zero external dependencies** (numpy only)
- **Deterministic** — same inputs → same states → same retrievals
- **Domain-agnostic** — works with text, code, DNA, images, audio

## Install

```bash
pip install bsm-foundation
```

## Documentation

- [Specification v1.0](docs/SPECIFICATION.md) — the BSM Foundation contract
- [RFCs](docs/rfc/) — protocol definitions and extension points
- [Examples](examples/) — quickstart and integrations

## Benchmarks

```bash
bsm-bench --report md
```

## License

MIT
