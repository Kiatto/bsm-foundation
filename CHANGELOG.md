# Changelog

## 1.0.0rc1 (2026-07-03)

- Unified `BSM` entry point (encode, observe, recall, predict, route, sleep)
- Three encoders: Hash, Projection, Learned
- Memory Store with POPCOUNT Hamming search
- Router with prototype-based classification
- Lifecycle: sleep (consolidate/forget), health checks
- Metrics: info(), health(), metrics()
- Persistence: save() / load() (`.bsm-store.npz` format)
- BSM-Bench: `bsm-bench` CLI for reproducible benchmarks
- Formal Specification v1.0 (`docs/SPECIFICATION.md`)
- 6 initial RFCs (`docs/rfc/RFC-0001` through `RFC-0006`)
- Foundation 1.0 experiments sealed in `bsm/core/`
