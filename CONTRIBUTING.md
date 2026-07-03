# Contributing to BSM Foundation

## How to contribute

1. Read the [Specification](docs/SPECIFICATION.md) first.
2. Check existing RFCs in `docs/rfc/`.
3. Open an issue to discuss changes before writing code.
4. Submit PRs against the `main` branch.

## Development

```bash
pip install -e .
pip install -e ".[torch,bench]"

# Run tests
python -m pytest bsm/tests/

# Run benchmark
bsm-bench --quick

# Single test file
python bsm/tests/test_encoder.py
```

## RFC process

Significant changes require an RFC:

1. Copy `docs/rfc/RFC-0000.md.template` (or use an existing RFC as template).
2. Assign the next available RFC number.
3. Submit as a PR with `[RFC]` prefix.
4. Discuss and iterate.
5. Once approved, implement.

## Code conventions

- No comments unless the "why" is non-obvious.
- Type hints required for all public APIs.
- Test coverage > 90 % for new code.
- Benchmark before and after performance changes.

## Guiding principles

- Memory is geometric, not statistical.
- Encoders plug in; the Core doesn't change.
- Small dependencies, small memory, small API.
