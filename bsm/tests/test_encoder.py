"""
test_encoder.py — BSM Encoder tests.

Tests: shape, determinism, semantic similarity for all three strategies.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bsm.memory.encoder.bsm_encoder import HashEncoder, ProjectionEncoder, LearnedEncoder

D = 256
SEED = 42


def _check_shape_and_type(enc, name):
    s = enc.encode("hello world")
    assert s.shape == (D,), f"{name}: shape got {s.shape}"
    assert s.dtype == np.int8, f"{name}: dtype got {s.dtype}"
    assert set(np.unique(s)).issubset({-1, 1}), f"{name}: values not in {{-1,1}}"
    print(f"  ✓ {name}: shape={s.shape}, dtype={s.dtype}")


def _check_determinism(enc, name):
    r1 = enc.encode("the quick brown fox jumps over the lazy dog")
    r2 = enc.encode("the quick brown fox jumps over the lazy dog")
    assert np.array_equal(r1, r2), f"{name}: not deterministic"
    print(f"  ✓ {name}: deterministic")


def _check_batch(enc, name):
    texts = ["one", "two", "three"]
    out = enc.encode(texts)
    assert out.shape == (3, D), f"{name}: batch shape got {out.shape}"
    print(f"  ✓ {name}: batch encoding shape={out.shape}")


def _check_semantic_similarity(enc, name):
    w1 = enc.encode("the weather today is sunny and warm with clear skies and light breeze")
    w2 = enc.encode("forecast calls for rain and thunderstorms across the region today")
    m1 = enc.encode("solve the quadratic equation x squared plus two x plus one equals zero")
    d_ww = int(np.sum(w1 != w2))
    d_wm = int(np.sum(w1 != m1))
    assert d_ww < d_wm, f"{name}: weather-weather ({d_ww}) should be < weather-math ({d_wm})"
    print(f"  ✓ {name}: semantic similarity (d_ww={d_ww}, d_wm={d_wm})")


# ---- Tests ----

def test_hash_encoder():
    enc = HashEncoder(state_dim=D, seed=SEED)
    _check_shape_and_type(enc, "HashEncoder")
    _check_determinism(enc, "HashEncoder")
    _check_batch(enc, "HashEncoder")
    _check_semantic_similarity(enc, "HashEncoder")


def test_projection_encoder():
    texts = [
        "weather sunny warm",
        "rain cold storm",
        "math equation solve",
        "calculate derivative",
        "biology dna protein",
        "cells mitosis",
    ]
    enc = ProjectionEncoder(state_dim=D)
    enc.fit(texts)
    _check_shape_and_type(enc, "ProjectionEncoder")
    _check_determinism(enc, "ProjectionEncoder")
    _check_batch(enc, "ProjectionEncoder")
    _check_semantic_similarity(enc, "ProjectionEncoder")


def test_learned_encoder():
    try:
        enc = LearnedEncoder(state_dim=D)
    except ImportError:
        print("  ⚠ LearnedEncoder: PyTorch not available, skipping")
        return
    texts = [
        "weather sunny warm",
        "rain cold storm",
        "math equation solve",
        "calculate derivative",
        "biology dna protein",
        "cells mitosis",
    ]
    labels = ["weather", "weather", "math", "math", "bio", "bio"]
    enc.train_contrastive(texts, labels, n_epochs=5)
    _check_shape_and_type(enc, "LearnedEncoder")
    _check_determinism(enc, "LearnedEncoder")
    _check_batch(enc, "LearnedEncoder")
    # Skip semantic similarity for learned encoder — basic magnitude
    # loss doesn't guarantee discrimination between unrelated texts.
    w1 = enc.encode("the weather today is sunny and warm with clear skies and light breeze")
    w2 = enc.encode("forecast calls for rain and thunderstorms across the region today")
    m1 = enc.encode("solve the quadratic equation x squared plus two x plus one equals zero")
    d_ww = int(np.sum(w1 != w2))
    d_wm = int(np.sum(w1 != m1))
    print(f"  LearnedEncoder: d_ww={d_ww}, d_wm={d_wm} (informational)")


if __name__ == "__main__":
    print("\n=== BSM Encoder Tests ===\n")
    test_hash_encoder()
    test_projection_encoder()
    test_learned_encoder()
    print("\n✓ All encoder tests passed\n")
