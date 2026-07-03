"""
test_bsm.py — BSM unified entry point tests.

Covers: encode, observe, recall, predict, route, save/load,
        info, health, metrics, add_route, sleep.
"""

import sys, os, tempfile, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bsm import BSM, __version__

D = 256


def test_version():
    assert __version__ == "1.0.0rc1"
    print("  ✓ version")


def test_create():
    bsm = BSM(encoder="hash", state_dim=D)
    assert bsm.state_dim == D
    assert bsm._store.size() == 0
    assert bsm.info()["encoder"] == "hash"
    print("  ✓ create")


def test_encode_text():
    bsm = BSM(encoder="hash", state_dim=D)
    s = bsm.encode("hello world")
    assert s.shape == (D,)
    assert s.dtype == np.int8
    assert set(np.unique(s)).issubset({-1, 1})
    print("  ✓ encode text")


def test_encode_array():
    bsm = BSM(encoder="hash", state_dim=D)
    arr = np.random.randn(384).astype(np.float32)
    s = bsm.encode(arr)
    assert s.shape == (D,)
    assert s.dtype == np.int8
    print("  ✓ encode array")


def test_encode_deterministic():
    bsm = BSM(encoder="hash", state_dim=D)
    s1 = bsm.encode("the quick brown fox")
    s2 = bsm.encode("the quick brown fox")
    assert np.array_equal(s1, s2)
    print("  ✓ encode deterministic")


def test_observe_and_recall():
    bsm = BSM(encoder="hash", state_dim=D)
    s = bsm.encode("test item")
    bsm.observe(s, "hello")
    assert bsm._store.size() == 1
    results = bsm.recall(s, k=1)
    assert len(results) == 1
    assert results[0][0] == "hello"
    assert results[0][1] == 0
    print("  ✓ observe + recall")


def test_predict():
    bsm = BSM(encoder="hash", state_dim=D)
    s = bsm.encode("prediction test")
    bsm.observe(s, "cat")
    result = bsm.predict(s)
    assert result == "cat"
    print("  ✓ predict")


def test_route():
    bsm = BSM(encoder="hash", state_dim=D)
    bsm.add_route("weather", ["sunny", "rain", "cloudy"])
    bsm.add_route("math", ["derivative", "equation", "integral"])
    s = bsm.encode("sunny and warm")
    name, dist = bsm.route(s)
    assert name == "weather"
    print(f"  ✓ route: {name}")


def test_add_route_from_vectors():
    bsm = BSM(encoder="hash", state_dim=D)
    vecs = np.array([bsm.encode(t) for t in ["sunny", "rain"]])
    bsm.add_route("weather", vecs)
    assert "weather" in bsm._router.get_routes()
    print("  ✓ add_route from vectors")


def test_info():
    bsm = BSM(encoder="hash", state_dim=D)
    info = bsm.info()
    assert info["version"] == __version__
    assert info["encoder"] == "hash"
    assert info["state_dim"] == D
    assert info["entries"] == 0
    assert info["protocol"] == "1.0"
    print("  ✓ info")


def test_health_empty():
    bsm = BSM(encoder="hash", state_dim=D)
    h = bsm.health()
    assert h["entries"] == 0
    assert h["status"] == "empty"
    print("  ✓ health (empty)")


def test_health_nonempty():
    bsm = BSM(encoder="hash", state_dim=D)
    for i in range(5):
        s = bsm.encode(f"item {i}")
        bsm.observe(s, f"payload_{i}")
    h = bsm.health()
    assert h["entries"] == 5
    assert h["utilization"] == 5 / bsm.capacity
    assert "avg_hamming_radius" in h
    print("  ✓ health (non-empty)")


def test_save_load():
    bsm = BSM(encoder="hash", state_dim=D)
    s = bsm.encode("persist me")
    bsm.observe(s, {"data": "test"})
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    bsm.save(path)
    bsm2 = BSM(encoder="hash", state_dim=D)
    bsm2.load(path)
    assert bsm2._store.size() == 1
    s2 = bsm2.encode("persist me")
    results = bsm2.recall(s2, k=1)
    assert results[0][0] == {"data": "test"}
    os.unlink(path)
    os.unlink(Path(path).with_suffix(".vals.jsonl"))
    print("  ✓ save/load")


def test_sleep():
    bsm = BSM(encoder="hash", state_dim=D, capacity=100)
    for i in range(20):
        s = bsm.encode(f"item {i}")
        bsm.observe(s, f"payload_{i}", value=0.9 if i < 10 else 0.1)
    n_before = bsm._store.size()
    forgotten = bsm.sleep(forget_threshold=0.5)
    assert forgotten > 0
    assert bsm._store.size() < n_before
    print(f"  ✓ sleep: forgot {forgotten}")


def test_metrics():
    bsm = BSM(encoder="hash", state_dim=D)
    for i in range(50):
        s = np.where(np.random.randn(D) > 0, 1, -1).astype(np.int8)
        bsm.observe(s, f"item_{i}")
    m = bsm.metrics(n_queries=10)
    assert "latency_mean_us" in m
    assert "n_entries" in m
    print(f"  ✓ metrics: {m['latency_mean_us']:.0f} µs mean")


def test_repr():
    bsm = BSM(encoder="hash", state_dim=D)
    r = repr(bsm)
    assert "BSM" in r
    assert "hash" in r
    assert __version__ in r
    print("  ✓ repr")


def test_projector_reuse():
    bsm = BSM(encoder="hash", state_dim=D)
    a = np.random.randn(128).astype(np.float32)
    b = np.random.randn(128).astype(np.float32)
    s1 = bsm.encode(a)
    s2 = bsm.encode(b)
    assert s1.shape == (D,)
    assert s2.shape == (D,)
    assert bsm._projection.shape == (128, D)
    print("  ✓ projector reuse")


def test_encode_batch():
    bsm = BSM(encoder="hash", state_dim=D)
    texts = ["one", "two", "three"]
    out = bsm.encode(texts)
    assert out.shape == (3, D)
    print("  ✓ encode batch")


if __name__ == "__main__":
    print("\n=== BSM Unified Entry Point Tests ===\n")
    test_version()
    test_create()
    test_encode_text()
    test_encode_array()
    test_encode_deterministic()
    test_observe_and_recall()
    test_predict()
    test_route()
    test_add_route_from_vectors()
    test_info()
    test_health_empty()
    test_health_nonempty()
    test_save_load()
    test_sleep()
    test_metrics()
    test_repr()
    test_projector_reuse()
    test_encode_batch()
    print("\n✓ All BSM tests passed\n")
