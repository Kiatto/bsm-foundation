"""
phase2_demo.py — BSM Phase II end-to-end demonstration.

Pipeline:
  1. Encode text → binary vectors (HashEncoder)
  2. Store binary vectors in MemoryStore
  3. Search MemoryStore via Hamming distance
  4. Route queries with BSMRouter
  5. Report latency / accuracy metrics
"""

import sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bsm.memory.encoder.bsm_encoder import HashEncoder
from bsm.memory.store.memory_store import MemoryStore
from bsm.memory.router.bsm_router import BSMRouter

D = 256


def main():
    print("=" * 60)
    print("BSM Phase II — Cognitive Memory Layer Demo")
    print("=" * 60)

    # ---- 1. Encoder ----
    print("\n[1] Initializing HashEncoder (D=256, seed=42)")
    enc = HashEncoder(state_dim=D, seed=42)

    # Build a small corpus of weather & math texts
    corpus = {
        "weather": [
            "sunny and clear skies today",
            "rain expected in the afternoon",
            "storm warning along the coast",
            "temperature dropping below freezing",
            "cloudy with chance of showers",
        ],
        "math": [
            "solve the quadratic equation x squared plus two",
            "calculate the definite integral from zero to pi",
            "find the derivative of sine x plus cosine x",
            "compute the eigenvalues of the matrix",
            "factor the polynomial x cubed minus one",
        ],
    }

    print(f"\n[2] Encoding {sum(len(v) for v in corpus.values())} texts "
          f"→ binary vectors")
    all_texts = []
    all_labels = []
    for label, texts in corpus.items():
        for text in texts:
            all_texts.append(text)
            all_labels.append(label)

    t0 = time.perf_counter()
    encoded = np.array([enc.encode(t) for t in all_texts])
    encode_time = (time.perf_counter() - t0) * 1000
    print(f"    Encoded {len(all_texts)} texts in {encode_time:.1f} ms "
          f"({encode_time/len(all_texts)*1000:.1f} µs/text)")
    print(f"    Binary vector shape: {encoded.shape}, dtype={encoded.dtype}")

    # ---- 2. Memory Store ----
    print("\n[3] Storing binary vectors in MemoryStore")
    store = MemoryStore(state_dim=D)
    t0 = time.perf_counter()
    store.put_batch(encoded, all_texts,
                    metas=[{"label": l, "ts": time.time()} for l in all_labels])
    store_time = (time.perf_counter() - t0) * 1000
    print(f"    Stored {store.size()} entries in {store_time:.1f} ms")

    # ---- 3. Search ----
    print("\n[4] Searching MemoryStore")
    queries = [
        "heavy rainfall expected tonight",
        "integrate x squared dx",
        "sunny weekend ahead",
        "calculate the limit",
    ]
    for q in queries:
        q_vec = enc.encode(q)
        t0 = time.perf_counter()
        results = store.search(q_vec, k=3)
        latency = (time.perf_counter() - t0) * 1e6
        print(f"    Query: '{q}'")
        for val, dist, meta in results:
            label = meta.get("label", "?")
            print(f"      → {val}  (dist={dist}, label={label})")
        print(f"      latency={latency:.0f} µs")

    # ---- 4. Router ----
    print("\n[5] Initializing BSMRouter")
    router = BSMRouter(state_dim=D)

    # Build prototypes from corpus encodings
    weather_encs = np.array([enc.encode(t) for t in corpus["weather"]])
    math_encs = np.array([enc.encode(t) for t in corpus["math"]])
    w_proto = np.where(weather_encs.mean(axis=0) >= 0, 1, -1).astype(np.int8)
    m_proto = np.where(math_encs.mean(axis=0) >= 0, 1, -1).astype(np.int8)
    router.add_route("weather", w_proto)
    router.add_route("math", m_proto)

    test_queries = [
        ("sunny and warm today", "weather"),
        ("storm approaching coast", "weather"),
        ("solve for x in the equation", "math"),
        ("calculate the derivative", "math"),
        ("cloudy with chance of rain", "weather"),
        ("find the determinant of the matrix", "math"),
    ]
    correct = 0
    print(f"\n[6] Routing {len(test_queries)} test queries:")
    for text, expected in test_queries:
        q_vec = enc.encode(text)
        t0 = time.perf_counter()
        pred, dist = router.route(q_vec)
        latency = (time.perf_counter() - t0) * 1e6
        ok = "✓" if pred == expected else "✗"
        print(f"    {ok} '{text}' → {pred} (expected {expected}, "
              f"dist={dist}, {latency:.0f} µs)")
        if pred == expected:
            correct += 1

    accuracy = correct / len(test_queries)
    print(f"\n    Router accuracy: {accuracy:.1%} ({correct}/{len(test_queries)})")

    # ---- 5. Summary ----
    print(f"\n{'=' * 60}")
    print(f"Phase II Demo Summary")
    print(f"{'=' * 60}")
    print(f"  Encoder:      HashEncoder D={D}")
    print(f"  Store:        {store.size()} entries, {store.state_dim}-bit vectors")
    print(f"  Router:       {len(router.get_routes())} routes")
    print(f"  Routes:       {router.get_routes()}")
    print(f"  Acc (target): {accuracy:.1%} (target > 80%)")
    if accuracy > 0.80:
        print(f"  ✓ SUCCESS: Router accuracy target met!")
    print()


if __name__ == "__main__":
    main()
