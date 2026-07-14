"""
test_hybrid.py — Tests per il layer ibrido geometrico:
confidence calibrata, EntityEncoder MinHash, GraphCache geometrico,
PrototypeIndex, pesi RRF adattivi.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from bsm import BSM
from bsm.memory.encoder.bsm_encoder import ProjectionEncoder
from bsm.memory.encoder.entity_encoder import EntityEncoder, _minhash_sketch
from bsm.memory.ensemble import EnsembleRetriever
from bsm.memory.reasoning_engine import ReasoningEngine, calibrated_confidence
from bsm.memory.graph_cache import GraphCache
from bsm.memory.prototypes import PrototypeIndex


KNOWLEDGE_BASE = [
    "Apple Inc. manufactures the iPhone smartphone.",
    "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
    "Microsoft Corporation develops the Windows operating system.",
    "Microsoft Corporation was founded by Bill Gates in 1975.",
    "Google LLC developed the Android operating system.",
    "Google LLC is headquartered in Mountain View, California.",
    "Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius.",
    "The speed of light in vacuum is 299,792 kilometers per second.",
]


# ---------------------------------------------------------------------------
# 1. Confidence calibrata
# ---------------------------------------------------------------------------

class TestCalibratedConfidence:
    def test_chance_level_is_half(self):
        # Due stati casuali distano D/2: indistinguibile dal rumore
        assert abs(calibrated_confidence(128, 256) - 0.5) < 1e-9

    def test_monotone_decreasing_in_distance(self):
        confs = [calibrated_confidence(d, 256) for d in range(0, 257, 16)]
        assert all(a > b for a, b in zip(confs, confs[1:]))

    def test_bounds(self):
        assert 0.0 < calibrated_confidence(256, 256) < 0.5
        assert 0.5 < calibrated_confidence(0, 256) < 1.0

    def test_strong_match_is_confident(self):
        # dist = D/4 → z = 8 deviazioni standard sotto il caso
        assert calibrated_confidence(64, 256) > 0.7


# ---------------------------------------------------------------------------
# 2. EntityEncoder MinHash: Jaccard ↔ Hamming
# ---------------------------------------------------------------------------

class TestMinHashEntityEncoder:
    def test_hamming_estimates_jaccard(self):
        # Insiemi con Jaccard noto: J({a,b},{a,b}) = 1 → dist 0;
        # J({a,b},{c,d}) = 0 → dist ≈ D/2
        d = 512
        s_ab = _minhash_sketch({"apple", "banana"}, d)
        s_ab2 = _minhash_sketch({"apple", "banana"}, d)
        s_cd = _minhash_sketch({"cherry", "date"}, d)
        assert np.array_equal(s_ab, s_ab2), "MinHash deve essere deterministico"
        dist_disjoint = np.count_nonzero(s_ab != s_cd)
        assert abs(dist_disjoint - d / 2) < 0.15 * d, \
            f"Insiemi disgiunti dovrebbero distare ~D/2, got {dist_disjoint}"

    def test_overlap_reduces_distance(self):
        d = 512
        s1 = _minhash_sketch({"apple", "google", "microsoft"}, d)
        s2 = _minhash_sketch({"apple", "google", "tesla"}, d)      # J = 0.5
        s3 = _minhash_sketch({"netflix", "amazon", "meta"}, d)     # J = 0
        d_similar = np.count_nonzero(s1 != s2)
        d_disjoint = np.count_nonzero(s1 != s3)
        assert d_similar < d_disjoint, \
            f"J=0.5 deve distare meno di J=0: {d_similar} vs {d_disjoint}"
        # E[dist | J=0.5] = D*(1-J)/2 = D/4
        assert abs(d_similar - d / 4) < 0.15 * d

    def test_encoder_contract(self):
        # Stesso contratto degli altri encoder: int8 {-1,+1} a state_dim
        enc = EntityEncoder(state_dim=256)
        state = enc.encode("Apple Inc. was founded by Steve Jobs")
        assert state.dtype == np.int8
        assert state.shape == (256,)
        assert set(np.unique(state)) <= {-1, 1}

    def test_usable_in_standard_bsm(self):
        enc = EntityEncoder(state_dim=256)
        bsm = BSM(encoder=enc, state_dim=256)
        for doc in KNOWLEDGE_BASE:
            bsm.observe(bsm.encode(doc), {"text": doc})
        results = bsm.recall(bsm.encode("iPhone Apple"), k=3)
        texts = [r[0]["text"] for r in results]
        assert any("iPhone" in t for t in texts), texts

    def test_no_entity_fallback_deterministic(self):
        enc = EntityEncoder(state_dim=256)
        a = enc.encode("water freezes at zero degrees")
        b = enc.encode("water freezes at zero degrees")
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# 3. GraphCache geometrico
# ---------------------------------------------------------------------------

class TestGeometricGraphCache:
    def _cache(self):
        return GraphCache(encoder=EntityEncoder(state_dim=256))

    def test_alias_lookup(self):
        gc = self._cache()
        gc.store(query="q", entity="Google LLC", bridge_entity="Android",
                 bridge_chunk="c", answer_payload={"text": "Mountain View"},
                 score=0.9)
        # Lookup con una variante del nome: stessa palla di Hamming
        edges = gc.lookup_entity("Google")
        assert edges is not None, "Alias 'Google' deve trovare 'Google LLC'"
        assert edges[0].answer_payload["text"] == "Mountain View"

    def test_unrelated_entity_misses(self):
        gc = self._cache()
        gc.store(query="q", entity="Google LLC", bridge_entity="Android",
                 bridge_chunk="c", answer_payload={"text": "Mountain View"},
                 score=0.9)
        assert gc.lookup_entity("Netflix Streaming Service") is None

    def test_geometric_merge(self):
        gc = self._cache()
        gc.store(query="q1", entity="Android", bridge_entity="Google LLC",
                 bridge_chunk="c", answer_payload={"text": "A"}, score=0.9)
        gc.store(query="q2", entity="Android", bridge_entity="Google",
                 bridge_chunk="c", answer_payload={"text": "A"}, score=0.85)
        gc.sleep(decay=False, merge=True, forget=False)
        edges = gc._graph.get("android", [])
        assert len(edges) == 1, \
            f"Target simili in Hamming devono fondersi, got {len(edges)}"

    def test_default_behavior_unchanged(self):
        gc = GraphCache()  # senza encoder: lookup esatto come prima
        gc.store(query="q", entity="Google LLC", bridge_entity="Android",
                 bridge_chunk="c", answer_payload={"text": "MV"}, score=0.9)
        assert gc.lookup_entity("Google") is None
        assert gc.lookup_entity("Google LLC") is not None


# ---------------------------------------------------------------------------
# 4. PrototypeIndex
# ---------------------------------------------------------------------------

class TestPrototypeIndex:
    def _bsm(self):
        enc = ProjectionEncoder(state_dim=256)
        enc.fit(KNOWLEDGE_BASE)
        bsm = BSM(encoder=enc, state_dim=256)
        for doc in KNOWLEDGE_BASE:
            bsm.observe(bsm.encode(doc), {"text": doc})
        return bsm

    def test_build_covers_all_memories(self):
        bsm = self._bsm()
        idx = PrototypeIndex(bsm, radius_frac=0.35)
        stats = idx.build()
        assert stats["memories"] == len(KNOWLEDGE_BASE)
        assert 1 <= stats["prototypes"] <= len(KNOWLEDGE_BASE)

    def test_centroid_is_majority_vote(self):
        states = [np.array([1, 1, -1, -1], dtype=np.int8),
                  np.array([1, -1, -1, -1], dtype=np.int8),
                  np.array([1, 1, 1, -1], dtype=np.int8)]
        c = PrototypeIndex._majority_vote(states)
        assert c.tolist() == [1, 1, -1, -1]

    def test_hierarchical_recall_matches_flat(self):
        bsm = self._bsm()
        idx = PrototypeIndex(bsm, radius_frac=0.35)
        idx.build()
        for query in ("iPhone Apple", "Windows Microsoft", "Android Google"):
            state = bsm.encode(query)
            flat = bsm.recall(state, k=1)
            hier = idx.recall(state, k=1, n_probe=3)
            assert hier, f"recall gerarchico vuoto per {query!r}"
            assert hier[0][1] == flat[0][1], \
                f"top-1 divergente per {query!r}: {hier[0]} vs {flat[0]}"

    def test_exact_distances_preserved(self):
        bsm = self._bsm()
        idx = PrototypeIndex(bsm)
        idx.build()
        state = bsm.encode(KNOWLEDGE_BASE[0])
        results = idx.recall(state, k=1, n_probe=len(idx._centroids))
        assert results[0][1] == 0, "Un ricordo esatto deve distare 0"


# ---------------------------------------------------------------------------
# 5. Pesi RRF adattivi
# ---------------------------------------------------------------------------

class TestAdaptiveRRF:
    def _ensemble(self):
        ens = EnsembleRetriever(state_dim=256)
        ens.fit(KNOWLEDGE_BASE)
        for doc in KNOWLEDGE_BASE:
            ens.observe(doc, {"text": doc})
        return ens

    def test_weights_start_uniform(self):
        ens = self._ensemble()
        assert all(abs(w - 1.0) < 1e-9 for w in ens.weights.values())

    def test_positive_reward_shifts_weights(self):
        ens = self._ensemble()
        answer = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
        for _ in range(5):
            ens.reward("Who founded Apple?", answer, correct=True)
        # I pesi si sono differenziati ma la media resta 1
        assert max(ens.weights.values()) > min(ens.weights.values())
        mean = sum(ens.weights.values()) / len(ens.weights)
        assert abs(mean - 1.0) < 1e-6

    def test_recall_still_works_after_reward(self):
        ens = self._ensemble()
        answer = "Apple Inc. manufactures the iPhone smartphone."
        ens.reward("What makes the iPhone?", answer, correct=True)
        results = ens.recall("iPhone", k=3)
        texts = [r[0]["text"] for r in results]
        assert any("iPhone" in t for t in texts)

    def test_weights_never_collapse(self):
        ens = self._ensemble()
        answer = "Apple Inc. manufactures the iPhone smartphone."
        for _ in range(50):
            ens.reward("iPhone", answer, correct=False)
        assert all(w > 0 for w in ens.weights.values())
