"""
test_vsa.py — Tests per l'algebra binaria (vsa.py).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from bsm.memory.vsa import (random_hv, bind_xor, permute, bundle, hamming,
                            ItemMemory, RoleProjection,
                            WorkingMemory, SemanticMemory)
from bsm.memory.encoder.entity_encoder import _minhash_sketch


class TestAlgebra:
    def test_bind_is_self_inverse(self):
        a, b = random_hv("a", 512), random_hv("b", 512)
        assert np.array_equal(bind_xor(bind_xor(a, b), b), a)

    def test_bound_is_dissimilar_from_operands(self):
        a, b = random_hv("a", 512), random_hv("b", 512)
        c = bind_xor(a, b)
        assert hamming(c, a) > 512 * 0.35
        assert hamming(c, b) > 512 * 0.35

    def test_bind_preserves_distances(self):
        a, a2, r = (random_hv("a", 512), random_hv("a2", 512),
                    random_hv("r", 512))
        assert hamming(a, a2) == hamming(bind_xor(a, r), bind_xor(a2, r))

    def test_bundle_similar_to_members(self):
        xs = [random_hv(f"x{i}", 512) for i in range(3)]
        t = bundle(xs)
        stranger = random_hv("stranger", 512)
        for x in xs:
            assert hamming(t, x) < hamming(t, stranger)

    def test_permute_marks_position(self):
        a = random_hv("a", 512)
        assert hamming(a, permute(a, 1)) > 512 * 0.35
        assert np.array_equal(permute(permute(a, 1), -1), a)

    def test_random_hv_null_distribution(self):
        # Due hv indipendenti distano ~D/2 (base della confidence calibrata)
        d = hamming(random_hv("x", 4096), random_hv("y", 4096))
        assert abs(d - 2048) < 200


class TestWorkingMemory:
    def test_holographic_recall_small(self):
        wm = WorkingMemory(1024)
        facts = [("android", "dev", "google"), ("iphone", "dev", "apple"),
                 ("windows", "dev", "microsoft")]
        for f in facts:
            wm.store(*f)
        for s, r, o in facts:
            assert wm.query(s, r)[0] == o

    def test_two_hop_pure_algebra(self):
        wm = WorkingMemory(1024)
        wm.store("android", "dev", "google")
        wm.store("google", "hq", "mountain view")
        bridge, _ = wm.query("android", "dev")
        answer, _ = wm.query(bridge, "hq")
        assert answer == "mountain view"

    def test_capacity_degrades_gracefully(self):
        wm = WorkingMemory(1024)
        n = 60
        facts = [(f"s{i}", f"r{i % 5}", f"o{i}") for i in range(n)]
        for f in facts:
            wm.store(*f)
        ok = sum(1 for s, r, o in facts if wm.query(s, r)[0] == o)
        assert ok > n * 0.5, f"degrado troppo brusco: {ok}/{n}"


class TestSemanticMemory:
    NAMES = ["google llc", "microsoft corporation", "apple inc",
             "amazon web services", "tesla inc", "netflix inc"]

    def _fill(self, sm, d=1024):
        for i, name in enumerate(self.NAMES):
            sm.store(_minhash_sketch(set(name.split()), d), "hq",
                     f"city{i}", s_name=name)
        return sm

    def test_xor_addressing_exact(self):
        sm = self._fill(SemanticMemory(1024, binding="xor"))
        for i, name in enumerate(self.NAMES):
            got, d, _ = sm.query(_minhash_sketch(set(name.split()), 1024), "hq")
            assert got == f"city{i}"
            assert d == 0

    def test_fitted_projection_generalizes_alias(self):
        states = [_minhash_sketch(set(n.split()), 1024) for n in self.NAMES]
        sm = self._fill(SemanticMemory(1024, binding="proj").fit(states, 12))
        got, _, _ = sm.query(_minhash_sketch({"google"}, 1024), "hq")
        assert got == "city0"

    def test_fitted_projection_widens_margin(self):
        states = [_minhash_sketch(set(n.split()), 1024) for n in self.NAMES]
        sm_x = self._fill(SemanticMemory(1024, binding="xor"))
        sm_p = self._fill(SemanticMemory(1024, binding="proj").fit(states, 12))
        q = _minhash_sketch({"google", "llc"}, 1024)
        _, _, m_x = sm_x.query(q, "hq")
        _, _, m_p = sm_p.query(q, "hq")
        assert m_p > m_x, f"margine proj {m_p} <= xor {m_x}"


class TestRoleProjection:
    def test_lossy_after_fit(self):
        states = [random_hv(f"e{i}", 256) for i in range(20)]
        p = RoleProjection("subj", 256).fit(states, rank=8)
        assert p.rank == 8

    def test_deterministic(self):
        a = RoleProjection("subj", 256).apply(random_hv("x", 256))
        b = RoleProjection("subj", 256).apply(random_hv("x", 256))
        assert np.array_equal(a, b)

    def test_different_roles_diverge(self):
        x = random_hv("x", 256)
        a = RoleProjection("subj", 256).apply(x)
        b = RoleProjection("obj", 256).apply(x)
        assert hamming(a, b) > 256 * 0.25


class TestItemMemory:
    def test_cleanup_recovers_noisy_state(self):
        im = ItemMemory(1024)
        for n in ("google", "apple", "tesla"):
            im.add(n)
        noisy = im.get("google").copy()
        idx = np.arange(0, 1024, 4)
        noisy[idx] = -noisy[idx]           # 25% di rumore
        name, _, _ = im.cleanup(noisy)
        assert name == "google"
