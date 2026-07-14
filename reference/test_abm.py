"""
test_abm.py — Property tests della reference implementation contro gli
assiomi e i teoremi di FORMALISM.md v2.0. Se questi test passano,
l'implementazione è conforme alla specifica.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from abm import (Memory, ItemMemory, bind, bundle, permute, random_hv,
                 hamming, phi, confidence, capacity, predicted_accuracy)

D = 1024


class TestAxiomA1:
    def test_involution(self):
        x, k = random_hv("x", D), random_hv("k", D)
        assert np.array_equal(bind(bind(x, k), k), x)

    def test_isometry(self):
        x, y, k = (random_hv(n, D) for n in "xyk")
        assert hamming(x, y) == hamming(bind(x, k), bind(y, k))

    def test_rho_homomorphism(self):
        x, y = random_hv("x", D), random_hv("y", D)
        assert np.array_equal(permute(bind(x, y)),
                              bind(permute(x), permute(y)))


class TestAxiomA2:
    def test_member_correlation(self):
        xs = [random_hv(f"m{i}", 8192) for i in range(9)]
        t = bundle(xs)
        rho = np.mean([1 - 2 * hamming(t, x) / 8192 for x in xs])
        expected = np.sqrt(2 / (np.pi * 9))
        assert abs(rho - expected) < 0.05

    def test_variational(self):
        # bundle = argmax of sum phi (bitwise): flipping any bit lowers it
        xs = [random_hv(f"v{i}", 256) for i in range(5)]
        t = bundle(xs)
        base = sum(phi(t, x) for x in xs)
        t2 = t.copy(); t2[0] = -t2[0]
        assert sum(phi(t2, x) for x in xs) <= base


class TestAxiomA3:
    def test_idempotence(self):
        im = ItemMemory(D)
        for n in ("a", "b", "c"):
            im.add(n)
        noisy = im.get("a").copy(); noisy[:100] = -noisy[:100]
        n1, _ = im.cleanup(noisy)
        n2, d2 = im.cleanup(im.get(n1))
        assert n1 == n2 and d2 == 0


class TestTheorems:
    def test_equivariance(self):
        # cleanup_{C+k}(x+k) == cleanup_C(x)+k  (Thm 2.7, via isometry)
        im = ItemMemory(D)
        states = [im.add(n) for n in ("a", "b", "c")]
        k = random_hv("k", D)
        noisy = states[0].copy(); noisy[:80] = -noisy[:80]
        base, _ = im.cleanup(noisy)
        shifted = [bind(s, k) for s in states]
        d = [hamming(bind(noisy, k), s) for s in shifted]
        assert states[int(np.argmin(d))] is im.get(base)

    def test_bridge_elimination(self):
        m = Memory(D)
        f1 = m.fact_hv("a", "r1", "b")
        f2 = m.fact_hv("b", "r2", "c")
        composed = m.compose(f1, f2)
        # exact: contains no b at all
        expected = bind(m.key("a", "r1"),
                        bind(permute(m.items.get("r2"), 1), m.items.get("c")))
        assert np.array_equal(composed, expected)
        assert abs(phi(composed, m.items.get("b"))) < 3

    def test_trace_self_cancellation(self):
        m = Memory(D)
        m.store("a", "r", "b"); m.store("c", "r", "d")
        raw = bind(m._trace, m.key("a", "r"))          # noisy b
        second = bind(m._trace, bind(raw, permute(m.items.add("r2"), 1)))
        # T xor T = 1: the result is trace-free, pure key material
        expected = bind(m.key("a", "r"), permute(m.items.get("r2"), 1))
        assert np.array_equal(second, expected)

    def test_hop_composition(self):
        m = Memory(2048)
        m.store("x", "r1", "y"); m.store("y", "r2", "z")
        for i in range(40):
            m.store(f"p{i}", f"pr{i}", f"po{i}")
        node, conf = m.chain("x", ["r1", "r2"])
        assert node == "z" and conf > 0.25

    def test_capacity_contract(self):
        # predicted accuracy within 10 points of measured at mid load
        m = Memory(2048)
        n = 120
        facts = [(f"s{i}", f"r{i % 11}", f"o{i}") for i in range(n)]
        for f in facts:
            m.store(*f)
        measured = np.mean([m.query(s, r)[0] == o for s, r, o in facts])
        assert abs(m.expected_accuracy() - measured) < 0.10

    def test_membership_oracle(self):
        m = Memory(2048)
        for i in range(20):
            m.store(f"s{i}", "is", f"o{i}")
        assert m.member("s3", "is", "o3")
        assert not m.member("s3", "is", "o7")

    def test_compiled_beats_interpreted_at_high_load(self):
        m = Memory(2048)
        chains = []
        for i in range(120):
            m.store(f"x{i}", "r1", f"y{i}")
            m.store(f"y{i}", "r2", f"z{i}")
            chains.append((f"x{i}", f"y{i}", f"z{i}"))
        compiled = m.compile_pairs([((x, "r1", y), (y, "r2", z))
                                    for x, y, z in chains])
        naive = np.mean([m.chain(x, ["r1", "r2"])[0] == z
                         for x, _, z in chains])
        comp = np.mean([compiled.query_compiled(x, "r1", "r2")[0] == z
                        for x, _, z in chains])
        assert comp > naive + 0.2


class TestCalibration:
    def test_chance_is_half(self):
        assert abs(confidence(D / 2, D) - 0.5) < 1e-9

    def test_capacity_monotone(self):
        assert capacity(4096, 500) > capacity(2048, 500) > capacity(1024, 500)
        assert capacity(2048, 100) > capacity(2048, 10000)
