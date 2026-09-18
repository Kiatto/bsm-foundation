"""
test_bitpack.py — Property and equivalence tests for Bitpacked SIMD Memory.
"""

import hashlib
import threading

import pytest
import numpy as np

from reference.abm import Memory as StandardMemory
from bsm.memory.bitpack import (
    BitpackedMemory,
    pack_vector,
    unpack_vector,
    bind_packed,
    hamming_packed,
    bundle_packed,
)


def test_pack_unpack_roundtrip():
    rng = np.random.RandomState(42)
    dense = np.where(rng.rand(2048) > 0.5, 1, -1).astype(np.int8)
    packed = pack_vector(dense)
    unpacked = unpack_vector(packed, 2048)
    np.testing.assert_array_equal(dense, unpacked)


def test_bind_packed_equivalence():
    rng = np.random.RandomState(42)
    a_dense = np.where(rng.rand(2048) > 0.5, 1, -1).astype(np.int8)
    b_dense = np.where(rng.rand(2048) > 0.5, 1, -1).astype(np.int8)

    a_packed = pack_vector(a_dense)
    b_packed = pack_vector(b_dense)

    bind_dense = (a_dense.astype(np.int16) * b_dense.astype(np.int16)).astype(np.int8)
    bind_packed_res = bind_packed(a_packed, b_packed)

    np.testing.assert_array_equal(bind_dense, unpack_vector(bind_packed_res, 2048))


def test_hamming_packed_equivalence():
    rng = np.random.RandomState(42)
    a_dense = np.where(rng.rand(2048) > 0.5, 1, -1).astype(np.int8)
    b_dense = np.where(rng.rand(2048) > 0.5, 1, -1).astype(np.int8)

    h_dense = int(np.count_nonzero(a_dense != b_dense))
    h_packed = hamming_packed(pack_vector(a_dense), pack_vector(b_dense))

    assert h_dense == h_packed


def test_bitpacked_memory_store_query_chain_member():
    mem = BitpackedMemory(dim=2048)
    mem.store("user_101", "has_role", "administrator")
    mem.store("administrator", "can_access", "database_server")
    mem.store("database_server", "located_in", "us_east_region")

    # Single Query
    target, conf = mem.query("user_101", "has_role")
    assert target == "administrator"
    assert conf > 0.90

    # Multi-hop Reasoning
    dest, chain_conf = mem.chain("user_101", ["has_role", "can_access", "located_in"])
    assert dest == "us_east_region"
    assert chain_conf > 0.80

    # Member Truth Oracle
    assert mem.member("user_101", "has_role", "administrator") is True
    assert mem.member("user_101", "has_role", "guest") is False


def test_bitpacked_vs_standard_equivalence():
    std_mem = StandardMemory(dim=2048)
    bp_mem = BitpackedMemory(dim=2048)

    facts = [
        ("service_a", "depends_on", "service_b"),
        ("service_b", "reads_from", "cache_redis"),
        ("service_b", "writes_to", "postgres_db"),
    ]

    for s, r, o in facts:
        std_mem.store(s, r, o)
        bp_mem.store(s, r, o)

    # Both must answer identically
    ans_std, conf_std = std_mem.query("service_a", "depends_on")
    ans_bp, conf_bp = bp_mem.query("service_a", "depends_on")

    assert ans_std == ans_bp
    assert abs(conf_std - conf_bp) < 1e-4

    assert std_mem.member("service_b", "reads_from", "cache_redis") == bp_mem.member("service_b", "reads_from", "cache_redis")


def test_permute_packed_equals_dense_roll():
    """Rho on the packed bitstring must match the dense np.roll for every shift."""
    from bsm.memory.bitpack import permute_packed

    rng = np.random.RandomState(11)
    for dim in (64, 256, 2048):
        packed = pack_vector(rng.choice([-1, 1], dim).astype(np.int8))
        for k in [0, 1, 5, 7, 8, 9, 63, 64, 65, dim - 1, dim, dim + 3, -1, -9]:
            expected = pack_vector(np.roll(unpack_vector(packed, dim), k))
            np.testing.assert_array_equal(expected, permute_packed(packed, k, dim))


def test_incremental_trace_matches_full_bundle():
    """The running vote counter must reproduce bundle_packed bit for bit.

    Covers the parity tie-break, which differs between even and odd fact counts,
    and weighted stores, which push several copies of one fact into the bundle.
    """
    for n_facts in range(1, 25):
        mem = BitpackedMemory(dim=256)
        for i in range(n_facts):
            mem.store(f"s{i}", "rel", f"o{i}", weight=1 + (i % 3))
        np.testing.assert_array_equal(bundle_packed(mem._facts, 256), mem.trace)


def test_calibrated_acceptance_keeps_true_hits_and_bounds_false_ones():
    """Acceptance must scale with the codebook, not with a fixed confidence cutoff.

    A fixed 0.75 confidence floor implies z >= 8.79 at temperature 8 and rejects
    every true hit once the trace holds a few dozen facts, even while predicted
    accuracy is 1.00. The calibrated test keeps them and holds false accepts near
    alpha.
    """
    from bsm.memory.bitpack import confidence_packed, hamming_z

    dim, n_facts, alpha = 2048, 30, 0.01
    mem = BitpackedMemory(dim=dim)
    for i in range(n_facts):
        mem.store(f"s{i}", "knows", f"o{i}")
    assert mem.expected_accuracy() > 0.99

    accepted = sum(1 for i in range(n_facts)
                   if mem.query_calibrated(f"s{i}", "knows", alpha=alpha)[0] == f"o{i}")
    assert accepted == n_facts

    fixed_floor = sum(1 for i in range(n_facts) if mem.query(f"s{i}", "knows")[1] >= 0.75)
    assert fixed_floor == 0  # the regression the calibrated test removes

    probes = 300
    false_accepts = sum(1 for i in range(probes)
                        if mem.query_calibrated(f"s{i % n_facts}", f"absent_{i}",
                                                alpha=alpha)[0] is not None)
    assert false_accepts <= 0.05 * probes


def test_acceptance_threshold_grows_with_codebook():
    """More candidates must demand a larger margin for the same false-accept rate."""
    from bsm.memory.bitpack import acceptance_z

    thresholds = [acceptance_z(m, alpha=0.01) for m in (10, 1_000, 100_000, 10_000_000)]
    assert thresholds == sorted(thresholds)
    assert all(b > a for a, b in zip(thresholds, thresholds[1:]))


def test_codeword_bits_are_stable_and_thread_safe():
    """Reusing one reseeded RandomState must not change any codeword.

    Constructing np.random.RandomState(seed) per call cost ~193 us and dominated
    every operation involving a new entity. Reseeding a shared generator is 1.65 us
    and provably yields the same stream — but the generator is now shared mutable
    state, so determinism has to hold under concurrent use too.
    """
    from bsm.memory.bitpack import random_packed_hv

    dim = 512
    expected = {name: random_packed_hv(name, dim) for name in ("alpha", "beta", "gamma")}
    # An independent generator instance must produce the identical bit pattern.
    for name, packed in expected.items():
        seed = int.from_bytes(hashlib.md5(name.lower().encode()).digest()[:4], "little")
        raw = np.random.RandomState(seed).rand(dim) > 0.5
        np.testing.assert_array_equal(np.packbits(raw, bitorder="big").view(np.uint64), packed)

    results = {}
    def worker(idx):
        results[idx] = [random_packed_hv(n, dim) for n in expected]
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for produced in results.values():
        for packed, name in zip(produced, expected):
            np.testing.assert_array_equal(expected[name], packed)


def test_store_many_matches_store():
    """Bulk loading must be indistinguishable from storing one fact at a time."""
    triples = [(f"s{i}", "rel", f"o{i}") for i in range(9)]
    for weight in (1, 3):
        one_by_one = BitpackedMemory(dim=256)
        bulk = BitpackedMemory(dim=256)
        for s, r, o in triples:
            one_by_one.store(s, r, o, weight=weight)
        bulk.store_many(triples, weight=weight)

        np.testing.assert_array_equal(one_by_one.trace, bulk.trace)
        assert one_by_one._weight == bulk._weight
        for s, r, o in triples:
            assert one_by_one.query_raw(s, r) == bulk.query_raw(s, r)
            assert one_by_one.member(s, r, o) == bulk.member(s, r, o)


def test_answers_only_restricts_to_object_role_and_is_consistent():
    """Restricting to object-role codewords must not change correct answers.

    Subjects and relations can never be the answer to a (subject, relation) query,
    so excluding them removes null competitors: it cuts latency and lowers the bar
    a true hit has to clear. Where the unrestricted cleanup is right, the restricted
    one must agree.
    """
    mem = BitpackedMemory(dim=2048)
    triples = [(f"s{i}", "rel", f"o{i}") for i in range(20)]
    mem.store_many(triples)

    assert len(mem._answers) == 20
    assert len(mem.items) > len(mem._answers)  # subjects and the relation are excluded

    for s, r, o in triples:
        full_name, full_dist = mem.query_raw(s, r)
        restricted_name, restricted_dist = mem.query_raw(s, r, answers_only=True)
        if full_name == o:
            assert restricted_name == o
            assert restricted_dist == full_dist
        assert restricted_name in {name for _s, _r, name in triples}


def test_load_without_answer_names_falls_back_to_full_codebook():
    """A pre-existing artifact must not start answering nothing."""
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        dir_path = Path(tmp_dir) / "legacy"
        mem = BitpackedMemory(dim=2048)
        mem.store("user", "lives_in", "verona")
        mem.save(dir_path)

        meta_path = dir_path / "meta.json"
        meta = json.loads(meta_path.read_text())
        del meta["answer_names"]  # simulate an artifact written before object roles
        meta_path.write_text(json.dumps(meta))

        legacy = BitpackedMemory.load(dir_path)

    assert legacy._answers is None
    assert legacy.query_raw("user", "lives_in", answers_only=True)[0] == "verona"
    assert legacy.query_calibrated("user", "lives_in")[0] == "verona"
    legacy.store("verona", "country", "italy")  # must not blow up on a None view
    assert legacy.query_calibrated("verona", "country")[0] == "italy"
