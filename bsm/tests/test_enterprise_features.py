"""
test_enterprise_features.py — Tests for MMap persistence, cleanup exactness,
Multi-Trace Sharding, Text Extractor, and REST API.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from bsm.memory.bitpack import (BitpackedMemory, BitpackedItemMemory, batch_hamming_packed,
                                pack_vector, unpack_vector)
from bsm.memory.sharded import MultiTraceMemory
from bsm.integrations.agent_memory import AgentMemory


def test_mmap_and_save_load_persistence():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dir_path = Path(tmp_dir) / "test_mem"

        # 1. Create and populate memory
        mem = BitpackedMemory(dim=2048)
        mem.store("user_alpha", "lives_in", "verona")
        mem.store("verona", "country", "italy")
        mem.save(dir_path)

        # 2. Load with zero-copy mmap
        loaded_mem = BitpackedMemory.load(dir_path, mmap_mode="r")

        # 3. Verify query and member check on loaded instance
        ans, conf = loaded_mem.query("user_alpha", "lives_in")
        assert ans == "verona"
        assert conf > 0.85
        assert loaded_mem.member("user_alpha", "lives_in", "verona") is True


def test_cleanup_is_exact_under_noise():
    """Cleanup must return the true argmin, not merely an exact-match lookup.

    Replaces the old VP-Tree test: metric pruning was measured to visit ~90% of a
    5,000-item codebook (cleanup queries sit at 0.45-0.48*D from their own codeword,
    non-targets at 0.50*D), so the index was removed in favour of the exact scan.
    """
    dim, m = 512, 300
    item_mem = BitpackedItemMemory(dim=dim)
    names = [f"item_{i}" for i in range(m)]
    for n in names:
        item_mem.add(n)

    rng = np.random.RandomState(7)
    for i in range(0, m, 20):
        dense = unpack_vector(item_mem.get(names[i]), dim).copy()
        dense[rng.choice(dim, dim // 8, replace=False)] *= -1
        noisy = pack_vector(dense)

        got_name, got_dist = item_mem.cleanup(noisy)
        brute = batch_hamming_packed(np.vstack(item_mem._states), noisy)
        assert got_dist == int(brute.min())
        assert got_name == names[int(np.argmin(brute))]


def test_cleanup_sees_items_added_after_first_query():
    """A cached codebook matrix must be invalidated by add()."""
    item_mem = BitpackedItemMemory(dim=256)
    for n in ("a", "b", "c"):
        item_mem.add(n)
    item_mem.cleanup(item_mem.get("a"))  # populates the matrix cache

    item_mem.add("zeta")
    assert item_mem.cleanup(item_mem.get("zeta")) == ("zeta", 0)


def test_cleanup_subset_ignores_unknown_names():
    """Names absent from the codebook must not shift the returned label."""
    item_mem = BitpackedItemMemory(dim=256)
    for n in ("a", "b", "c"):
        item_mem.add(n)

    name, dist = item_mem.cleanup(item_mem.get("c"), subset=["ghost", "a", "c"])
    assert (name, dist) == ("c", 0)


def test_multi_trace_sharded_memory():
    # Set small shard capacity to trigger sharding
    multi_mem = MultiTraceMemory(dim=2048, max_facts_per_shard=3)

    facts = [
        ("entity_1", "rel_a", "target_1"),
        ("entity_2", "rel_a", "target_2"),
        ("entity_3", "rel_a", "target_3"),
        ("entity_4", "rel_a", "target_4"),  # Triggers 2nd shard
        ("entity_5", "rel_a", "target_5"),
    ]

    for s, r, o in facts:
        multi_mem.store(s, r, o)

    assert len(multi_mem.shards) == 2
    assert multi_mem.total_facts == 5

    # Query across shards
    ans4, conf4 = multi_mem.query("entity_4", "rel_a")
    assert ans4 == "target_4"
    assert conf4 > 0.85


def test_natural_text_extractor():
    agent_mem = AgentMemory(dim=2048)
    res = agent_mem.remember_text("Marco works at Google. Google located in California.")

    assert len(res) >= 2
    ans1, conf1 = agent_mem.recall("marco", "works_at")
    assert ans1 == "google"

    ans2, conf2 = agent_mem.recall("google", "located_in")
    assert ans2 == "california"


def test_fastapi_rest_server():
    try:
        import sys
        sys.modules.pop('bsm.server', None)
        from fastapi.testclient import TestClient
        import bsm.server
        app = bsm.server.app
        if app is None:
            pytest.skip("FastAPI app is None.")
    except (ImportError, AttributeError):
        pytest.skip("FastAPI / TestClient not installed.")

    client = TestClient(app)



    # 1. Store via REST
    resp = client.post("/remember", json={"subject": "api_user", "relation": "role", "object": "admin"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    # 2. Recall via REST
    resp_recall = client.post("/recall", json={"subject": "api_user", "relation": "role"})
    assert resp_recall.status_code == 200
    assert resp_recall.json()["answer"] == "admin"

    # 3. Semantic Route via REST
    resp_route = client.post("/semantic_route", json={
        "prompt": "What role does api_user have?",
        "entity": "api_user",
        "relation": "role"
    })
    assert resp_route.status_code == 200
    assert resp_route.json()["handled"] is True
    assert resp_route.json()["answer"] == "admin"

    # 4. Stats Endpoint
    resp_stats = client.get("/stats")
    assert resp_stats.status_code == 200
    assert resp_stats.json()["dim"] == 2048


def test_sharded_query_selects_global_nearest_not_best_confidence():
    """Selection across shards must be the argmin over the union of shards.

    Picking the shard with the highest confidence is the same comparison only when
    confidence is monotone in distance; asserting the argmin directly pins the
    property that matters and keeps the reported confidence tied to it.
    """
    multi_mem = MultiTraceMemory(dim=2048, max_facts_per_shard=4)
    facts = [(f"entity_{i}", "rel_a", f"target_{i}") for i in range(12)]
    for s, r, o in facts:
        multi_mem.store(s, r, o)
    assert len(multi_mem.shards) == 3

    for s, r, o in facts:
        name, dist = min((shard.query_raw(s, r) for shard in multi_mem.shards
                          if shard._facts), key=lambda pair: pair[1])
        assert multi_mem.query(s, r)[0] == name


def test_sharded_acceptance_accounts_for_shard_count():
    """False accepts must not grow with the shard count.

    Every shard contributes its own minimum over the codebook, so a hit is the best
    of (shards x candidates) null draws; a per-shard threshold would let the
    false-accept rate scale linearly in the number of shards.
    """
    multi_mem = MultiTraceMemory(dim=2048, max_facts_per_shard=4)
    for i in range(40):
        multi_mem.store(f"entity_{i}", "rel_a", f"target_{i}")
    assert len(multi_mem.shards) == 10

    stored = sum(1 for i in range(40)
                 if multi_mem.query_calibrated(f"entity_{i}", "rel_a")[0] == f"target_{i}")
    assert stored == 40

    probes = 200
    false_accepts = sum(1 for i in range(probes)
                        if multi_mem.query_calibrated(f"entity_{i % 40}", f"absent_{i}")[0] is not None)
    assert false_accepts <= 0.05 * probes


def test_shards_actually_share_one_codebook():
    """Shards must use the codebook they were given, not a private copy.

    BitpackedItemMemory defines __len__, so an empty shared codebook is falsy: a
    `items or BitpackedItemMemory(dim)` default silently gave every shard its own
    codebook, left stats()["items_count"] at 0, broke any candidate-count
    calibration, and made load() (which reassigns the shared codebook) behave
    differently from the in-memory object it restored.
    """
    multi_mem = MultiTraceMemory(dim=2048, max_facts_per_shard=2)
    for i in range(6):
        multi_mem.store(f"entity_{i}", "rel_a", f"target_{i}")

    assert len(multi_mem.shards) == 3
    assert all(shard.items is multi_mem.shared_items for shard in multi_mem.shards)
    assert len(multi_mem.shared_items) > 0
    assert multi_mem.stats()["items_count"] == len(multi_mem.shared_items)


def test_sharded_save_load_preserves_answers():
    """Reload must reproduce the same answers, not merely load without error."""
    multi_mem = MultiTraceMemory(dim=2048, max_facts_per_shard=3)
    facts = [(f"entity_{i}", "rel_a", f"target_{i}") for i in range(9)]
    for s, r, o in facts:
        multi_mem.store(s, r, o)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dir_path = Path(tmp_dir) / "multi"
        multi_mem.save(dir_path)
        reloaded = MultiTraceMemory.load(dir_path)

    assert reloaded.total_facts == multi_mem.total_facts
    for s, r, o in facts:
        assert reloaded.query(s, r) == multi_mem.query(s, r)
        assert reloaded.member(s, r, o) == multi_mem.member(s, r, o)
