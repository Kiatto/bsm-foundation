"""
sharded.py — Multi-Trace Holographic Sharding.

Distributes relational facts across multiple holographic memory shards so that no
single trace exceeds its noise floor (Law IV), keeping each shard inside its
Memory Contract as the total fact count grows.

Scope, measured: sharding buys *accuracy*, not query cost. A query carries no key
to route on, so it scans every shard: cost is O(shards), i.e. proportional to
total_facts / max_facts_per_shard. Prefer query_calibrated() over query() when a
wrong answer is worse than no answer — the shard count enters the threshold.
"""

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union, Dict, Any

from bsm.memory.bitpack import (BitpackedMemory, BitpackedItemMemory, acceptance_z,
                                confidence_packed, hamming_z)


class MultiTraceMemory:
    """Distributed multi-shard holographic memory."""

    def __init__(self, dim: int = 2048, max_facts_per_shard: int = 400):
        self.dim = dim
        self.max_facts_per_shard = max_facts_per_shard
        self.shared_items = BitpackedItemMemory(dim=dim)
        self.shards: List[BitpackedMemory] = []
        self._add_shard()

    def _add_shard(self) -> BitpackedMemory:
        shard = BitpackedMemory(dim=self.dim, items=self.shared_items)
        self.shards.append(shard)
        return shard

    def store(self, subject: str, relation: str, obj: str, weight: int = 1):
        """Store a fact into an available shard that has capacity."""
        for shard in self.shards:
            if len(shard._facts) < self.max_facts_per_shard:
                shard.store(subject, relation, obj, weight=weight)
                return
        
        # All shards full -> spawn a new shard
        new_shard = self._add_shard()
        new_shard.store(subject, relation, obj, weight=weight)

    def query(self, subject: str, relation: str, subset: Optional[Sequence[str]] = None) -> Tuple[str, float]:
        """Query every shard and return the globally nearest codeword.

        Selection is by minimum Hamming distance, which is the cleanup over the
        union of shards. Note this is O(shards) per query: sharding buys accuracy
        (each trace stays below its noise floor), not query cost.
        """
        best_answer = ""
        best_dist = self.dim + 1

        for shard in self.shards:
            if not shard._facts:
                continue
            ans, dist = shard.query_raw(subject, relation, subset=subset)
            if ans and dist < best_dist:
                best_answer = ans
                best_dist = dist

        if not best_answer:
            return "", 0.5
        return best_answer, confidence_packed(best_dist, self.dim)

    def query_calibrated(self, subject: str, relation: str,
                         subset: Optional[Sequence[str]] = None,
                         alpha: float = 0.01) -> Tuple[Optional[str], float, float]:
        """Query with the false-accept rate capped at `alpha` across all shards.

        Each shard contributes its own minimum over the codebook, so the reported
        hit is the best of (shards x candidates) draws under the null. The
        acceptance threshold is computed on that product; using a per-shard
        threshold — or a max of per-shard confidences — inflates false accepts
        linearly in the shard count.
        """
        best_answer = ""
        best_dist = self.dim + 1
        candidates = 0
        for shard in self.shards:
            if not shard._facts:
                continue
            ans, dist = shard.query_raw(subject, relation, subset=subset, answers_only=True)
            candidates += shard._candidate_count(subset, True)
            if ans and dist < best_dist:
                best_answer, best_dist = ans, dist

        if not best_answer:
            return None, 0.5, 0.0

        z = hamming_z(best_dist, self.dim)
        conf = confidence_packed(best_dist, self.dim)
        if z < acceptance_z(candidates, alpha):
            return None, conf, z
        return best_answer, conf, z

    def chain(self, start: str, relations: Sequence[str]) -> Tuple[str, float]:
        """Multi-hop reasoning across multi-shard memory."""
        node, conf = start, 1.0
        for r in relations:
            node, c = self.query(node, r)
            conf *= c
        return node, conf

    def member(self, subject: str, relation: str, obj: str) -> bool:
        """Check if triple exists in any shard using algebraic truth oracle."""
        return any(shard.member(subject, relation, obj) for shard in self.shards if shard._facts)

    @property
    def total_facts(self) -> int:
        return sum(len(s._facts) for s in self.shards)

    def stats(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "shards_count": len(self.shards),
            "total_facts": self.total_facts,
            "max_facts_per_shard": self.max_facts_per_shard,
            "items_count": len(self.shared_items),
        }

    def save(self, dir_path: Union[str, Path]):
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        self.shared_items.save(path / "shared_items")

        meta = {
            "dim": self.dim,
            "max_facts_per_shard": self.max_facts_per_shard,
            "shards_count": len(self.shards)
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f)

        for i, shard in enumerate(self.shards):
            shard.save(path / f"shard_{i}")

    @classmethod
    def load(cls, dir_path: Union[str, Path], mmap_mode: Optional[str] = None) -> "MultiTraceMemory":
        path = Path(dir_path)
        with open(path / "meta.json", "r") as f:
            meta = json.load(f)

        shared_items = BitpackedItemMemory.load(path / "shared_items", mmap_mode=mmap_mode)
        instance = cls.__new__(cls)
        instance.dim = meta["dim"]
        instance.max_facts_per_shard = meta["max_facts_per_shard"]
        instance.shared_items = shared_items
        instance.shards = []

        for i in range(meta["shards_count"]):
            shard_path = path / f"shard_{i}"
            if shard_path.exists():
                shard = BitpackedMemory.load(shard_path, mmap_mode=mmap_mode)
                shard.items = shared_items
                instance.shards.append(shard)

        return instance
