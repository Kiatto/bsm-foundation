"""
memory_engine.py — BSM Foundation Core Memory Engine

A geometric content-addressable memory with full lifecycle:
  observe → encode → remember → sleep → dream → plan → predict → reflect

Usage:
    mem = MemoryEngine(state_dim=128)
    mem.observe(context_bits, next_token)   # store
    token = mem.predict(context_bits)        # retrieve & vote
    mem.sleep()                              # consolidate + forget
    mem.dream(n_steps=100)                   # reorganize via simulation
    
Part of the BSM Foundation (v1.0).
"""

import torch
import numpy as np
import json, time, os, math
from collections import Counter, defaultdict
from typing import Optional, List, Tuple, Dict, Any, Union


# ============================================================
# Core Memory Engine
# ============================================================

class MemoryEngine:
    """
    Content-addressable geometric memory.
    
    Stores (state, experience) pairs and retrieves via Hamming distance
    with LSH acceleration.
    """
    
    def __init__(self, 
                 state_dim: int = 128,
                 capacity: int = 100000,
                 n_bucket_bits: int = 12,
                 n_candidates: int = 200,
                 n_neighbors: int = 4,
                 min_confidence: float = 0.3,
                 simhash_seed: int = 42):
        
        self.state_dim = state_dim
        self.capacity = capacity
        self.n_candidates = n_candidates
        self.n_neighbors = n_neighbors
        self.min_confidence = min_confidence
        self.n_buckets = 1 << n_bucket_bits
        self.n_bucket_bits = n_bucket_bits
        
        rng = torch.Generator().manual_seed(simhash_seed)
        self.simhash = torch.randn(n_bucket_bits, state_dim, generator=rng)
        
        # Storage
        self.size = 0
        self._states = torch.zeros(capacity, state_dim, dtype=torch.int8)
        self._experiences = [None] * capacity
        self._values = np.zeros(capacity, dtype=np.float32)  # confidence/utility
        self._timestamps = np.zeros(capacity, dtype=np.int64)
        self._access_counts = np.zeros(capacity, dtype=np.int32)
        
        # LSH buckets
        self._buckets = [[] for _ in range(self.n_buckets)]
        
        # Metrics
        self.n_retrievals = 0
        self.n_hits = 0
        self.total_latency_us = 0
        self.n_observations = 0
        self.n_predictions = 0
        self.n_correct = 0
        self.creation_time = time.time()
    
    # ========================
    # Lifecycle API
    # ========================
    
    def observe(self, state: torch.Tensor, experience: Any, value: float = 1.0):
        """
        1. Observe: perceive a (state, experience) pair.
        Stores with initial confidence value.
        """
        idx = self.size if self.size < self.capacity else self._evict_one()
        
        self._states[idx] = state.to(torch.int8)
        self._experiences[idx] = experience
        self._values[idx] = value
        self._timestamps[idx] = int(time.time() * 1e6)
        self._access_counts[idx] = 0
        
        b = self._hash(state)
        self._buckets[b].append(idx)
        
        if self.size < self.capacity:
            self.size += 1
        self.n_observations += 1
    
    def remember(self, state: torch.Tensor, experience: Any, value: float = 1.0):
        """Alias for observe (backward compatibility)."""
        self.observe(state, experience, value)
    
    def recall(self, 
               state: torch.Tensor, 
               top_k: int = None,
               min_value: float = None,
               with_details: bool = False) -> List[Any]:
        """
        2. Recall: retrieve similar experiences from state.
        
        Args:
            state: query state [D]
            top_k: number of nearest neighbors
            min_value: minimum value threshold
            with_details: return (exp, distance, value) tuples
            
        Returns:
            List of experiences (or tuples with details)
        """
        top_k = top_k or self.n_neighbors
        min_value = min_value if min_value is not None else -np.inf
        
        self.n_retrievals += 1
        t0 = time.perf_counter()
        
        candidates = self._find_candidates(state)
        
        if not candidates:
            return []
        
        cand_states = self._states[candidates].float()
        dists = (state.unsqueeze(0).float() != cand_states).sum(dim=1).cpu()
        
        k = min(top_k, len(candidates))
        top_dists, top_idx = torch.topk(dists, k=k, largest=False)
        
        results = []
        for i in range(k):
            ci = candidates[top_idx[i].item()]
            d = top_dists[i].item()
            v = self._values[ci]
            
            if v < min_value:
                continue
            
            w = 1.0 / (1.0 + d / (self.state_dim * 2))
            
            if with_details:
                results.append((self._experiences[ci], d, w, v))
            else:
                results.append(self._experiences[ci])
            
            self._access_counts[ci] += 1
            self._timestamps[ci] = int(time.time() * 1e6)
        
        self.total_latency_us += (time.perf_counter() - t0) * 1e6
        
        if results:
            self.n_hits += 1
        
        return results
    
    def predict(self, state: torch.Tensor, decoder_fn=None) -> Any:
        """
        3. Predict: recall nearest experiences and vote.
        Falls back to decoder_fn if memory is insufficient.
        """
        self.n_predictions += 1
        experiences = self.recall(state, top_k=self.n_neighbors,
                                  min_value=self.min_confidence)
        
        if not experiences and decoder_fn:
            return decoder_fn(state)
        
        if not experiences:
            return None
        
        # Weighted vote by value × recency
        votes = Counter()
        for exp in experiences:
            votes[exp] += 1
        
        return votes.most_common(1)[0][0]
    
    def plan(self, state: torch.Tensor, n_steps: int = 5) -> List[Any]:
        """
        4. Plan: simulate a trajectory by chaining recall → next state.
        Uses the stored (state, next_state) pairs to walk forward.
        """
        trajectory = []
        current = state.clone()
        
        for _ in range(n_steps):
            # Recall similar experiences
            experiences = self.recall(current, top_k=1)
            if not experiences:
                break
            
            # The experience is (predicted_token, next_state)
            exp = experiences[0]
            if not isinstance(exp, tuple) or len(exp) != 2:
                trajectory.append(exp)
                break
            
            token, next_state = exp
            trajectory.append((token, next_state))
            current = next_state
        
        return trajectory
    
    def sleep(self, 
              forget_threshold: float = 0.3,
              consolidate: bool = True):
        """
        5. Sleep: maintenance cycle.
        - Forget low-value entries
        - Consolidate LSH buckets
        - Normalize values
        """
        n_before = self.size
        
        # Forget: prune low-value, low-access entries
        to_keep = []
        for i in range(self.size):
            age = (time.time() * 1e6 - self._timestamps[i]) / 1e6  # seconds
            access_rate = self._access_counts[i] / max(age, 1)
            normalized_value = self._values[i] / max(self._values[:self.size].max(), 1)
            
            # Keep if: recent, frequently accessed, or high value
            if (age < 3600 or                     # recent (< 1 hour)
                access_rate > 0.001 or            # accessed recently
                normalized_value > forget_threshold):  # high value
                to_keep.append(i)
        
        n_forgotten = len(to_keep)
        if len(to_keep) < self.size:
            self._rebuild(to_keep)
        
        # Consolidate: rebuild LSH
        if consolidate:
            self.consolidate()
        
        return n_before - n_forgotten
    
    def dream(self, n_steps: int = 100, noise_scale: float = 0.1):
        """
        6. Dream: explore state space via random walks.
        Generates synthetic experiences to fill memory gaps.
        Used for consolidation and exploration.
        """
        if self.size == 0:
            return 0
        
        n_generated = 0
        current = self._states[np.random.randint(self.size)].float()
        
        for _ in range(n_steps):
            # Add noise to current state
            noise = torch.randn(self.state_dim) * noise_scale
            perturbed = current + noise
            # Re-binarize
            dream_state = torch.sign(perturbed)
            
            # Find nearest real experience
            results = self.recall(dream_state, top_k=1)
            if results:
                # Create interpolated experience
                self.observe(dream_state, results[0], value=0.3)
                n_generated += 1
            
            current = dream_state
        
        return n_generated
    
    def reflect(self) -> Dict[str, Any]:
        """
        7. Reflect: return metrics about memory health and performance.
        """
        if self.size == 0:
            return {"entries": 0, "status": "empty"}
        
        values = self._values[:self.size]
        access = self._access_counts[:self.size]
        ages = (time.time() * 1e6 - self._timestamps[:self.size]) / 1e6
        
        return {
            "entries": self.size,
            "capacity": self.capacity,
            "usage_pct": 100.0 * self.size / self.capacity,
            "mean_value": float(values.mean()),
            "max_value": float(values.max()),
            "min_value": float(values.min()),
            "mean_access": int(access.mean()),
            "max_access": int(access.max()),
            "zero_access_pct": 100.0 * (access == 0).sum() / len(access),
            "mean_age_s": float(ages.mean()),
            "buckets_used": int(sum(1 for b in self._buckets if b)),
            "bucket_load": float(np.mean([len(b) for b in self._buckets if b]) if any(self._buckets) else 0),
            "retrievals": int(self.n_retrievals),
            "hit_rate": float(self.n_hits / max(self.n_retrievals, 1)),
            "avg_latency_us": float(self.total_latency_us / max(self.n_retrievals, 1)),
            "observations": int(self.n_observations),
            "predictions": int(self.n_predictions),
            "uptime_s": float(time.time() - self.creation_time),
            "memory_bytes": int(self._memory_usage()),
        }
    
    # ========================
    # Persistence
    # ========================
    
    def save(self, path: str):
        torch.save({
            "config": {
                "state_dim": self.state_dim,
                "capacity": self.capacity,
                "n_bucket_bits": self.n_bucket_bits,
                "n_candidates": self.n_candidates,
                "n_neighbors": self.n_neighbors,
                "min_confidence": self.min_confidence,
            },
            "size": self.size,
            "states": self._states[:self.size],
            "experiences": self._experiences[:self.size],
            "values": self._values[:self.size],
            "timestamps": self._timestamps[:self.size],
            "access_counts": self._access_counts[:self.size],
            "metrics": {
                "n_retrievals": self.n_retrievals,
                "n_hits": self.n_hits,
                "total_latency_us": self.total_latency_us,
                "n_observations": self.n_observations,
                "n_predictions": self.n_predictions,
                "n_correct": self.n_correct,
            }
        }, path, protocol=torch.serialization.DEFAULT_PROTOCOL)
    
    def load(self, path: str):
        data = torch.load(path, weights_only=False)
        cfg = data["config"]
        self.state_dim = cfg["state_dim"]
        self.capacity = cfg["capacity"]
        self.n_bucket_bits = cfg["n_bucket_bits"]
        self.n_buckets = 1 << self.n_bucket_bits
        self.n_candidates = cfg["n_candidates"]
        self.n_neighbors = cfg["n_neighbors"]
        self.min_confidence = cfg["min_confidence"]
        
        self.size = data["size"]
        self._states[:self.size] = data["states"]
        self._experiences[:self.size] = data["experiences"]
        self._values[:self.size] = data["values"]
        self._timestamps[:self.size] = data["timestamps"]
        self._access_counts[:self.size] = data["access_counts"]
        
        for k, v in data["metrics"].items():
            setattr(self, k, v)
        
        # Rebuild LSH
        self._buckets = [[] for _ in range(self.n_buckets)]
        rng = torch.Generator().manual_seed(42)
        self.simhash = torch.randn(self.n_bucket_bits, self.state_dim, generator=rng)
        for i in range(self.size):
            b = self._hash(self._states[i].float())
            self._buckets[b].append(i)
        
        return self
    
    # ========================
    # Internal
    # ========================
    
    def _hash(self, state: torch.Tensor) -> int:
        proj = (state.float() @ self.simhash.T).flatten()
        b = 0
        for i in range(self.n_bucket_bits):
            if proj[i] > 0:
                b |= 1 << i
        return b
    
    def _find_candidates(self, state: torch.Tensor) -> List[int]:
        b = self._hash(state)
        cs = set()
        for idx in self._buckets[b]:
            cs.add(idx)
        if len(cs) < self.n_candidates:
            for bb in range(self.n_bucket_bits):
                nb = b ^ (1 << bb)
                for idx in self._buckets[nb]:
                    cs.add(idx)
                if len(cs) >= self.n_candidates:
                    break
        return list(cs)[:self.n_candidates]
    
    def _evict_one(self) -> int:
        """Evict lowest-value entry."""
        values = self._values[:self.size]
        return int(np.argmin(values))
    
    def _rebuild(self, keep_indices: List[int]):
        n = len(keep_indices)
        new_states = [self._states[i].clone() for i in keep_indices]
        new_exps = [self._experiences[i] for i in keep_indices]
        new_vals = self._values[keep_indices].copy()
        new_ts = self._timestamps[keep_indices].copy()
        new_ac = self._access_counts[keep_indices].copy()
        
        self._buckets = [[] for _ in range(self.n_buckets)]
        self.size = n
        
        for i in range(n):
            self._states[i] = new_states[i]
            self._experiences[i] = new_exps[i]
            self._values[i] = new_vals[i]
            self._timestamps[i] = new_ts[i]
            self._access_counts[i] = new_ac[i]
            b = self._hash(new_states[i].float())
            self._buckets[b].append(i)
    
    def consolidate(self):
        """Rebuild LSH from scratch (after value drift)."""
        self._buckets = [[] for _ in range(self.n_buckets)]
        for i in range(self.size):
            b = self._hash(self._states[i].float())
            self._buckets[b].append(i)
    
    def _memory_usage(self) -> int:
        """Estimated memory usage in bytes."""
        state_bytes = self.size * self.state_dim  # int8
        exp_bytes = sum(len(str(e)) if e else 0 for e in self._experiences[:self.size])
        overhead = (self.size * 8 * 3)  # values, timestamps, access_counts
        lsh_size = sum(len(b) * 4 for b in self._buckets)
        return int(state_bytes + exp_bytes + overhead + lsh_size + 100000)
