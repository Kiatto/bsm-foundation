"""
mib.py — Memory Intelligence Benchmark (MIB)

Evaluates memory systems on:
  - Knowledge Density (bits/MB)
  - Retrieval accuracy (hit rate, top-k)
  - Latency (µs/query)
  - Forgetting quality (graceful degradation)
  - Consolidation gain
  - Continual learning (no catastrophic forgetting)
  - Memory efficiency (accuracy/KB)

Usage:
    benchmark = MIB()
    results = benchmark.evaluate(memory, test_states, test_targets)
    benchmark.report(results)
"""

import torch
import numpy as np
import time, math, json
from typing import Callable, Dict, Any, List, Tuple


class MIB:
    """
    Memory Intelligence Benchmark.
    
    Measures 7 dimensions of memory system performance.
    Final score is the harmonic mean of all dimensions.
    """
    
    def __init__(self, 
                 n_train: int = 10000,
                 n_test: int = 1000,
                 state_dim: int = 128,
                 state_generator: str = "structured"):
        
        self.n_train = n_train
        self.n_test = n_test
        self.state_dim = state_dim
        self.state_generator = state_generator
        
        # Generate synthetic data
        rng = torch.Generator().manual_seed(42)
        if state_generator == "structured":
            # Generate states on a low-dimensional manifold
            latent_dim = 30
            latent = torch.randn(n_train + n_test, latent_dim, generator=rng)
            proj = torch.randn(latent_dim, state_dim, generator=rng)
            states = torch.sign(latent @ proj)
            targets = (latent.sum(dim=1) > 0).long()  # binary classification
        else:
            states = torch.sign(torch.randn(n_train + n_test, state_dim, generator=rng))
            targets = torch.randint(0, 10, (n_train + n_test,), generator=rng)
        
        self.train_states = states[:n_train]
        self.train_targets = targets[:n_train]
        self.test_states = states[n_train:n_train + n_test]
        self.test_targets = targets[n_train:n_train + n_test]
    
    def evaluate(self, memory, store_fn=None, recall_fn=None):
        """
        Run full MIB evaluation.
        
        Args:
            memory: MemoryEngine instance (or any object with store/recall)
            store_fn: function(state, exp) → None (optional)
            recall_fn: function(state, top_k) → list (optional)
        
        Returns:
            dict of scores
        """
        results = {}
        
        # Phase 1: Storage
        t0 = time.time()
        for i in range(self.n_train):
            s = self.train_states[i]
            t = self.train_targets[i].item()
            if store_fn:
                store_fn(s, t)
            else:
                memory.observe(s, t, value=1.0)
        store_time = time.time() - t0
        
        results["store_throughput"] = self.n_train / store_time
        
        # Phase 2: KD (Knowledge Density)
        mem_bytes = memory._memory_usage() if hasattr(memory, '_memory_usage') else 1
        # Compute prediction accuracy
        correct_predictions = 0
        total_predictions = 0
        latencies = []
        
        t0 = time.time()
        for i in range(self.n_test):
            s = self.test_states[i]
            t = self.test_targets[i].item()
            
            tq = time.perf_counter()
            if recall_fn:
                exps = recall_fn(s, top_k=4)
            else:
                exps = memory.recall(s, top_k=4)
            latencies.append((time.perf_counter() - tq) * 1e6)
            
            if exps:
                # Vote
                from collections import Counter
                votes = Counter(exps)
                pred = votes.most_common(1)[0][0]
                if pred == t:
                    correct_predictions += 1
                total_predictions += 1
        
        query_time = time.time() - t0
        accuracy = correct_predictions / max(total_predictions, 1)
        
        # Information gain
        n_classes = len(set(self.train_targets.tolist()))
        baseline_uncertainty = math.log2(n_classes)
        pred_uncertainty = math.log2(max(1 / max(accuracy, 0.01), 2))
        bits_gained = baseline_uncertainty - pred_uncertainty
        
        results["accuracy"] = accuracy
        results["bits_gained"] = bits_gained
        results["memory_bytes"] = mem_bytes
        results["knowledge_density"] = bits_gained / (mem_bytes / 1e6)
        results["avg_latency_us"] = np.mean(latencies) if latencies else 0
        results["p50_latency_us"] = float(np.percentile(latencies, 50)) if latencies else 0
        results["p99_latency_us"] = float(np.percentile(latencies, 99)) if latencies else 0
        
        # Phase 3: Continual learning
        # Add new classes sequentially, measure forgetting
        results["forgetting_rate"] = self._measure_forgetting(memory, store_fn, recall_fn)
        
        # Phase 4: Consolidation gain
        results["consolidation_gain"] = self._measure_consolidation(memory, store_fn, recall_fn)
        
        # Phase 5: Efficiency
        results["accuracy_per_kb"] = accuracy / (mem_bytes / 1024) if mem_bytes > 0 else 0
        results["throughput"] = total_predictions / max(query_time, 1e-6)
        
        # Composite score (harmonic mean)
        scores = [
            max(results["knowledge_density"], 0.001),
            max(results["accuracy"], 0.001),
            1 / max(results["avg_latency_us"], 1),
            1 / max(results["p99_latency_us"], 1),
            max(1 - results["forgetting_rate"], 0.001),
            max(results["consolidation_gain"], 0.001),
            max(results["accuracy_per_kb"], 0.001) * 100,
        ]
        
        # Normalize each to [0.001, 1] roughly
        norms = [1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 10.0]  # rough normalization
        normalized = [min(s / n, 1.0) for s, n in zip(scores, norms)]
        
        hmean = len(normalized) / sum(1 / max(n, 0.001) for n in normalized)
        results["mib_score"] = hmean
        
        return results
    
    def _measure_forgetting(self, memory, store_fn, recall_fn) -> float:
        """Measure catastrophic forgetting by adding new patterns."""
        rng = np.random.RandomState(42)
        base_accuracy = 0.0
        
        # Add 5 new batches, measure accuracy after each
        accuracies = []
        for batch in range(5):
            new_states = torch.sign(torch.randn(500, self.state_dim))
            new_targets = torch.randint(10, 20, (500,))
            
            for i in range(500):
                s = new_states[i]
                t = new_targets[i].item()
                if store_fn:
                    store_fn(s, t)
                else:
                    memory.observe(s, t, value=1.0)
            
            # Evaluate on original test set
            correct = 0
            total = 0
            for i in range(min(200, self.n_test)):
                s = self.test_states[i]
                t = self.test_targets[i].item()
                if recall_fn:
                    exps = recall_fn(s, top_k=4)
                else:
                    exps = memory.recall(s, top_k=4)
                if exps:
                    from collections import Counter
                    votes = Counter(exps)
                    pred = votes.most_common(1)[0][0]
                    if pred == t:
                        correct += 1
                    total += 1
            acc = correct / max(total, 1)
            accuracies.append(acc)
        
        if len(accuracies) < 2:
            return 0.0
        
        # Forgetting rate: relative drop from first to last
        return max(0, (accuracies[0] - accuracies[-1]) / max(accuracies[0], 0.01))
    
    def _measure_consolidation(self, memory, store_fn, recall_fn) -> float:
        """Measure gain from consolidation (latency improvement)."""
        # Measure latency before
        latencies_before = []
        for i in range(min(200, self.n_test)):
            s = self.test_states[i]
            tq = time.perf_counter()
            if recall_fn:
                recall_fn(s, top_k=4)
            else:
                memory.recall(s, top_k=4)
            latencies_before.append((time.perf_counter() - tq) * 1e6)
        
        # Consolidate
        if hasattr(memory, 'consolidate'):
            memory.consolidate()
        
        # Measure latency after
        latencies_after = []
        for i in range(min(200, self.n_test)):
            s = self.test_states[i]
            tq = time.perf_counter()
            if recall_fn:
                recall_fn(s, top_k=4)
            else:
                memory.recall(s, top_k=4)
            latencies_after.append((time.perf_counter() - tq) * 1e6)
        
        before = np.mean(latencies_before) if latencies_before else 1
        after = np.mean(latencies_after) if latencies_after else 1
        
        return max(0, (before - after) / max(before, 0.01))
    
    def report(self, results: Dict[str, Any]):
        """Pretty-print benchmark results."""
        print(f"\n{'='*65}")
        print(f"  MEMORY INTELLIGENCE BENCHMARK (MIB)")
        print(f"{'='*65}")
        print(f"\n  Configuration:")
        print(f"    Train set:     {self.n_train} samples")
        print(f"    Test set:      {self.n_test} samples")
        print(f"    State dim:     {self.state_dim}")
        print(f"    Generator:     {self.state_generator}")
        
        print(f"\n  {'Metric':30s} {'Value':>15s}")
        print(f"  {'-'*47}")
        
        def fmt(v):
            if isinstance(v, float):
                if abs(v) < 0.01: return f"{v:.6f}"
                if abs(v) < 1: return f"{v:.4f}"
                if abs(v) < 1000: return f"{v:.2f}"
                return f"{v:.0f}"
            return str(v)
        
        for key in ["accuracy", "knowledge_density", "avg_latency_us", 
                     "p50_latency_us", "p99_latency_us",
                     "store_throughput", "throughput",
                     "forgetting_rate", "consolidation_gain",
                     "accuracy_per_kb", "memory_bytes", "mib_score"]:
            if key in results:
                label = key.replace("_", " ").title()
                print(f"  {label:30s} {fmt(results[key]):>15s}")
        
        print(f"\n  Final MIB Score: {results.get('mib_score', 0):.4f}")
        print(f"{'='*65}")
