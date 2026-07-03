"""
metrics/engine.py — Official BSM metric computation.

Defined in BSM Foundation Specification v1.0, Part VII.
"""

import numpy as np
from typing import List, Dict, Any


def compute_accuracy(correct: int, total: int) -> float:
    """Return ACC = correct / total."""
    return correct / max(total, 1)


def compute_recall_at_k(predictions: List[int], targets: List[int], k: int) -> float:
    """Return recall@k."""
    if not predictions or not targets:
        return 0.0
    hits = sum(1 for p, t in zip(predictions, targets) if t in p[:k])
    return hits / len(targets)


def compute_knowledge_density(n_classes: int, accuracy: float,
                              memory_bytes: int) -> float:
    """Return Knowledge Density in bits/byte.

    KD = (log2(n_classes) - H_posterior) / memory_bytes

    where H_posterior is approximated from accuracy.
    """
    if memory_bytes <= 0:
        return 0.0
    h_prior = np.log2(n_classes) if n_classes > 1 else 1.0
    # Approximate posterior entropy from accuracy
    p_correct = accuracy
    p_incorrect = max(1 - accuracy, 1e-10)
    h_posterior = -(p_correct * np.log2(p_correct) +
                    p_incorrect * np.log2(p_incorrect / max(n_classes - 1, 1)))
    return (h_prior - h_posterior) / memory_bytes


def compute_participation_ratio(states: np.ndarray) -> float:
    """Return Participation Ratio from a set of binary states.

    states: (N, D) int8 array in {-1, +1}
    """
    N, D = states.shape
    if N < 2:
        return 1.0
    cov = np.cov(states.astype(np.float32).T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) == 0:
        return 1.0
    return float(np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2))


def compute_gini(mi_per_bit: np.ndarray) -> float:
    """Return Gini coefficient for bit-wise mutual information."""
    mi = mi_per_bit.flatten()
    if mi.sum() == 0:
        return 0.0
    mi_sorted = np.sort(mi)
    n = len(mi_sorted)
    cumsum = np.cumsum(mi_sorted)
    return float(1 - 2 * np.sum(cumsum) / (n * mi_sorted.sum()))


def compute_latency_stats(times_us: np.ndarray) -> Dict[str, float]:
    """Return p50, p99, mean latency from an array of µs timings."""
    return {
        "latency_mean_us": float(np.mean(times_us)),
        "latency_p50_us": float(np.median(times_us)),
        "latency_p99_us": float(np.percentile(times_us, 99)),
    }
