"""
Benchmark binary tree head accuracy vs FP32 head.
Three tree construction strategies:
  1. Random bits (token ID bits) — baseline
  2. Hierarchical K-means on head weights
  3. Hierarchical K-means on embedding table

Output: accuracy@1, accuracy@5, accuracy@10 for each method.
"""

import argparse
import struct
import sys
import os
import math
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np


def next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def balanced_kmeans_split(vectors, max_imbalance=2.0, rng_seed=42):
    """
    Split a set of vectors into 2 balanced clusters using K-means.

    Args:
        vectors: [N, D] numpy array
        max_imbalance: max ratio between larger and smaller cluster
        rng_seed: random seed

    Returns:
        (left_indices, right_indices, centroid_diff)
        where indices are positions in the input array (0..N-1)
    """
    n = len(vectors)
    if n <= 2:
        if n <= 1:
            return list(range(n)), [], np.zeros(vectors.shape[1])
        return [0], [1], vectors[0] - vectors[1]

    rng = np.random.RandomState(rng_seed)
    best_labels = None
    best_inertia = float('inf')

    for trial in range(30):
        idx = rng.choice(n, min(2, n), replace=False)
        centers = vectors[idx].copy()

        for _ in range(20):
            dists = np.array([
                np.sum((vectors - c) ** 2, axis=1) for c in centers
            ])
            labels = np.argmin(dists, axis=0)
            if len(np.unique(labels)) < 2:
                break
            new_centers = np.array([
                vectors[labels == k].mean(axis=0) for k in range(2)
            ])
            if np.allclose(centers, new_centers, atol=1e-8):
                break
            centers = new_centers
        else:
            inertia = 0
            for k in range(2):
                mask = labels == k
                if mask.sum() > 0:
                    inertia += np.sum((vectors[mask] - centers[k]) ** 2)
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()

    if best_labels is None:
        best_labels = np.array([0] * (n // 2) + [1] * (n - n // 2))
        rng.shuffle(best_labels)

    left = np.where(best_labels == 0)[0].tolist()
    right = np.where(best_labels == 1)[0].tolist()

    if len(left) < n / max_imbalance or len(right) < n / max_imbalance:
        centered = vectors - vectors.mean(axis=0)
        try:
            _, _, vt = np.linalg.svd(centered.reshape(n, -1), full_matrices=False)
            proj = vectors @ vt[0]
            median = np.median(proj)
            left = np.where(proj <= median)[0].tolist()
            right = np.where(proj > median)[0].tolist()
        except np.linalg.LinAlgError:
            left = list(range(n // 2))
            right = list(range(n // 2, n))

    if not left or not right:
        left = list(range(n // 2))
        right = list(range(n // 2, n))

    # Centroid difference
    centroid_left = vectors[left].mean(axis=0) if left else np.zeros(vectors.shape[1])
    centroid_right = vectors[right].mean(axis=0) if right else np.zeros(vectors.shape[1])
    centroid_diff = centroid_right - centroid_left

    return left, right, centroid_diff


def pack_vector(vec):
    """Pack float vector to binary uint64 words. sign(vec) -> bits.

    vec[j] > 0 → bit=1 (+1 in {-1,+1} space)
    vec[j] <= 0 → bit=0 (-1 in {-1,+1} space)
    """
    nw = (len(vec) + 63) // 64
    packed = bytearray(nw * 8)
    for j in range(len(vec)):
        if vec[j] > 0:
            w = j // 64
            b = j % 64
            curr = struct.unpack('<Q', packed[w*8:(w+1)*8])[0]
            packed[w*8:(w+1)*8] = struct.pack('<Q', curr | (1 << b))
    return bytes(packed)


def build_tree_kmeans(prototypes, token_ids):
    """
    Build binary tree via hierarchical balanced K-means.

    Args:
        prototypes: [V, D] numpy array — token representation vectors
        token_ids: list of V token IDs (in original order)

    Returns:
        nodes: list of packed binary weight bytes (heap-indexed)
        leaf_to_token: dict mapping leaf_index → token_id
    """
    V = len(token_ids)
    leaf_count = next_pow2(V)
    num_nodes = 2 * leaf_count - 1
    leaf_base = leaf_count - 1
    D = prototypes.shape[1]
    nw = (D + 63) // 64
    node_size = nw * 8

    nodes = [None] * num_nodes
    leaf_map = {}  # leaf_index -> token_id

    def build_node(node_idx, indices, level=0):
        """indices: positions in the prototypes array for this node's tokens."""
        nonlocal leaf_map

        if node_idx >= num_nodes or level > 30:
            return

        if len(indices) <= 1:
            # Leaf node
            leaf_idx = node_idx - leaf_base
            if 0 <= leaf_idx < leaf_count and indices:
                leaf_map[leaf_idx] = token_ids[indices[0]]
            return

        # Try to split
        vecs = prototypes[indices]
        left_idxs, right_idxs, centroid_diff = balanced_kmeans_split(vecs)

        # Map back to original indices
        left_orig = [indices[i] for i in left_idxs]
        right_orig = [indices[i] for i in right_idxs]

        # Pack node weight
        packed = pack_vector(centroid_diff)
        nodes[node_idx] = packed

        # If either side is empty, make this a leaf
        if not left_orig or not right_orig:
            leaf_idx = node_idx - leaf_base
            if 0 <= leaf_idx < leaf_count and indices:
                leaf_map[leaf_idx] = token_ids[indices[len(indices)//2]]
            return

        build_node(2 * node_idx + 1, left_orig, level + 1)
        build_node(2 * node_idx + 2, right_orig, level + 1)

    build_node(0, list(range(V)), 0)

    return nodes, leaf_map, leaf_count, nw


def build_tree_random_bits(prototypes, token_ids):
    """Build tree using range-based splitting.

    Complete binary tree over leaf indices.
    At each internal node covering [lo, hi):
      mid = (lo + hi) / 2
      left_set = token IDs in [lo, min(mid, V))
      right_set = token IDs in [mid, min(hi, V))
      centroid = mean(left) - mean(right)
    """
    V = len(token_ids)
    leaf_count = next_pow2(V)
    num_nodes = 2 * leaf_count - 1
    leaf_base = leaf_count - 1
    D = prototypes.shape[1]
    nw = (D + 63) // 64
    node_size = nw * 8

    nodes = [None] * num_nodes

    def build_node(node_idx, lo, hi):
        if node_idx >= num_nodes or hi - lo <= 1:
            return

        mid = (lo + hi) // 2
        left_lo, left_hi = lo, min(mid, V)
        right_lo, right_hi = mid, min(hi, V)

        if left_lo >= left_hi or right_lo >= right_hi:
            build_node(2 * node_idx + 1, lo, mid)
            build_node(2 * node_idx + 2, mid, hi)
            return

        centroid = np.zeros(D, dtype=np.float32)
        centroid += prototypes[right_lo:right_hi].mean(axis=0)
        centroid -= prototypes[left_lo:left_hi].mean(axis=0)

        nodes[node_idx] = pack_vector(centroid)

        build_node(2 * node_idx + 1, lo, mid)
        build_node(2 * node_idx + 2, mid, hi)

    build_node(0, 0, leaf_count)

    leaf_map = {i: i for i in range(V)}
    return nodes, leaf_map, leaf_count, nw


def softmax_argmax(logits):
    """Return argmax of logits (FP32 head reference)."""
    return np.argmax(logits)


def tree_predict(nodes, leaf_map, leaf_base, nw, state, D, vocab_size=None):
    """Greedy tree traversal.

    Walks the binary tree from root to leaf. Stops when:
    - node is past the array, None, or has zero/shorter-than-expected weight
    - node is at leaf level (>= leaf_base)
    """
    node = 0
    for _ in range(20):
        if node >= len(nodes) or nodes[node] is None:
            break
        if node >= leaf_base:
            break
        if len(nodes[node]) < nw * 8:
            break

        dot = 0
        for w in range(nw):
            off = w * 8
            node_word = struct.unpack('<Q', nodes[node][off:off+8])[0]
            state_word = struct.unpack('<Q', state[off:off+8])[0]
            match = (~(node_word ^ state_word)) & 0xFFFFFFFFFFFFFFFF
            dot += 2 * match.bit_count() - 64

        if dot > 0:
            node = 2 * node + 2
        else:
            node = 2 * node + 1

    leaf_count = leaf_base + 1
    # If stopped at an untrained internal node (None), map to midpoint of its range
    if node < leaf_base:
        depth = (node + 1).bit_length() - 1  # floor(log2(node+1))
        size = leaf_count >> depth
        start = size * (node + 1 - (1 << depth))
        leaf_idx = start + size // 2
    else:
        leaf_idx = node - leaf_base
    if leaf_idx < 0:
        leaf_idx = 0
    if leaf_idx >= leaf_count:
        leaf_idx = leaf_count - 1
    return leaf_map.get(leaf_idx, 0)


def tree_predict_topk(nodes, leaf_map, leaf_base, nw, state, D, k=5):
    """Beam search over the tree: keep K best paths at each level."""
    beam = [(0, 0)]  # (node_idx, cumulative_score)

    for level in range(20):
        if len(beam) == 0:
            break

        candidates = []
        for node, score in beam:
            if node >= len(nodes) or nodes[node] is None or len(nodes[node]) < nw * 8:
                # Leaf node — keep as result candidate
                leaf_idx = node - leaf_base
                tok = leaf_map.get(leaf_idx)
                if tok is not None:
                    candidates.append((node, score))
                continue

            dot = 0
            for w in range(nw):
                off = w * 8
                node_word = struct.unpack('<Q', nodes[node][off:off+8])[0]
                state_word = struct.unpack('<Q', state[off:off+8])[0]
                match = (~(node_word ^ state_word)) & 0xFFFFFFFFFFFFFFFF
                dot += 2 * match.bit_count() - 64

            left_node = 2 * node + 1
            right_node = 2 * node + 2

            left_score = score + (dot if dot <= 0 else -dot)
            right_score = score + (dot if dot > 0 else -dot)

            candidates.append((left_node, left_score))
            candidates.append((right_node, right_score))

        # Keep top-K
        candidates.sort(key=lambda x: -x[1])
        beam = candidates[:k]

    result = []
    for node, score in beam:
        leaf_idx = node - leaf_base
        tok = leaf_map.get(leaf_idx)
        if tok is not None and tok not in result:
            result.append(tok)
    return result


def run_benchmark(model_path, tokenizer_path, data_path, num_steps=500):
    """Full benchmark: FP32 head vs tree heads."""
    from blm.model import BSMModel, BSMConfig
    from blm.tokenizer import BPETokenizer

    # Load checkpoint
    import json
    with open(model_path, 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu', weights_only=False)

    cfg_data = checkpoint['config']
    if isinstance(cfg_data, dict):
        cfg = BSMConfig(**cfg_data)
    else:
        cfg = cfg_data

    model = BSMModel(cfg)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    tok = BPETokenizer.load(tokenizer_path)

    # Read test data
    with open(data_path) as f:
        test_text = f.read()

    # Encode
    encoded = tok.encode(test_text)
    V = cfg.vocab_size
    D = cfg.hidden_dim
    nw = (D + 63) // 64

    print(f"Model: V={V} D={D} L={cfg.num_layers}")
    print(f"Test tokens: {len(encoded)}")

    # Collect (hidden_state, target_token) pairs
    hidden_states = []
    targets = []

    with torch.no_grad():
        # Process through model
        states = model.init_states(1)
        for i in range(min(len(encoded) - 1, num_steps)):
            token_id = int(encoded[i])
            target_id = int(encoded[i + 1])

            if target_id in (0, 1, 2, 3):  # Skip special tokens
                continue

            # Forward pass
            x = model.embedding(torch.tensor([[token_id]]))
            x = x.squeeze(1)

            new_states = []
            for layer_idx, layer in enumerate(model.layers):
                x, new_state = layer(x, states[layer_idx], single_step=True)
                new_states.append(new_state)
            states = new_states

            # x is the hidden state before the output head [1, D]
            hidden = x.squeeze(0).cpu().numpy()
            hidden_states.append(hidden)
            targets.append(target_id)

    print(f"Collected {len(hidden_states)} (hidden, target) pairs")

    # FP32 head reference
    head_w = model.head.weight.data.cpu().numpy()  # [V, D]

    # Tree construction data
    # Use HEAD WEIGHTS as prototypes (what the model learned to score tokens)
    prototypes = head_w  # [V, D]

    # Also try embedding as prototypes
    emb_export = model.embedding.export_binary()  # [V, D/8] uint8
    emb_np = emb_export.cpu().numpy()
    emb_float = np.zeros((V, D), dtype=np.float32)
    for i in range(V):
        for j in range(D):
            byte_idx, bit_idx = j // 8, j % 8
            emb_float[i, j] = 1.0 if (emb_np[i, byte_idx] >> bit_idx) & 1 else -1.0

    token_ids = list(range(V))

    # Build trees
    print("\nBuilding trees...")

    t0 = time.time()
    nodes_rnd, leaf_map_rnd, lc_rnd, nw_rnd = build_tree_random_bits(prototypes, token_ids)
    t_rnd = time.time() - t0
    print(f"  Random bits tree: {t_rnd:.2f}s")

    t0 = time.time()
    nodes_km_head, leaf_map_km_head, lc_km, nw_km = build_tree_kmeans(prototypes, token_ids)
    t_km_head = time.time() - t0
    print(f"  K-means (head weights) tree: {t_km_head:.2f}s")

    t0 = time.time()
    nodes_km_emb, leaf_map_km_emb, lc_km2, nw_km2 = build_tree_kmeans(emb_float, token_ids)
    t_km_emb = time.time() - t0
    print(f"  K-means (embedding) tree: {t_km_emb:.2f}s")

    leaf_base = lc_rnd - 1  # leaf_count - 1 = index of first leaf

    # Pack each hidden state for binary operations
    packed_states = []
    for h in hidden_states:
        packed = bytearray(nw * 8)
        for j in range(D):
            if h[j] >= 0:
                w = j // 64
                b = j % 64
                curr = struct.unpack('<Q', packed[w*8:(w+1)*8])[0]
                packed[w*8:(w+1)*8] = struct.pack('<Q', curr | (1 << b))
        packed_states.append(bytes(packed))

    # Evaluate
    def evaluate(method_name, nodes, leaf_map):
        correct_1 = 0
        correct_5 = 0
        correct_10 = 0
        total = 0
        step_times = []

        for idx, (h, target) in enumerate(zip(hidden_states, targets)):
            state = packed_states[idx]

            # FP32 reference
            logits = head_w @ h  # [V]
            ref = int(np.argmax(logits))

            # Tree greedy
            t0 = time.perf_counter_ns()
            pred = tree_predict(nodes, leaf_map, leaf_base, nw, state, D)
            t1 = time.perf_counter_ns()
            step_times.append(t1 - t0)

            if pred == ref:
                correct_1 += 1

            # Tree top-5
            topk = tree_predict_topk(nodes, leaf_map, leaf_base, nw, state, D, k=5)
            if ref in topk:
                correct_5 += 1

            # Tree top-10
            top10 = tree_predict_topk(nodes, leaf_map, leaf_base, nw, state, D, k=10)
            if ref in top10:
                correct_10 += 1

            total += 1

        avg_ns = np.mean(step_times) if step_times else 0

        print(f"\n  {method_name}:")
        print(f"    Top-1 accuracy: {correct_1}/{total} = {100*correct_1/total:.1f}%")
        print(f"    Top-5 recall:   {correct_5}/{total} = {100*correct_5/total:.1f}%")
        print(f"    Top-10 recall:  {correct_10}/{total} = {100*correct_10/total:.1f}%")
        print(f"    Avg step:       {avg_ns:.0f} ns")
        return correct_1 / total, correct_5 / total, correct_10 / total

    acc1_rnd, acc5_rnd, acc10_rnd = evaluate("Random bits (current)", nodes_rnd, leaf_map_rnd)
    acc1_kmh, acc5_kmh, acc10_kmh = evaluate("K-means (head weights)", nodes_km_head, leaf_map_km_head)
    acc1_kme, acc5_kme, acc10_kme = evaluate("K-means (embeddings)", nodes_km_emb, leaf_map_km_emb)

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"{'Method':<25} {'Top-1':>8} {'Top-5':>8} {'Top-10':>8}")
    print("-" * 50)
    print(f"{'Random bits':<25} {100*acc1_rnd:>7.1f}% {100*acc5_rnd:>7.1f}% {100*acc10_rnd:>7.1f}%")
    print(f"{'K-means (head)':<25} {100*acc1_kmh:>7.1f}% {100*acc5_kmh:>7.1f}% {100*acc10_kmh:>7.1f}%")
    print(f"{'K-means (emb)':<25} {100*acc1_kme:>7.1f}% {100*acc5_kme:>7.1f}% {100*acc10_kme:>7.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/tmp/bsm_bench_v4/checkpoint_final.pt")
    parser.add_argument("--tokenizer", default="/tmp/bench_v4_tokenizer.json")
    parser.add_argument("--data", default="../testdata/tiny_corpus.txt")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    run_benchmark(args.checkpoint, args.tokenizer, args.data, args.steps)


if __name__ == "__main__":
    main()
