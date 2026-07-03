"""
Train the binary tree head node weights via STE.

Strategy:
  1. Collect (hidden_state, target_token) pairs from forward pass
  2. For EACH internal node, find samples that pass through it
  3. Each sample has a ground-truth LEFT/RIGHT at that node
  4. Train each node's binary weight vector with BCE + STE

This produces a tree head that actually learns to separate tokens
at each node, rather than relying on centroid approximations.
"""

import argparse
import json
import struct
import sys
import os
import io
import time
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

from blm.binary_ops import binarize_weight


class BinaryNodeWeight(nn.Module):
    """Single node weight with STE binarization."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.raw = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def forward(self) -> torch.Tensor:
        return binarize_weight(self.raw)


def next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def collect_data(model, tokenizer, data_path, max_steps=5000):
    """Run model on data, collect (hidden_state, target_token) pairs."""
    from blm.model import BSMModel
    device = next(model.parameters()).device

    # Only read enough chars for max_steps + safety margin
    # Compression ratio is ~4 chars/token for TinyStories
    chars_needed = max_steps * 8  # generous margin
    with open(data_path, encoding="utf-8") as f:
        text = f.read(chars_needed)

    encoded = tokenizer.encode(text)
    print(f"  Read: {len(text)} chars, {len(encoded)} tokens (needed {max_steps+10})")

    hidden_states = []
    targets = []

    with torch.no_grad():
        states = model.init_states(1)
        for i in range(min(len(encoded) - 1, max_steps)):
            token_id = int(encoded[i])
            target_id = int(encoded[i + 1])
            if target_id in (0, 1, 2, 3):
                continue

            x = model.embedding(torch.tensor([[token_id]], device=device))
            x = x.squeeze(1)

            new_states = []
            for layer_idx, layer in enumerate(model.layers):
                x, new_state = layer(x, states[layer_idx], single_step=True)
                new_states.append(new_state)
            states = new_states

            hidden_states.append(x.squeeze(0).cpu())
            targets.append(target_id)

    H = torch.stack(hidden_states)
    T = torch.tensor(targets, dtype=torch.long)
    print(f"  Collected {len(H)} (hidden, target) pairs")
    return H, T


def compute_token_weights(T, vocab_size, scheme="sqrt_inv"):
    """Compute per-token frequency weights for all tokens in T.

    Args:
        T: [N] tensor of token IDs
        scheme: 'sqrt_inv' = 1/sqrt(freq), 'log_inv' = log(N/freq)

    Returns:
        [N] tensor of weights (one per sample in T)
    """
    N = len(T)
    freqs = torch.zeros(vocab_size)
    for t in T:
        freqs[t] += 1.0
    freqs = freqs.clamp(min=1.0)
    
    if scheme == "sqrt_inv":
        weights = 1.0 / freqs.sqrt()
    elif scheme == "log_inv":
        weights = (N / freqs).log().clamp(min=0.1)
    elif scheme == "uniform":
        weights = torch.ones(vocab_size)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    
    weights = weights / weights.mean()
    return weights[T.long()]


def train_node(node_idx, lo, hi, H, T, hidden_dim, vocab_size,
               epochs=100, lr=0.01, batch_size=256,
               init_weight=None, sample_weights=None):
    """Train a single internal node's binary weight vector.

    Args:
        init_weight: optional [D] float tensor to initialize weights from
        sample_weights: optional [N_node] tensor of per-sample loss weights

    Returns:
        (best_raw_weights, accuracy, n_samples, stats_dict) or None
    """
    mid = (lo + hi) // 2

    mask_left = (T >= lo) & (T < min(mid, vocab_size))
    mask_right = (T >= mid) & (T < min(hi, vocab_size))
    mask = mask_left | mask_right

    n_samples = mask.sum().item()
    if n_samples < 2:
        return None

    h = H[mask]
    t = T[mask]
    labels = (t >= mid).float()
    left_count = mask_left[mask].sum().item()
    right_count = mask_right[mask].sum().item()

    n_left = (labels == 0).sum().item()
    n_right = (labels == 1).sum().item()
    p_left = n_left / n_samples if n_samples > 0 else 0.5
    p_right = n_right / n_samples if n_samples > 0 else 0.5
    if p_left > 0 and p_right > 0:
        entropy = -(p_left * math.log2(p_left) + p_right * math.log2(p_right))
    else:
        entropy = 0.0

    weight = BinaryNodeWeight(hidden_dim)
    if init_weight is not None:
        with torch.no_grad():
            weight.raw.copy_(init_weight)
    optimizer = torch.optim.Adam(weight.parameters(), lr=lr)
    best_loss = float('inf')
    best_weights = None
    stall_count = 0

    n_batches = max(1, n_samples // batch_size)
    indices = torch.randperm(n_samples)

    grad_norms = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for bi in range(n_batches):
            start = bi * batch_size
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]

            h_batch = h[batch_idx]
            label_batch = labels[batch_idx]

            w = weight()
            dot = (w.unsqueeze(0) * h_batch).sum(dim=1) / hidden_dim

            if sample_weights is not None:
                sw_batch = sample_weights[mask][batch_idx]
                loss = nn.functional.binary_cross_entropy_with_logits(
                    dot, label_batch, weight=sw_batch)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(dot, label_batch)

            optimizer.zero_grad()
            loss.backward()
            
            gn = weight.raw.grad.norm().item() if weight.raw.grad is not None else 0.0
            grad_norms.append(gn)
            
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= n_batches

        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            best_weights = weight.raw.detach().clone()
            stall_count = 0
        else:
            stall_count += 1

        if stall_count >= 20:
            break

    acc = eval_node_accuracy(best_weights, h, labels, hidden_dim)
    avg_gn = float(np.mean(grad_norms)) if grad_norms else 0.0

    stats = {
        "n_samples": n_samples,
        "n_left": n_left,
        "n_right": n_right,
        "p_left": round(p_left * 100, 1),
        "p_right": round(p_right * 100, 1),
        "entropy": round(entropy, 4),
        "accuracy": round(acc * 100, 1),
        "avg_grad_norm": round(avg_gn, 4),
        "left_count": int(left_count),
        "right_count": int(right_count),
    }
    return best_weights, acc, n_samples, stats


def eval_node_accuracy(raw_w, H, labels, hidden_dim):
    """Compute classification accuracy for a trained node."""
    with torch.no_grad():
        w = torch.where(raw_w >= 0, 1.0, -1.0)
        dot = (w.unsqueeze(0) * H).sum(dim=1) / hidden_dim
        pred = (dot > 0).float()
        acc = (pred == labels).float().mean().item()
    return acc


def train_tree(model, H, T, vocab_size, hidden_dim, model_head_w=None,
               epochs=200, lr=0.01, weighting_scheme=None):
    """Train all internal nodes and return packed tree weights.

    Args:
        model_head_w: [V, D] numpy array of FP32 head weights (for centroid init)
        weighting_scheme: None, 'sqrt_inv', or 'log_inv' — frequency weighting
    """
    leaf_count = next_pow2(vocab_size)
    num_nodes = 2 * leaf_count - 1
    leaf_base = leaf_count - 1
    nw = (hidden_dim + 63) // 64
    nodes = [None] * num_nodes

    epoch_stats = {}

    # Precompute per-sample frequency weights
    sample_weights_all = None
    if weighting_scheme is not None:
        sample_weights_all = compute_token_weights(T, vocab_size, scheme=weighting_scheme)
        print(f"  Weighting: {weighting_scheme}, weights range: "
              f"{sample_weights_all.min().item():.3f} - {sample_weights_all.max().item():.3f}")

    # Precompute centroids for each node from head weights
    centroids = {}
    if model_head_w is not None:
        def compute_centroid(lo, hi):
            mid = (lo + hi) // 2
            left_lo, left_hi = lo, min(mid, vocab_size)
            right_lo, right_hi = mid, min(hi, vocab_size)
            if left_lo >= left_hi or right_lo >= right_hi:
                return None
            centroid = np.zeros(hidden_dim, dtype=np.float32)
            centroid += model_head_w[right_lo:right_hi].mean(axis=0)
            centroid -= model_head_w[left_lo:left_hi].mean(axis=0)
            return torch.from_numpy(centroid).float()

        def compute_centroids_recursive(node_idx, lo, hi):
            if node_idx >= num_nodes or hi - lo <= 1:
                return
            mid = (lo + hi) // 2
            c = compute_centroid(lo, hi)
            if c is not None:
                centroids[node_idx] = c
            compute_centroids_recursive(2 * node_idx + 1, lo, mid)
            compute_centroids_recursive(2 * node_idx + 2, mid, hi)

        compute_centroids_recursive(0, 0, leaf_count)

    # Collect node stats for entropy report
    node_stats = {}
    entropy_report = []

    def train_node_recursive(node_idx, lo, hi, depth=0):
        indent = "  " * depth
        if node_idx >= num_nodes or hi - lo <= 1:
            return

        mid = (lo + hi) // 2
        left_lo, left_hi = lo, min(mid, vocab_size)
        right_lo, right_hi = mid, min(hi, vocab_size)

        if left_lo >= left_hi or right_lo >= right_hi:
            train_node_recursive(2 * node_idx + 1, lo, mid, depth + 1)
            train_node_recursive(2 * node_idx + 2, mid, hi, depth + 1)
            return

        init_w = centroids.get(node_idx, None)
        sw_node = sample_weights_all if sample_weights_all is not None else None
        result = train_node(node_idx, lo, hi, H, T, hidden_dim, vocab_size,
                            epochs=epochs, lr=lr, init_weight=init_w,
                            sample_weights=sw_node)

        if result is not None:
            raw_w, acc, n_samp, stats = result
            node_w = torch.where(raw_w >= 0, 1.0, -1.0)
            packed = bytearray(nw * 8)
            for j in range(hidden_dim):
                if node_w[j] > 0:
                    w_idx = j // 64
                    b = j % 64
                    import struct as st
                    curr = st.unpack('<Q', packed[w_idx*8:(w_idx+1)*8])[0]
                    packed[w_idx*8:(w_idx+1)*8] = st.pack('<Q', curr | (1 << b))
            nodes[node_idx] = bytes(packed)

            # Compact per-node print
            print(f"{indent}N{node_idx:4d} [{lo:3d},{hi:3d}) "
                  f"acc={stats['accuracy']:.1f}% "
                  f"L={stats['p_left']:.0f}% R={stats['p_right']:.0f}% "
                  f"H={stats['entropy']:.2f} "
                  f"n={n_samp:4d}")

            epoch_stats[node_idx] = {
                'range': [lo, hi],
                'accuracy': stats['accuracy'],
                'samples': n_samp,
                'loss': round(1 - acc, 4),
                'entropy': stats['entropy'],
                'left_pct': stats['p_left'],
                'right_pct': stats['p_right'],
            }
            node_stats[node_idx] = stats
        else:
            print(f"{indent}N{node_idx:4d} [{lo:3d},{hi:3d}) SKIP (n<2)")

        train_node_recursive(2 * node_idx + 1, lo, mid, depth + 1)
        train_node_recursive(2 * node_idx + 2, mid, hi, depth + 1)

    train_node_recursive(0, 0, leaf_count)

    # Print entropy report
    print()
    print("═" * 60)
    print("ROUTING ENTROPY REPORT")
    print("═" * 60)
    print(f"{'Node':<6} {'Range':<14} {'Samples':<8} {'L%':<6} {'R%':<6} "
          f"{'H':<8} {'Acc%':<6} {'|g|':<6} {'Status':<10}")
    print("─" * 70)
    
    high_entropy_count = 0
    low_data_count = 0
    for ni in sorted(node_stats.keys()):
        s = node_stats[ni]
        r = epoch_stats[ni]['range']
        status = "OK"
        if s['entropy'] > 0.99:
            status = "NEAR-RANDOM"
            high_entropy_count += 1
        if s['n_samples'] < 10:
            status = "LOW-DATA"
            low_data_count += 1
        if s['accuracy'] < 55:
            status = "WEAK"
        
        print(f"N{ni:<5} [{r[0]:3d},{r[1]:3d}) "
              f"{s['n_samples']:<8} {s['p_left']:<5.0f}% {s['p_right']:<5.0f}% "
              f"{s['entropy']:<8.3f} {s['accuracy']:<5.1f}% "
              f"{s['avg_grad_norm']:<6.3f} {status:<10}")
    
    print("─" * 70)
    print(f"  Total trained: {len(node_stats)} nodes")
    print(f"  Near-random (H>0.99): {high_entropy_count}")
    print(f"  Low-data (n<10): {low_data_count}")
    
    good = sum(1 for s in node_stats.values() if s['accuracy'] >= 70)
    weak = sum(1 for s in node_stats.values() if s['accuracy'] < 55)
    print(f"  Good (acc>=70%): {good}  Weak (acc<55%): {weak}")

    # Serialize with bitmask for trained/untrained distinction
    buf = io.BytesIO()
    import struct as st
    buf.write(st.pack('<IIII', num_nodes, nw, vocab_size, leaf_base))
    for i in range(num_nodes):
        if nodes[i] is not None:
            buf.write(nodes[i])
        else:
            buf.write(b'\x00' * (nw * 8))
    # Append a bitmask at the end: 1 = trained, 0 = untrained
    bitmask = bytearray((num_nodes + 7) // 8)
    for i in range(num_nodes):
        if nodes[i] is not None:
            bitmask[i // 8] |= (1 << (i % 8))
    buf.write(bytes(bitmask))

    return buf.getvalue(), epoch_stats, nodes, leaf_base, nw


def evaluate_tree(H, T, nodes, leaf_base, nw, vocab_size, hidden_dim):
    """Evaluate tree head accuracy against FP32 reference."""
    N = len(H)
    correct_1 = 0
    correct_5 = 0
    correct_10 = 0

    pred_token_1_list = []
    ref_list = []

    for i in range(N):
        h = H[i]
        t = T[i].item()

        # Pack hidden state
        packed = bytearray(nw * 8)
        for j in range(hidden_dim):
            if h[j] > 0:
                w_idx = j // 64
                b = j % 64
                import struct as st
                curr = st.unpack('<Q', packed[w_idx*8:(w_idx+1)*8])[0]
                packed[w_idx*8:(w_idx+1)*8] = st.pack('<Q', curr | (1 << b))
        state_bytes = bytes(packed)

        # Greedy tree
        node = 0
        for _ in range(20):
            if node >= len(nodes) or nodes[node] is None or len(nodes[node]) < nw * 8:
                break
            dot = 0
            for w in range(nw):
                off = w * 8
                import struct as st
                nw_word = st.unpack('<Q', nodes[node][off:off+8])[0]
                sw_word = st.unpack('<Q', state_bytes[off:off+8])[0]
                match = (~(nw_word ^ sw_word)) & 0xFFFFFFFFFFFFFFFF
                dot += 2 * match.bit_count() - 64
            if dot > 0:
                node = 2 * node + 2
            else:
                node = 2 * node + 1
        leaf_idx = node - leaf_base
        pred = max(0, min(leaf_idx, vocab_size - 1))

        if pred == t:
            correct_1 += 1
        pred_token_1_list.append(pred)
        ref_list.append(t)

    acc1 = correct_1 / N * 100

    return {
        'total': N,
        'correct_top1': correct_1,
        'accuracy_top1': round(acc1, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/tmp/bsm_bench_v4/checkpoint_final.pt")
    parser.add_argument("--tokenizer", default="/tmp/bench_v4_tokenizer.json")
    parser.add_argument("--data", default="testdata/tiny_corpus.txt")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--output", default="/tmp/trained_tree_head.bin")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--compare", action="store_true", default=True,
                        help="Compare trained vs K-means accuracy")
    args = parser.parse_args()

    print("=" * 60)
    print("BINARY TREE HEAD TRAINING")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading model...")
    from blm.model import BSMModel, BSMConfig
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg_data = checkpoint['config']
    if isinstance(cfg_data, dict):
        cfg = BSMConfig(**cfg_data)
    else:
        cfg = cfg_data
    model = BSMModel(cfg)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Vocab: {cfg.vocab_size}, Hidden: {cfg.hidden_dim}, Layers: {cfg.num_layers}")
    print(f"  Leaf count: {next_pow2(cfg.vocab_size)}")

    head_w = model.head.weight.data.cpu().numpy()

    # Load tokenizer
    from blm.tokenizer import BPETokenizer
    tok = BPETokenizer.load(args.tokenizer)

    # Collect data
    print(f"\n[2/4] Collecting training data from: {args.data}")
    H, T = collect_data(model, tok, args.data, max_steps=args.steps)

    # Train tree
    V = cfg.vocab_size
    D = cfg.hidden_dim
    nw = (D + 63) // 64

    print(f"\n[3/4] Training tree head ({V} vocab, {D} dim, {nw} words/node)...")
    print(f"  lr={args.lr}, epochs={args.epochs}")
    print()

    t0 = time.time()
    tree_bytes, stats, nodes, leaf_base, learned_nw = train_tree(
        model, H, T, V, D,
        model_head_w=head_w,
        epochs=args.epochs,
        lr=args.lr,
    )
    t_train = time.time() - t0

    # Save trained tree
    with open(args.output, 'wb') as f:
        f.write(tree_bytes)
    tree_size = len(tree_bytes)
    fp32_size = V * D * 4

    print(f"\n  Training time: {t_train:.1f}s")
    print(f"  Tree size: {tree_size:,} bytes (vs FP32 {fp32_size:,} bytes, {fp32_size/tree_size:.0f}x smaller)")

    # Evaluate trained tree
    print(f"\n[4/4] Evaluating...")
    print()

    from scripts.bench_tree_accuracy import build_tree_kmeans, build_tree_random_bits, tree_predict_topk

    # Build K-means baseline
    print("Building reference trees...")
    protos_head = head_w
    nodes_rnd, leaf_map_rnd, lc_rnd, nw_rnd = build_tree_random_bits(protos_head, list(range(V)))
    nodes_km, leaf_map_km, lc_km, nw_km = build_tree_kmeans(protos_head, list(range(V)))

    # Leaf map for trained tree (simple: leaf i -> token i)
    leaf_map_trained = {i: i for i in range(V)}

    print()
    print(f"{'Method':<30} {'Top-1':>8} {'Top-5':>8} {'Top-10':>8} {'Time':>8}")
    print("-" * 62)

    N_eval = min(len(H), 1000)
    H_eval = H[:N_eval]
    T_eval = T[:N_eval]

    for name, n_list, l_map, l_base, l_nw in [
        ("Random bits (token ID)", nodes_rnd, leaf_map_rnd, lc_rnd - 1, nw_rnd),
        ("K-means (head weights)", nodes_km, leaf_map_km, lc_km - 1, nw_km),
        ("TRAINED (STE)", nodes, leaf_map_trained, leaf_base, learned_nw),
    ]:
        top1 = 0
        top5_total = 0
        top10_total = 0
        step_times = []

        for idx in range(N_eval):
            h = H_eval[idx]
            t = T_eval[idx].item()

            packed = bytearray(l_nw * 8)
            for j in range(D):
                if h[j] > 0:
                    w_idx = j // 64
                    b = j % 64
                    import struct as st
                    curr = st.unpack('<Q', packed[w_idx*8:(w_idx+1)*8])[0]
                    packed[w_idx*8:(w_idx+1)*8] = st.pack('<Q', curr | (1 << b))
            state_bytes = bytes(packed)

            t0 = time.perf_counter_ns()
            node = 0
            for _ in range(20):
                if node >= len(n_list) or n_list[node] is None or len(n_list[node]) < l_nw * 8:
                    break
                dot = 0
                for w in range(l_nw):
                    off = w * 8
                    import struct as st
                    nw_word = st.unpack('<Q', n_list[node][off:off+8])[0]
                    sw_word = st.unpack('<Q', state_bytes[off:off+8])[0]
                    match = (~(nw_word ^ sw_word)) & 0xFFFFFFFFFFFFFFFF
                    dot += 2 * match.bit_count() - 64
                if dot > 0:
                    node = 2 * node + 2
                else:
                    node = 2 * node + 1
            leaf_idx = node - l_base
            pred = l_map.get(leaf_idx, 0)

            step_ns = time.perf_counter_ns() - t0
            step_times.append(step_ns)

            if pred == t:
                top1 += 1

            if l_nw == learned_nw and n_list is nodes:
                topk = tree_predict_topk(n_list, l_map, l_base, l_nw, state_bytes, D, k=5)
                if t in topk:
                    top5_total += 1
                topk10 = tree_predict_topk(n_list, l_map, l_base, l_nw, state_bytes, D, k=10)
                if t in topk10:
                    top10_total += 1

        avg_ns = np.mean(step_times) if step_times else 0
        t5_str = f"{top5_total/N_eval*100:6.1f}%" if n_list is nodes else "     N/A"
        t10_str = f"{top10_total/N_eval*100:6.1f}%" if n_list is nodes else "     N/A"

        print(f"{name:<30} {100*top1/N_eval:>7.1f}% {t5_str:>8} {t10_str:>8} {avg_ns:>7.0f}ns")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Training data:  {len(H)} pairs from {args.data}")
    print(f"  Tree size:      {tree_size:,} bytes ({fp32_size/tree_size:.0f}x smaller than FP32)")
    print(f"  Training time:  {t_train:.1f}s")

    print(f"\n  File saved to: {args.output}")


if __name__ == "__main__":
    main()
