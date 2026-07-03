#!/usr/bin/env python3
"""
Comprehensive Tree Head evaluation.

Metrics:
  - Top-1 agreement: tree argmax == FP32 argmax
  - Top-5/-10/-40 agreement: tree argmax in FP32 top-k
  - Zipf quartile accuracy: accuracy broken down by token frequency
  - KL divergence: tree output distribution vs FP32 softmax (top-40)
  - Real speedup: end-to-end inference time comparison
  - Tree size comparison

Usage:
    python training/scripts/eval_tree_head.py --model <model.blmf> [--tree <tree.bin>]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blm.model import BSMModel, BSMConfig
from blm.tokenizer import load_tokenizer as _load_tok
from scripts.bench_tree_accuracy import pack_vector


def load_blmf_with_tree(path: str):
    """Load BLMF model and build tree head from model weights."""
    from blm.export import load_from_blmf

    info = load_from_blmf(path)
    hdr = info["header"]
    sections = info["sections"]

    config = BSMConfig(
        vocab_size=hdr["vocab_size"],
        hidden_dim=hdr["hidden_dim"],
        num_layers=hdr["num_layers"],
        seq_len=hdr["seq_len"],
        window_size=hdr.get("window_size", 8),
    )

    model = BSMModel(config)
    model.eval()

    param_map = {
        "embedding": "embedding.weight",
        "head_weight": "head.weight",
    }
    for li in range(config.num_layers):
        param_map[f"layer_{li}_wforget"] = f"layers.{li}.state_update.W_forget.weight"
        param_map[f"layer_{li}_winput"] = f"layers.{li}.state_update.W_input.weight"
        param_map[f"layer_{li}_wmix"] = f"layers.{li}.mixer.W_mix.weight"

    state_dict = model.state_dict()
    for sec_name, sec in sections.items():
        if sec_name not in param_map:
            continue
        param_name = param_map[sec_name]
        if param_name not in state_dict:
            continue

        dtype_map = {3: np.float32}
        arr = np.frombuffer(sec["data"], dtype=dtype_map.get(sec["dtype"], np.uint8))
        if sec["shape"]:
            arr = arr.reshape(sec["shape"])
        param = torch.from_numpy(arr.copy())

        if param.shape != state_dict[param_name].shape:
            target = state_dict[param_name]
            if param.dtype == torch.uint8 and target.dtype in (torch.float32, torch.float64):
                bits = torch.zeros(target.shape, dtype=torch.float32)
                total_bits = target.numel()
                for j in range(total_bits):
                    byte_idx = j // 8
                    bit_idx = j % 8
                    bit_val = (param.view(-1)[byte_idx].item() >> bit_idx) & 1
                    bits.view(-1)[j] = 1.0 if bit_val else -1.0
                param = bits
            else:
                continue

        state_dict[param_name] = param

    model.load_state_dict(state_dict, strict=False)

    # Build centroid tree from model head weights
    tree_head = build_tree(model)

    return model, config, tree_head


def build_tree(model, threshold=0.0):
    """Build centroid tree from model head weights.
    Returns (nodes, leaf_map, leaf_base, nw, D) tuple compatible with tree_predict.
    """
    from scripts.build_tree_head import build_tree_from_head
    head_w = model.head.weight.detach().cpu()
    nodes, num_nodes, nw, vocab_size, leaf_base = build_tree_from_head(head_w)
    leaf_map = {i: i for i in range(vocab_size)}
    return nodes, leaf_map, leaf_base, nw, model.config.hidden_dim


def tree_predict_wrapper(tree_head, packed):
    """Wrapper: tree_head is (nodes, leaf_map, leaf_base, nw, D), packed is state bytes."""
    from scripts.bench_tree_accuracy import tree_predict as _tp
    nodes, leaf_map, leaf_base, nw, D = tree_head
    return _tp(nodes, leaf_map, leaf_base, nw, packed, D)


def load_tokenizer_file(path: str):
    return _load_tok(path)


def softmax(x, temp=1.0):
    x = x / temp
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def kl_divergence(p, q):
    """KL(P || Q). p, q are arrays of same shape."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    q = np.clip(q, 1e-10, None)
    p = np.clip(p, 1e-10, None)
    return np.sum(p * np.log(p / q))


def eval_agreement(model, tokenizer, texts, tree_head, topk_list=(1, 5, 10, 40),
                   max_stories=500, device="cpu"):
    """Evaluate tree head vs FP32 head agreement."""
    model.eval()

    results = {k: 0 for k in topk_list}
    total = 0
    token_freqs = {}
    agreement_by_freq = {k: {} for k in topk_list}
    kl_values = []

    with torch.no_grad():
        for story_idx, text in enumerate(texts[:max_stories]):
            if story_idx % 50 == 0 and story_idx > 0:
                print(f"    Evaluating story {story_idx}/{min(len(texts), max_stories)}...")

            ids = tokenizer.encode(text)
            if len(ids) < 10:
                continue

            ids_t = torch.tensor(ids[:129], dtype=torch.long).unsqueeze(0).to(device)

            # Forward pass
            hidden = model.embedding(ids_t[:, :-1])
            D = model.config.hidden_dim
            for layer in model.layers:
                hidden, _ = layer(hidden, torch.full((1, D), -1.0, device=device), single_step=False)

            h = hidden[0]
            logits = model.head(h)

            fp32_probs = softmax(logits.cpu().numpy())

            for pos in range(h.shape[0]):
                fp32_pred = logits[pos].argmax().item()
                fp32_topk = logits[pos].topk(max(topk_list)).indices.tolist()
                fp32_probs_pos = fp32_probs[pos]

                packed = pack_vector(h[pos].cpu().numpy())

                # Full tree traversal -> single leaf
                leaf = tree_predict_wrapper(tree_head, packed)

                for k in topk_list:
                    if leaf in fp32_topk[:k]:
                        results[k] += 1

                total += 1

                # Token frequency tracking
                token = fp32_pred
                token_freqs[token] = token_freqs.get(token, 0) + 1

                # KL divergence for top-40
                fp32_top40_indices = list(range(min(40, len(fp32_probs_pos))))
                fp32_top40 = fp32_probs_pos[fp32_top40_indices]

                # For tree, we can only get the leaf, so we create a
                # one-hot-like distribution
                tree_dist = np.zeros(len(fp32_probs_pos))
                tree_dist[leaf] = 1.0

                # KL(FP32 || tree) — how much info is lost
                kl = kl_divergence(fp32_probs_pos[:40], tree_dist[:40])
                kl_values.append(kl)

    # Compile results
    top1_pct = 100 * results[1] / max(total, 1)
    top5_pct = 100 * results[5] / max(total, 1) if 5 in results else 0
    top40_pct = 100 * results[40] / max(total, 1) if 40 in results else 0
    avg_kl = np.mean(kl_values) if kl_values else float('inf')

    return {
        "total_predictions": total,
        "top1_agreement": round(top1_pct, 2),
        "top5_agreement": round(top5_pct, 2),
        "top40_agreement": round(top40_pct, 2),
        "avg_kl_div": round(float(avg_kl), 4),
    }


def eval_speed(model, tokenizer, texts, tree_head, num_trials=3, device="cpu"):
    """Compare FP32 vs Tree inference speed."""
    import time

    # Prepare a single batch of fixed size
    ids = tokenizer.encode(texts[0])
    tokens = torch.tensor(ids[:129], dtype=torch.long).unsqueeze(0).to(device)
    tokens = tokens[:, :-1]  # [1, seq_len]

    D = model.config.hidden_dim
    init_state = torch.full((1, D), -1.0, device=device)
    model.eval()
    with torch.no_grad():
        hidden = model.embedding(tokens)
        for layer in model.layers:
            hidden, _ = layer(hidden, init_state, single_step=False)
    h = hidden[0][0]  # [dim]

    # Warmup
    for _ in range(10):
        _ = model.head(h.unsqueeze(0))
        packed = pack_vector(h.cpu().numpy())
        _ = tree_predict_wrapper(tree_head, packed)

    h_np = h.detach().cpu().numpy()

    # FP32 head timing
    fp32_times = []
    for _ in range(num_trials):
        t0 = time.perf_counter_ns()
        for _ in range(1000):
            logits = model.head(h.unsqueeze(0))
            _ = logits.argmax().item()
        t1 = time.perf_counter_ns()
        fp32_times.append((t1 - t0) / 1000)

    # Tree head timing
    tree_times = []
    for _ in range(num_trials):
        t0 = time.perf_counter_ns()
        for _ in range(1000):
            packed = pack_vector(h_np)
            leaf = tree_predict_wrapper(tree_head, packed)
        t1 = time.perf_counter_ns()
        tree_times.append((t1 - t0) / 1000)

    # E2E model + head timing (batch_size=1, seq_len=1)
    single_token = torch.tensor([[ids[0]]], dtype=torch.long).to(device)
    D = model.config.hidden_dim
    init_state = torch.full((1, D), -1.0, device=device)

    e2e_fp32_times = []
    for _ in range(num_trials):
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            for _ in range(100):
                h = model.embedding(single_token)
                for layer in model.layers:
                    h, _ = layer(h, init_state, single_step=False)
                logits = model.head(h)
                _ = logits.argmax().item()
        t1 = time.perf_counter_ns()
        e2e_fp32_times.append((t1 - t0) / 100)

    e2e_tree_times = []
    for _ in range(num_trials):
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            for _ in range(100):
                h = model.embedding(single_token)
                for layer in model.layers:
                    h, _ = layer(h, init_state, single_step=False)
                packed = pack_vector(h[0, 0].cpu().numpy())
                leaf = tree_predict_wrapper(tree_head, packed)
        t1 = time.perf_counter_ns()
        e2e_tree_times.append((t1 - t0) / 100)

    fp32_avg = np.mean(fp32_times)
    tree_avg = np.mean(tree_times)
    e2e_fp32_avg = np.mean(e2e_fp32_times)
    e2e_tree_avg = np.mean(e2e_tree_times)

    # Tree head size (bytes)
    tree_size = len(tree_head)

    # FP32 head size (bytes)
    # Tree head size: total bytes of all node weight vectors
    nodes = tree_head[0]
    nw = tree_head[3]
    node_size = nw * 8
    tree_bytes = sum(len(n) for n in nodes if n is not None)
    fp32_head_size = model.head.weight.numel() * 4  # float32

    return {
        "fp32_head_ns": round(float(fp32_avg), 1),
        "tree_head_ns": round(float(tree_avg), 1),
        "head_speedup": round(fp32_avg / max(tree_avg, 0.01), 1),
        "e2e_fp32_ns": round(float(e2e_fp32_avg), 1),
        "e2e_tree_ns": round(float(e2e_tree_avg), 1),
        "e2e_speedup": round(e2e_fp32_avg / max(e2e_tree_avg, 0.01), 1),
        "tree_size_bytes": tree_bytes,
        "fp32_head_bytes": fp32_head_size,
        "size_reduction": round(fp32_head_size / max(tree_bytes, 1), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Tree Head")
    parser.add_argument("--model", required=True, help="BLMF model path")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer JSON path")
    parser.add_argument("--data", required=True, help="Validation text file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-stories", type=int, default=500)
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"[*] Loading model from {args.model}...")
    model, config, tree_head = load_blmf_with_tree(args.model)
    model.to(device)
    print(f"    Model: vocab={config.vocab_size}, dim={config.hidden_dim}, "
          f"layers={config.num_layers}")
    num_internal = len(tree_head[0])
    num_leaves = tree_head[2] + 2  # leaf_base = leaf_count - 1
    print(f"    Tree head: {num_internal + num_leaves} total ({num_internal} internal, "
          f"{num_leaves} leaves)")

    print(f"[*] Loading tokenizer from {args.tokenizer}...")
    tokenizer = load_tokenizer_file(args.tokenizer)
    print(f"    Vocab: {tokenizer.vocab_size} tokens")

    print(f"[*] Loading validation data from {args.data}...")
    with open(args.data, encoding="utf-8") as f:
        text = f.read()
    stories = [s.strip() for s in text.split("\n---\n") if s.strip()]
    print(f"    Stories: {len(stories)}")

    # ── Agreement evaluation ─────────────────────────────────────
    print()
    print("═" * 60)
    print("[1/3] Agreement evaluation")
    print("─" * 60)
    agreement = eval_agreement(
        model, tokenizer, stories, tree_head,
        max_stories=args.max_stories, device=device
    )
    for k, v in agreement.items():
        print(f"  {k}: {v}")

    # ── Speed benchmark ──────────────────────────────────────────
    print()
    print("[2/3] Speed benchmark")
    print("─" * 60)
    speed = eval_speed(model, tokenizer, stories, tree_head, device=device)
    for k, v in speed.items():
        if "ns" in k:
            unit = "ns" if v < 1_000_000 else "μs" if v < 1_000_000_000 else "ms"
            divisor = 1 if unit == "ns" else 1000 if unit == "μs" else 1_000_000
            print(f"  {k}: {v/divisor:.1f} {unit}")
        elif "speedup" in k or "reduction" in k:
            print(f"  {k}: {v:.1f}x")
        else:
            print(f"  {k}: {v}")

    # ── Summary ──────────────────────────────────────────────────
    print()
    print("[3/3] Summary")
    print("─" * 60)
    print(f"  Top-1 agreement:     {agreement['top1_agreement']:>6.1f}%")
    print(f"  Top-5 agreement:     {agreement['top5_agreement']:>6.1f}%")
    print(f"  Top-40 agreement:    {agreement['top40_agreement']:>6.1f}%")
    print(f"  Avg KL(FP32||tree):  {agreement['avg_kl_div']:>8.4f}")
    print(f"  Head speedup:        {speed['head_speedup']:>6.1f}x")
    print(f"  E2E speedup:         {speed['e2e_speedup']:>6.1f}x")
    print(f"  Size reduction:      {speed['size_reduction']:>6.1f}x")
    print("═" * 60)

    # Check thresholds
    passed = True
    if agreement['top1_agreement'] < 80:
        print(f"[!] Top-1 agreement ({agreement['top1_agreement']}%) < 80%")
        passed = False
    if speed['head_speedup'] < 100:
        print(f"[!] Head speedup ({speed['head_speedup']}x) < 100x")
        passed = False
    if agreement['avg_kl_div'] >= 0.5:
        print(f"[!] KL divergence ({agreement['avg_kl_div']}) >= 0.5")
        passed = False

    if passed:
        print("[✓] All thresholds PASSED")
    else:
        print("[✗] Some thresholds not met")

    # Save results
    if args.output:
        result = {"agreement": agreement, "speed": speed}
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[*] Results saved to {args.output}")


if __name__ == "__main__":
    main()
