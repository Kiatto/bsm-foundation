#!/usr/bin/env python3
"""
STE-train the binary tree head using hidden states from the trained model.

Collects (hidden_state, target_token) pairs from the model, then trains
each internal node's binary weight vector using Straight-Through Estimator.

Usage:
    python3 training/scripts/train_tree_on_model.py \
        --model checkpoints/tinystories_fast.blmf \
        --tokenizer checkpoints/tinystories_vocab4096.json \
        --data data/tinystories_train.txt \
        --output /tmp/ste_trained_tree.bin
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blm.model import BSMModel, BSMConfig
from blm.tokenizer import load_tokenizer
from blm.export import load_from_blmf
from scripts.train_tree_head import collect_data, train_tree, evaluate_tree


def load_model_from_blmf(path: str) -> tuple:
    """Load model from BLMF file."""
    info = load_from_blmf(path)
    hdr = info["header"]
    sections = info["sections"]

    config = BSMConfig(**{k: hdr[k] for k in ['vocab_size','hidden_dim','num_layers','seq_len','window_size']})
    model = BSMModel(config)

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
    model.eval()
    return model, config


def build_tree_wrapper(model, H, T, vocab_size, hidden_dim, epochs=200, lr=0.01,
                       weighting=None):
    """Train tree and return (nodes, leaf_map, leaf_base, nw, D) tuple."""
    head_w = model.head.weight.data.cpu().numpy()
    
    packed_bytes, epoch_stats, nodes, leaf_base, nw = train_tree(
        model=model, H=H, T=T,
        vocab_size=vocab_size, hidden_dim=hidden_dim,
        model_head_w=head_w,
        epochs=epochs, lr=lr,
        weighting_scheme=weighting,
    )
    
    leaf_map = {i: i for i in range(vocab_size)}
    return nodes, leaf_map, leaf_base, nw, hidden_dim, packed_bytes, epoch_stats


def evaluate_tree_head(model, tokenizer, val_texts, tree_head, device="cpu", max_stories=200):
    """Evaluate tree head agreement vs FP32 argmax."""
    from scripts.bench_tree_accuracy import pack_vector
    from scripts.eval_tree_head import tree_predict_wrapper

    model.eval()
    total = 0
    top1_agree = 0
    top5_agree = 0
    top40_agree = 0

    with torch.no_grad():
        for story_idx, text in enumerate(val_texts[:max_stories]):
            ids = tokenizer.encode(text)
            if len(ids) < 10:
                continue
            ids_t = torch.tensor(ids[:129], dtype=torch.long).unsqueeze(0).to(device)

            hidden = model.embedding(ids_t[:, :-1])
            D = model.config.hidden_dim
            state = torch.full((1, D), -1.0, device=device)
            for layer in model.layers:
                hidden, state = layer(hidden, state, single_step=False)

            h = hidden[0]
            logits = model.head(h)

            for pos in range(h.shape[0]):
                fp32_pred = logits[pos].argmax().item()
                fp32_top5 = logits[pos].topk(5).indices.tolist()
                fp32_top40 = logits[pos].topk(40).indices.tolist()

                packed = pack_vector(h[pos].cpu().numpy())
                leaf = tree_predict_wrapper(tree_head, packed)

                total += 1
                if leaf == fp32_pred:
                    top1_agree += 1
                if leaf in fp32_top5:
                    top5_agree += 1
                if leaf in fp32_top40:
                    top40_agree += 1

    return {
        "total": total,
        "top1": 100 * top1_agree / max(total, 1),
        "top5": 100 * top5_agree / max(total, 1),
        "top40": 100 * top40_agree / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="STE-Train Tree Head on Model")
    parser.add_argument("--model", default="checkpoints/tinystories_fast.blmf")
    parser.add_argument("--tokenizer", default="checkpoints/tinystories_vocab4096.json")
    parser.add_argument("--data", default="data/tinystories_train.txt")
    parser.add_argument("--output", default="/tmp/ste_trained_tree.bin")
    parser.add_argument("--collect-steps", type=int, default=20000,
                        help="Number of steps to collect hidden states")
    parser.add_argument("--epochs", type=int, default=200, help="STE epochs per node")
    parser.add_argument("--lr", type=float, default=0.01, help="STE learning rate")
    parser.add_argument("--weighting", default=None,
                        choices=["none", "sqrt_inv", "log_inv"],
                        help="Frequency weighting scheme for loss")
    args = parser.parse_args()
    weighting = None if args.weighting == "none" else args.weighting

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Device: {device}")

    print(f"[*] Loading model from {args.model}...")
    model, config = load_model_from_blmf(args.model)
    model.to(device)
    print(f"    Model: V={config.vocab_size} D={config.hidden_dim} L={config.num_layers}")

    print(f"[*] Loading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer)

    # Step 1: Collect hidden states
    print()
    print("═" * 60)
    print("[1/3] Collecting hidden states from model...")
    print("─" * 60)
    t0 = time.time()
    H, T = collect_data(model, tokenizer, args.data, max_steps=args.collect_steps)
    print(f"    Collected in {time.time()-t0:.1f}s")
    print(f"    H shape: {H.shape}, T shape: {T.shape}")
    print(f"    Unique tokens: {T.unique().numel()}/{config.vocab_size}")

    # Step 2: Train tree head
    print()
    print("[2/3] Training tree head (STE)...")
    print("─" * 60)
    t0 = time.time()
    nodes, leaf_map, leaf_base, nw, D, packed_bytes, stats = build_tree_wrapper(
        model, H, T,
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        weighting=weighting,
    )
    tree_head = (nodes, leaf_map, leaf_base, nw, D)
    print(f"    Trained in {time.time()-t0:.1f}s")
    print(f"    Tree: {len(nodes)} nodes, {len([n for n in nodes if n is not None])} trained")

    # Save packed tree
    with open(args.output, "wb") as f:
        f.write(packed_bytes)
    print(f"    Saved to {args.output} ({len(packed_bytes)} bytes)")

    # Save epoch stats for entropy report analysis
    stats_path = args.output.replace(".bin", "_stats.json")
    with open(stats_path, "w") as f:
        import json
        json.dump(stats, f, indent=2)
    print(f"    Stats saved to {stats_path}")

    # Step 3: Evaluate on validation data
    print()
    print("[3/3] Evaluating tree head agreement...")
    print("─" * 60)
    with open("data/tinystories_val.txt", encoding="utf-8") as f:
        val_text = f.read()
    val_stories = [s.strip() for s in val_text.split("\n---\n") if s.strip()]

    results = evaluate_tree_head(model, tokenizer, val_stories, tree_head,
                                  device=device, max_stories=200)
    print(f"    Total predictions: {results['total']}")
    print(f"    Top-1 agreement:   {results['top1']:.2f}%")
    print(f"    Top-5 agreement:   {results['top5']:.2f}%")
    print(f"    Top-40 agreement:  {results['top40']:.2f}%")

    print()
    print("═" * 60)
    print("STE-Trained Tree Head Complete")
    print("═" * 60)


if __name__ == "__main__":
    main()
