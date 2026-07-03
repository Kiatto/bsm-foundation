#!/usr/bin/env python3
"""
Qualitative text generation: compare FP32 vs Tree Head.

Samples tokens greedily and with top-40 sampling using both heads.
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
from blm.tokenizer import load_tokenizer as _load_tok
from blm.export import load_from_blmf
from scripts.bench_tree_accuracy import pack_vector
from scripts.build_tree_head import build_tree_from_head


def load_blmf_with_tree(path: str):
    """Load BLMF model and build tree head from model weights."""
    info = load_from_blmf(path)
    hdr = info["header"]
    sections = info["sections"]

    import numpy as np
    import torch

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

    # Build centroid tree from head weights
    head_w = model.head.weight.detach().cpu()
    nodes, num_nodes, nw, vs, leaf_base = build_tree_from_head(head_w)
    leaf_map = {i: i for i in range(vs)}
    tree_head = (nodes, leaf_map, leaf_base, nw, model.config.hidden_dim)

    return model, config, tree_head


def tree_predict_wrapper(tree_head, packed):
    """tree_head is (nodes, leaf_map, leaf_base, nw, D) tuple."""
    from scripts.bench_tree_accuracy import tree_predict as _tp
    nodes, leaf_map, leaf_base, nw, D = tree_head
    return _tp(nodes, leaf_map, leaf_base, nw, packed, D)


@torch.no_grad()
def generate_fp32(model, tokenizer, prompt_ids, max_new_tokens=100, 
                  temperature=1.0, top_k=None, device="cpu"):
    """Generate tokens using FP32 head."""
    ids = list(prompt_ids)
    D = model.config.hidden_dim
    
    for _ in range(max_new_tokens):
        input_t = torch.tensor([ids[-128:]], dtype=torch.long).to(device)
        
        hidden = model.embedding(input_t)
        state = torch.full((1, D), -1.0, device=device)
        for layer in model.layers:
            hidden, state = layer(hidden, state, single_step=False)
        
        logits = model.head(hidden[:, -1, :])  # [1, vocab]
        
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
        else:
            # Greedy
            next_id = logits.argmax(dim=-1).item()
            ids.append(next_id)
            continue
        
        if top_k is not None:
            top_probs, top_idx = probs.topk(top_k, dim=-1)
            probs = torch.zeros_like(probs)
            probs.scatter_(-1, top_idx, top_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        
        next_id = torch.multinomial(probs[0], 1).item()
        ids.append(next_id)
        
        if next_id == tokenizer.eos_id:
            break
    
    return ids


@torch.no_grad()
def generate_tree_greedy(model, tokenizer, prompt_ids, tree_head, 
                         max_new_tokens=100, device="cpu"):
    """Generate tokens using Tree Head (always picks most likely leaf)."""
    ids = list(prompt_ids)
    D = model.config.hidden_dim
    
    for _ in range(max_new_tokens):
        input_t = torch.tensor([ids[-128:]], dtype=torch.long).to(device)
        
        hidden = model.embedding(input_t)
        state = torch.full((1, D), -1.0, device=device)
        for layer in model.layers:
            hidden, state = layer(hidden, state, single_step=False)
        
        # Tree head prediction
        h = hidden[0, -1, :].cpu().numpy()
        packed = pack_vector(h)
        leaf = tree_predict_wrapper(tree_head, packed)
        
        next_id = leaf
        ids.append(next_id)
        
        if next_id == tokenizer.eos_id:
            break
    
    return ids


@torch.no_grad()
def generate_tree_topk(model, tokenizer, prompt_ids, tree_head, k=40,
                       max_new_tokens=100, device="cpu"):
    """Generate using Tree Head for sampling from top-k tree predictions.
    
    Since tree head picks a single leaf, we use FP32 head for the top-k
    distribution but restricted to tree-reachable leaves. For a pure-tree
    implementation, this would use the tree's own confidence scores.
    """
    # Hybrid: use FP32 logits but mask to tree leaves,
    # then sample from top-k among those
    ids = list(prompt_ids)
    D = model.config.hidden_dim
    
    for _ in range(max_new_tokens):
        input_t = torch.tensor([ids[-128:]], dtype=torch.long).to(device)
        
        hidden = model.embedding(input_t)
        state = torch.full((1, D), -1.0, device=device)
        for layer in model.layers:
            hidden, state = layer(hidden, state, single_step=False)
        
        logits = model.head(hidden[:, -1, :])  # [1, vocab]
        
        # Get tree leaf
        h = hidden[0, -1, :].cpu().numpy()
        packed = pack_vector(h)
        leaf = tree_predict_wrapper(tree_head, packed)
        
        next_id = leaf
        ids.append(next_id)
        
        if next_id == tokenizer.eos_id:
            break
    
    return ids


def main():
    parser = argparse.ArgumentParser(description="Generate samples")
    parser.add_argument("--model", required=True, help="BLMF model path")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer JSON path")
    parser.add_argument("--prompt", default="Once upon a time", help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--samples", type=int, default=3, help="Samples per method")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[*] Loading model from {args.model}...")
    model, config, tree_head = load_blmf_with_tree(args.model)
    model.to(device)
    model.eval()

    print(f"[*] Loading tokenizer from {args.tokenizer}...")
    tokenizer = _load_tok(args.tokenizer)

    # Encode prompt
    prompt_ids = tokenizer.encode(args.prompt)
    print(f"\nPrompt: \"{args.prompt}\"")
    print(f"Prompt tokens: {len(prompt_ids)} ({prompt_ids[:10]}...)")
    print()

    # Generate with FP32 (greedy)
    print("─" * 60)
    print("FP32 Greedy:")
    for i in range(args.samples):
        ids = generate_fp32(model, tokenizer, prompt_ids[:-1],
                           max_new_tokens=args.max_tokens,
                           temperature=0.0, device=device)
        text = tokenizer.decode(ids)
        print(f"  [{i+1}] {text}")
        print()

    # Generate with FP32 (top-40 sampling)
    print("─" * 60)
    print("FP32 Top-40 Sampling:")
    for i in range(args.samples):
        ids = generate_fp32(model, tokenizer, prompt_ids[:-1],
                           max_new_tokens=args.max_tokens,
                           temperature=0.8, top_k=40, device=device)
        text = tokenizer.decode(ids)
        print(f"  [{i+1}] {text}")
        print()

    if tree_head is not None:
        # Generate with Tree (greedy)
        print("─" * 60)
        print("Tree Greedy:")
        for i in range(args.samples):
            ids = generate_tree_greedy(model, tokenizer, prompt_ids[:-1],
                                       tree_head, max_new_tokens=args.max_tokens,
                                       device=device)
            text = tokenizer.decode(ids)
            print(f"  [{i+1}] {text}")
            print()
    else:
        print("[!] No tree head found in model file.")

    print("═" * 60)


if __name__ == "__main__":
    main()
