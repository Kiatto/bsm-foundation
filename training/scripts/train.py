#!/usr/bin/env python3
"""
Unified BSM training script with Tree Head integration.

Reads YAML config, trains the BSM model, builds/trains the Tree Head
at warmup_steps, and handles Ctrl+C gracefully.

Usage:
    python training/scripts/train.py --config training/configs/tinystories_fast.yaml
"""

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blm.model import BSMModel, BSMConfig
from blm.tokenizer import HFTokenizer, load_tokenizer
from blm.data import TextDataset
from blm.trainer import Trainer, TrainerConfig
from blm.export import export_to_blmf


# ── Global state for signal handler ──────────────────────────────────
_should_exit = False
_trainer = None
_model = None
_tokenizer = None
_config = None
_tree_head_exported = False


def signal_handler(sig, frame):
    global _should_exit
    print(f"\n[!] Caught Ctrl+C, initiating graceful shutdown...")
    _should_exit = True
    if _trainer is not None:
        print("[!] Trainer will stop after current step...")


def load_texts(path: str) -> list[str]:
    """Load stories separated by --- delimiter."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    stories = [s.strip() for s in text.split("\n---\n") if s.strip()]
    if not stories:
        stories = [line.strip() for line in text.split("\n") if line.strip()]
    return stories


def build_tree(model):
    """Build centroid tree from model head weights.
    Returns (nodes, leaf_map, leaf_base, nw, D) tuple.
    """
    from scripts.build_tree_head import build_tree_from_head
    head_w = model.head.weight.detach().cpu()
    nodes, num_nodes, nw, vocab_size, leaf_base = build_tree_from_head(head_w)
    leaf_map = {i: i for i in range(vocab_size)}
    return nodes, leaf_map, leaf_base, nw, model.config.hidden_dim


def tree_predict_wrapper(tree_head, packed):
    from scripts.bench_tree_accuracy import tree_predict as _tp
    nodes, leaf_map, leaf_base, nw, D = tree_head
    return _tp(nodes, leaf_map, leaf_base, nw, packed, D)


def evaluate_tree_head(model, tokenizer, val_texts, tree_head, device="cpu"):
    """Evaluate tree head agreement vs FP32 argmax on validation data."""
    from scripts.bench_tree_accuracy import pack_vector

    model.eval()
    total = 0
    agree = 0
    top5_agree = 0

    with torch.no_grad():
        for text in val_texts[:200]:  # 200 stories
            ids = tokenizer.encode(text)
            if len(ids) < 10:
                continue
            ids_t = torch.tensor(ids[:129], dtype=torch.long).unsqueeze(0).to(device)

            # Get hidden states
            hidden = model.embedding(ids_t[:, :-1])  # [1, seq_len, dim]
            for layer in model.layers:
                hidden, _ = layer(hidden, torch.full((1, model.config.hidden_dim), -1.0, device=device), single_step=False)

            h = hidden[0]  # [seq_len, dim]

            # FP32 argmax
            logits = model.head(h)  # [seq_len, vocab]
            fp32_pred = logits.argmax(dim=-1)  # [seq_len]

            # Tree head predictions
            for pos in range(h.shape[0]):
                packed = pack_vector(h[pos].cpu().numpy())
                leaf = tree_predict_wrapper(tree_head, packed)
                tree_pred = leaf

                total += 1
                if tree_pred == fp32_pred[pos].item():
                    agree += 1

                top5 = logits[pos].topk(5).indices.tolist()
                if tree_pred in top5:
                    top5_agree += 1

    top1 = 100 * agree / max(total, 1)
    top5 = 100 * top5_agree / max(total, 1)
    return top1, top5, total


def main():
    global _trainer, _model, _tokenizer, _config, _tree_head_exported

    parser = argparse.ArgumentParser(description="Train BSM with Tree Head")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--export-only", default=None, help="Path to model .pt; export BLMF and exit")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    _config = cfg

    device = args.device if torch.cuda.is_available() else "cpu"
    if args.device != "cpu" and not torch.cuda.is_available():
        print("[!] CUDA not available, falling back to CPU")

    torch.manual_seed(cfg["train"]["seed"])

    # ── Model config ─────────────────────────────────────────────
    mc = cfg["model"]
    config = BSMConfig(
        vocab_size=mc["vocab_size"],
        hidden_dim=mc["hidden_dim"],
        num_layers=mc["num_layers"],
        seq_len=mc["seq_len"],
        window_size=mc.get("window_size", 8),
    )

    # ── Tokenizer ────────────────────────────────────────────────
    tok_path = cfg["data"]["tokenizer_path"]
    if os.path.exists(tok_path):
        print(f"[*] Loading tokenizer from {tok_path}")
        tokenizer = load_tokenizer(tok_path)
    else:
        print(f"[*] Training tokenizer from {cfg['data']['train_path']}...")
        tokenizer = HFTokenizer(vocab_size=config.vocab_size)
        with open(cfg["data"]["train_path"]) as f:
            corpus = f.read()
        tokenizer.train(corpus, verbose=True)
        tokenizer.save(tok_path)
    _tokenizer = tokenizer
    print(f"[*] Tokenizer: vocab={tokenizer.vocab_size}")

    # ── Data ─────────────────────────────────────────────────────
    print("[*] Loading training data...")
    train_texts = load_texts(cfg["data"]["train_path"])
    val_texts = load_texts(cfg["data"]["val_path"])
    print(f"    Train stories: {len(train_texts)}, Val stories: {len(val_texts)}")

    dataset = TextDataset(tokenizer, train_texts, seq_len=config.seq_len)
    val_dataset = TextDataset(tokenizer, val_texts[:500], seq_len=config.seq_len)
    print(f"    Train sequences: {len(dataset)}, tokens: {dataset.num_tokens:,}")
    print(f"    Val sequences: {len(val_dataset)}")

    # ── Model ────────────────────────────────────────────────────
    print("[*] Creating model...")
    model = BSMModel(config)
    model.to(device)
    _model = model
    print(model.summary())

    # ── Trainer ──────────────────────────────────────────────────
    tc = cfg["train"]
    trainer_config = TrainerConfig(
        batch_size=tc["batch_size"],
        learning_rate=tc["learning_rate"],
        max_steps=tc["max_steps"],
        warmup_steps=tc.get("warmup_steps", 200),
        weight_decay=tc.get("weight_decay", 0.01),
        output_dir=cfg["output"]["checkpoint_dir"],
        log_interval=cfg["output"].get("log_interval", 50),
        save_interval=cfg["output"].get("save_interval", 500),
        eval_interval=cfg["output"].get("eval_interval", 500),
    )

    trainer = Trainer(
        model=model,
        config=trainer_config,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        device=device,
    )
    _trainer = trainer

    if args.resume and os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)
        print(f"[*] Resumed from {args.resume}")

    # ── Training ─────────────────────────────────────────────────
    print(f"[*] Training for {tc['max_steps']} steps...")
    print(f"[*] Press Ctrl+C to save checkpoint and exit gracefully")
    print()

    # Wrap train with SIGINT handling
    try:
        summary = trainer.train()
    except KeyboardInterrupt:
        print(f"\n[*] Training interrupted at step {trainer.step}")
        trainer._save_checkpoint("interrupted")

    # ── Build Tree Head ─────────────────────────────────────────
    print()
    print("═" * 60)
    print("[TREE HEAD] Building tree head from trained model...")

    model.eval()
    tree_head = build_tree(model)
    print(f"[TREE HEAD] Tree built: {len(tree_head[0])} internal nodes")

    # Evaluate agreement
    print(f"[TREE HEAD] Evaluating validation agreement...")
    top1, top5, n = evaluate_tree_head(model, tokenizer, val_texts, tree_head, device)
    print(f"[TREE HEAD] Top-1 agreement: {top1:.1f}% | Top-5: {top5:.1f}% | n={n}")
    print("═" * 60)

    # ── Export ──────────────────────────────────────────────────
    output_dir = Path(cfg["output"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    export_path = cfg["output"]["export_path"]
    print(f"[*] Exporting BLMF to {export_path}...")
    export_to_blmf(model, tokenizer, export_path)
    print(f"[*] BLMF exported (with centroid tree head)")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
