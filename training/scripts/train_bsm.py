#!/usr/bin/env python3
"""
BSM v2.1 — Binary State Machine: train from scratch on TinyStories.

Usage:
    python training/scripts/train_bsm.py [--dim 64] [--steps 3000]

No GPU required. No CUDA. Minimal dependencies.
"""

import argparse
import math
import random
import signal
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from bsm import BinaryStateMachine, BSMTrainer, TrainerConfig
from bsm.data import TextDataset
from blm.tokenizer import load_tokenizer
def load_texts(path: str) -> list[str]:
    """Load a text file, split by <|endoftext|> or double newline."""
    with open(path) as f:
        text = f.read()
    stories = []
    for s in text.split("<|endoftext|>"):
        s = s.strip()
        if s:
            stories.append(s)
    if not stories:
        stories = [s.strip() for s in text.split("\n\n") if s.strip()]
    return stories or [text]


def main():
    parser = argparse.ArgumentParser(description="Train Binary State Machine")
    parser.add_argument("--dim", type=int, default=64, help="State dimension")
    parser.add_argument("--steps", type=int, default=3000, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("=" * 60)
    print(f"BSM v2.1 — Binary State Machine")
    print(f"  Dim: {args.dim}  Steps: {args.steps}  LR: {args.lr}")
    print("=" * 60)

    # ── Data ──
    print("\n[*] Loading tokenizer...")
    tok = load_tokenizer("checkpoints/tinystories_vocab4096.json")
    print(f"    Vocab: {tok.vocab_size}")

    print("[*] Loading data...")
    train_texts = load_texts("data/tinystories_train.txt")
    val_texts = load_texts("data/tinystories_val.txt")
    train_dataset = TextDataset(tok, train_texts[:10000], seq_len=args.seq_len)
    val_dataset = TextDataset(tok, val_texts[:500], seq_len=args.seq_len)
    print(f"    Train: {len(train_dataset)} sequences")
    print(f"    Val:   {len(val_dataset)} sequences")

    # ── Model ──
    print("[*] Creating model...")
    model = BinaryStateMachine(
        vocab_size=tok.vocab_size,
        hidden_dim=args.dim,
    )
    print(model.summary())

    # ── Trainer ──
    log_interval = max(1, args.steps // 100)
    eval_interval = max(1, args.steps // 20)
    save_interval = max(1, args.steps // 10)

    trainer_config = TrainerConfig(
        batch_size=args.batch,
        learning_rate=args.lr,
        warmup_steps=min(500, args.steps // 10),
        max_steps=args.steps,
        log_interval=log_interval,
        eval_interval=eval_interval,
        save_interval=save_interval,
        output_dir=f"checkpoints/bsm_d{args.dim}",
    )

    trainer = BSMTrainer(model, trainer_config, train_dataset, val_dataset)
    summary = trainer.train()

    # ── Final test ──
    print("\n[*] Final validation...")
    model.eval()
    total_acc = 0
    total_bits = 0
    n_batches = 0

    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    with torch.no_grad():
        for input_ids, target_ids in val_loader:
            logits, _ = model(input_ids)
            acc = model.accuracy(logits, target_ids)
            bit_acc = model.bit_accuracy(logits, target_ids)
            total_acc += acc.item()
            total_bits += bit_acc.item()
            n_batches += 1
            if n_batches >= 10:
                break

    print(f"\n  Validation token accuracy: {total_acc/n_batches*100:.1f}%")
    print(f"  Validation bit accuracy:  {total_bits/n_batches*100:.1f}%")
    print(f"  Random baseline token:    {100/4096:.2f}%")
    print(f"  Random baseline bit:      50.0%")
    print(f"\n  Expected token acc if bits are independent: ")
    print(f"    (0.5 + bit_acc/2)^{model.log_vocab}")

    per_bit_independent = ((0.5 + (total_bits/n_batches)/2) ** model.log_vocab) * 100
    print(f"    ≈ {per_bit_independent:.1f}%")

    print("\nDone!")


if __name__ == "__main__":
    main()
