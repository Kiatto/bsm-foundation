#!/usr/bin/env python3
"""
Train BPE tokenizer on TinyStories with vocab_size=4096.

Uses HuggingFace tokenizers (Rust backend) for fast training.
Saves in HuggingFace JSON format.
"""

import os
import sys
import time
import random
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blm.tokenizer import HFTokenizer


def main():
    train_path = "data/tinystories_train.txt"
    output_dir = Path("checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tinystories_vocab4096.json"

    with open(train_path, encoding="utf-8") as f:
        corpus = f.read()

    print(f"Training BPE tokenizer (vocab_size=4096) on {train_path}...")
    print(f"  Corpus size: {len(corpus):,} chars")
    print()

    t0 = time.time()
    tok = HFTokenizer(vocab_size=4096)
    tok.train(corpus, verbose=True)
    t_total = time.time() - t0

    tok.save(str(output_path))

    print()
    print("BPE Tokenizer trained")
    print("═" * 50)
    print(f"  Vocab size:      {tok.vocab_size:>6,}")
    print(f"  Training time:   {t_total:>6.1f}s")
    print("─" * 50)

    random.seed(42)
    with open("data/tinystories_val.txt", encoding="utf-8") as f:
        val_text = f.read()
    val_stories = [s.strip() for s in val_text.split("\n---\n") if s.strip()]

    sample = random.sample(val_stories, min(100, len(val_stories)))
    total_tokens = 0
    total_unks = 0
    total_chars = 0
    for story in sample:
        ids = tok.encode(story)
        total_tokens += len(ids)
        total_unks += ids.count(tok.unk_id)
        total_chars += len(story)

    unk_rate = 100 * total_unks / max(total_tokens, 1)
    comp_ratio = total_chars / max(total_tokens, 1)

    print(f"  Coverage test (100 random stories):")
    print(f"    UNK rate:           {unk_rate:>6.2f}%")
    print(f"    Compression ratio:  {comp_ratio:>6.1f}x")
    print("═" * 50)
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
