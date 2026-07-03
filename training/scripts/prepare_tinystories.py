#!/usr/bin/env python3
"""Download TinyStories, split train/val, save as text files."""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = Path("data")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets")
        sys.exit(1)

    print("Downloading TinyStories (50K stories)...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    stories = [ds[i]["text"] for i in range(50000)]

    train_stories = stories[:45000]
    val_stories = stories[45000:50000]

    train_path = DATA_DIR / "tinystories_train.txt"
    val_path = DATA_DIR / "tinystories_val.txt"

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_stories:
            f.write(s)
            f.write("\n---\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_stories:
            f.write(s)
            f.write("\n---\n")

    train_chars = sum(len(s) for s in train_stories)
    val_chars = sum(len(s) for s in val_stories)
    unique_chars = len(set("".join(stories)))
    avg_len = train_chars // len(train_stories)
    max_len = max(len(s) for s in stories)
    train_mb = os.path.getsize(train_path) / 1e6
    val_mb = os.path.getsize(val_path) / 1e6

    print()
    print("TinyStories Dataset")
    print("═" * 45)
    print(f"Train stories:     {len(train_stories):>6,}")
    print(f"Val   stories:     {len(val_stories):>6,}")
    print("─" * 45)
    print(f"Train tokens (est): {train_chars // 5:>9,}")
    print(f"Val   tokens (est): {val_chars // 5:>9,}")
    print(f"Unique chars:       {unique_chars:>9}")
    print(f"Avg story length:   {avg_len:>9} chars")
    print(f"Max story length:   {max_len:>9,} chars")
    print("─" * 45)
    print(f"Saved to:")
    print(f"  {train_path}  ({train_mb:.1f} MB)")
    print(f"  {val_path}    ({val_mb:.1f} MB)")
    print("═" * 45)

    print()
    for i in range(3):
        preview = stories[i][:200].replace("\n", " ")
        print(f"─── Story {i+1} (preview) ───")
        print(preview)
        print()


if __name__ == "__main__":
    main()
