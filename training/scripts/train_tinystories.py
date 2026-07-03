#!/usr/bin/env python3
"""
Train BSM model on TinyStories dataset.

Usage:
    python training/scripts/train_tinystories.py --data corpus.txt --config configs/bsm-2048.json
    python -m training.scripts.train_tinystories --data corpus.txt --config configs/bsm-2048.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blm.model import BSMModel, BSMConfig
from blm.tokenizer import BPETokenizer
from blm.data import TextDataset
from blm.trainer import Trainer, TrainerConfig
from blm.export import export_to_blmf


def main():
    parser = argparse.ArgumentParser(
        description="Train a BSM model on TinyStories text corpus"
    )
    parser.add_argument("--data", required=True, help="Path to corpus .txt file")
    parser.add_argument("--config", required=True, help="Path to model config .json")
    parser.add_argument("--output", default="checkpoints/model.blmf", help="Output BLMF model path")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", default=None, help="Path to checkpoint .pt to resume from")
    args = parser.parse_args()

    try:
        torch.manual_seed(args.seed)

        # Load config
        with open(args.config) as f:
            config_data = json.load(f)
        config = BSMConfig(**config_data)
        print(f"Config: vocab={config.vocab_size}, dim={config.hidden_dim}, "
              f"layers={config.num_layers}, seq_len={config.seq_len}")

        # Tokenizer — load from alongside config, or train from corpus
        config_dir = os.path.dirname(os.path.abspath(args.config))
        tokenizer_path = os.path.join(config_dir, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            print(f"Loading tokenizer from {tokenizer_path}")
            tokenizer = BPETokenizer.load(tokenizer_path)
        else:
            print("Training tokenizer...")
            tokenizer = BPETokenizer(vocab_size=config.vocab_size)
            with open(args.data) as f:
                corpus = f.read()
            tokenizer.train(corpus, verbose=True)
            tokenizer.save(tokenizer_path)
            print(f"Tokenizer saved to {tokenizer_path}")

        # Load corpus
        print(f"Loading data from {args.data}...")
        texts = [line.strip() for line in open(args.data) if line.strip()]
        print(f"Loaded {len(texts)} lines")

        # Create dataset
        dataset = TextDataset(tokenizer, texts, seq_len=config.seq_len)
        print(f"Dataset: {len(dataset)} sequences, {dataset.num_tokens} tokens")

        # Create model
        model = BSMModel(config)
        print(model.summary())

        # Trainer config: convert epochs to steps
        steps_per_epoch = max(len(dataset) // args.batch_size, 1)
        max_steps = args.epochs * steps_per_epoch
        trainer_config = TrainerConfig(
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_steps=max_steps,
            output_dir=args.output,
        )

        # Trainer
        trainer = Trainer(
            model=model,
            config=trainer_config,
            train_dataset=dataset,
            device=args.device,
        )

        # Resume from checkpoint if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)

        # Train
        print(f"Training for {args.epochs} epoch(s) ({max_steps} steps)...")
        summary = trainer.train()

        # Export model to BLMF and save tokenizer alongside
        model.eval()
        export_path = Path(args.output)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer_out = export_path.with_name(export_path.stem + "_tokenizer.json")
        export_to_blmf(model, tokenizer, str(export_path))
        tokenizer.save(str(tokenizer_out))
        print(f"Tokenizer saved to {tokenizer_out}")
        print(f"Training complete. Final loss: {summary['final_loss']:.4f}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
