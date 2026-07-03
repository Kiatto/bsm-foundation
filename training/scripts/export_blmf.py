#!/usr/bin/env python3
"""
Export a trained checkpoint + tokenizer to a single .blmf binary file.

Usage:
    python training/scripts/export_blmf.py \\
        --checkpoint checkpoints/checkpoint_final.pt \\
        --tokenizer tokenizer.json \\
        --output model.blmf
"""

import argparse
import json
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blm.model import BSMModel, BSMConfig
from blm.tokenizer import BPETokenizer
from blm.export import export_to_blmf


def main():
    parser = argparse.ArgumentParser(
        description="Export trained checkpoint + tokenizer to BLMF format"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer .json file")
    parser.add_argument("--config", default=None, help="Path to model config JSON (optional, overrides checkpoint config)")
    parser.add_argument("--output", default="model.blmf", help="Output .blmf file path")
    parser.add_argument("--mode", default=None, help="Quantization mode (ignored for now)")
    args = parser.parse_args()

    try:
        # Load checkpoint
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint.get("config")
        if isinstance(config, dict):
            config = BSMConfig(**config)
        elif not isinstance(config, BSMConfig):
            print(f"Error: unknown config type {type(config)}", file=sys.stderr)
            sys.exit(1)

        # Override config from JSON if provided
        if args.config and os.path.exists(args.config):
            with open(args.config) as f:
                config_data = json.load(f)
            config = BSMConfig(**config_data)
            print(f"Using config from {args.config}")

        print(f"Config: vocab={config.vocab_size}, dim={config.hidden_dim}, "
              f"layers={config.num_layers}")

        # Load tokenizer
        print(f"Loading tokenizer from {args.tokenizer}...")
        tokenizer = BPETokenizer.load(args.tokenizer)

        # Create model and restore weights
        model = BSMModel(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(model.summary())

        # Export to BLMF
        export_to_blmf(model, tokenizer, args.output)
        print(f"Done. Exported to {args.output}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
