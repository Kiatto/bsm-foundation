#!/usr/bin/env python3
"""
Verify embedding parity between Python and Go.

Creates a BinaryEmbedding, exports packed bytes,
and saves reference for Go parity test.
"""

import json
import os
import sys
import struct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from blm.layers.binary_embedding import BinaryEmbedding
from blm.binary_ops import binarize_weight, pack_bits


def main():
    # Deterministic seed
    torch.manual_seed(42)

    # Create embedding
    vocab_size = 64
    hidden_dim = 128
    emb = BinaryEmbedding(vocab_size, hidden_dim)
    stats = emb.stats()
    print(f"Embedding: vocab={vocab_size}, dim={hidden_dim}")
    print(f"  pos_ratio: {stats['pos_ratio']:.4f}")

    # Export packed bytes
    packed = emb.export_binary()  # [vocab_size, hidden_dim/8]
    raw_bytes = packed.numpy().tobytes()

    # Save to file
    out_dir = "/tmp"
    bin_path = os.path.join(out_dir, "test_embedding.bin")
    with open(bin_path, "wb") as f:
        # Header: vocab_size (int), hidden_dim (int), then raw bytes
        f.write(struct.pack("<II", vocab_size, hidden_dim))
        f.write(raw_bytes)
    print(f"Saved binary to {bin_path} ({len(raw_bytes)} bytes)")

    # Tokenize test phrases
    from blm.tokenizer import BPETokenizer
    corpus_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "testdata", "tiny_corpus.txt"
    )
    with open(corpus_path) as f:
        corpus = f.read()

    tok = BPETokenizer(vocab_size=vocab_size)
    tok.train(corpus)

    # Get embeddings for some tokens and verify binary property
    binary_weights = binarize_weight(emb.weight)
    test_tokens = ["the", "cat", "sat", "binary", "system"]

    ref_data = {
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "bin_path": bin_path,
        "tokens": {},
    }

    for token_name in test_tokens:
        if token_name not in tok.vocab:
            print(f"  Token '{token_name}' not in vocab, skipping")
            continue
        tid = tok.vocab[token_name]
        vec = binary_weights[tid].tolist()
        ref_data["tokens"][token_name] = {
            "id": tid,
            "values": vec,
        }
        print(f"  Token '{token_name}' (id={tid}): first 10 values = {vec[:10]}")

    ref_path = os.path.join(out_dir, "embedding_reference.json")
    with open(ref_path, "w") as f:
        json.dump(ref_data, f, indent=2)
    print(f"Saved reference to {ref_path}")
    print("Done.")


if __name__ == "__main__":
    main()
