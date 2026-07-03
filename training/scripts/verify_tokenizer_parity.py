#!/usr/bin/env python3
"""
Verify tokenizer parity between Python and Go.

Trains a tokenizer, saves it, tokenizes test phrases,
and writes reference output for Go parity test.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blm.tokenizer import BPETokenizer


def main():
    corpus_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "testdata", "tiny_corpus.txt"
    )
    with open(corpus_path) as f:
        corpus = f.read()

    # Train tokenizer
    tok = BPETokenizer(vocab_size=512)
    tok.train(corpus, verbose=True)
    print(f"Trained: {tok.vocab_size} tokens, {len(tok.merges)} merges")

    # Save vocabulary
    out_dir = "/tmp"
    vocab_path = os.path.join(out_dir, "test_vocab.json")
    tok.save(vocab_path)
    print(f"Saved vocab to {vocab_path}")

    # Tokenize test phrases
    test_phrases = [
        "the cat sat",
        "binary systems are fast",
        "simple model",
        "the",
        "a dog ran in the park",
        "stars shine",
        "hello world unknown",
        "the neural network processes",
        "each layer transforms",
        "training continues until convergence",
    ]

    results = []
    for phrase in test_phrases:
        ids = tok.encode(phrase)
        decoded = tok.decode(ids)
        results.append({
            "text": phrase,
            "ids": ids,
            "decoded": decoded,
        })
        print(f"  '{phrase}' -> {ids}")

    # Save reference
    ref_path = os.path.join(out_dir, "tokenizer_reference.json")
    with open(ref_path, "w") as f:
        json.dump({
            "vocab_path": vocab_path,
            "vocab": tok.vocab,
            "results": results,
        }, f, indent=2)
    print(f"Saved reference to {ref_path}")
    print("Done. Run Go parity test to verify.")


if __name__ == "__main__":
    main()
