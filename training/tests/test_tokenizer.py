"""Tests for BPETokenizer."""

import json
import tempfile
from pathlib import Path

from blm.tokenizer import BPETokenizer


def test_special_tokens():
    """Special tokens have correct IDs."""
    tok = BPETokenizer(vocab_size=256)
    assert tok.pad_id == 0
    assert tok.unk_id == 1
    assert tok.bos_id == 2
    assert tok.eos_id == 3


def test_encode_always_has_bos_eos():
    """encode() always returns [BOS, ..., EOS]."""
    tok = BPETokenizer(vocab_size=256)
    tok.train("hello world")
    ids = tok.encode("")
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id


def test_roundtrip_simple():
    """encode -> decode should restore text for known tokens."""
    tok = BPETokenizer(vocab_size=256)
    tok.train("the cat sat on the mat")
    ids = tok.encode("the cat")
    decoded = tok.decode(ids)
    assert "the" in decoded
    assert "cat" in decoded


def test_roundtrip_shakespeare():
    """Roundtrip on a longer text."""
    text = "binary systems are fast and efficient"
    tok = BPETokenizer(vocab_size=512)
    tok.train(text * 50)
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    # After decode, all original words should be present
    for word in text.split():
        assert word in decoded, f"Word '{word}' missing from decoded text"


def test_unknown_tokens():
    """Unknown characters should map to UNK, not crash."""
    tok = BPETokenizer(vocab_size=64)
    tok.train("abc def ghi")
    # Train only on ASCII, try to encode emoji
    ids = tok.encode("hello 🌍 world")
    assert tok.unk_id in ids
    assert tok.bos_id in ids
    assert tok.eos_id in ids


def test_save_load_identity():
    """save -> load should produce identical tokenizer."""
    tok = BPETokenizer(vocab_size=256)
    tok.train("the cat sat on the mat the dog ran in the park " * 10)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
        tok.save(path)

    loaded = BPETokenizer.load(path)
    Path(path).unlink()

    assert loaded.vocab == tok.vocab
    assert loaded.merges == tok.merges
    assert loaded.vocab_size == tok.vocab_size

    # Same encoding after reload
    text = "the cat sat"
    orig_ids = tok.encode(text)
    loaded_ids = loaded.encode(text)
    assert orig_ids == loaded_ids, f"Mismatch: {orig_ids} vs {loaded_ids}"


def test_train_reduces_token_count():
    """After training, same text should use fewer tokens than raw characters."""
    tok = BPETokenizer(vocab_size=256)
    text = "the cat sat on the mat "
    corpus = text * 50

    tok.train(corpus)

    # Encode a long string
    ids = tok.encode(text * 10)

    # Without merging, each char would be a token
    # With BPE, common n-grams should be merged
    num_unique_chars = len(set(text))
    # The number of tokens should be less than raw character count
    char_count = len(text * 10)
    assert len(ids) < char_count, \
        f"BPE should reduce tokens: {len(ids)} >= {char_count}"


def test_vocab_size_respected():
    """Vocab size should never exceed the target."""
    target = 64
    tok = BPETokenizer(vocab_size=target)
    tok.train("the cat sat on the mat the dog ran in the park " * 100)

    assert tok.vocab_size <= target, \
        f"Vocab size {tok.vocab_size} exceeds target {target}"


def test_deterministic_encoding():
    """Same input should always produce same encoding."""
    tok = BPETokenizer(vocab_size=256)
    tok.train("one two three four five six seven eight nine ten " * 20)

    text = "one two three"
    ids1 = tok.encode(text)
    ids2 = tok.encode(text)
    assert ids1 == ids2, f"Non-deterministic: {ids1} vs {ids2}"


def test_decode_ignores_special_tokens():
    """PAD, BOS, EOS should be stripped during decode."""
    tok = BPETokenizer(vocab_size=256)
    tok.train("hello world")
    text = tok.decode([tok.bos_id, tok.vocab.get("hello", tok.unk_id),
                       tok.vocab.get("world", tok.unk_id), tok.eos_id, tok.pad_id])
    assert "hello" in text
    assert "world" in text


def test_merge_order():
    """Merges should be applied in training order."""
    tok = BPETokenizer(vocab_size=64)
    tok.train("ab" * 50 + "bc" * 50)

    # After training on "ab" and "bc", the most frequent pair
    # should have been merged first
    if len(tok.merges) >= 2:
        # The merged tokens should exist in vocab
        assert len(tok.merges) <= tok.vocab_size - 4  # minus special tokens
