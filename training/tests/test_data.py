"""Tests for TextDataset and StreamingTextDataset."""

import torch
from blm.data import TextDataset, StreamingTextDataset
from blm.tokenizer import BPETokenizer


def test_dataset_returns_correct_shapes():
    tok = BPETokenizer(vocab_size=64)
    tok.train("hello world this is a test corpus for bpe tokenizer training")

    ds = TextDataset(tok, ["hello world this is a test"], seq_len=4)
    assert len(ds) > 0, "Dataset should have at least one sequence"

    input_ids, target_ids = ds[0]
    assert input_ids.shape == (4,), f"Expected (4,), got {input_ids.shape}"
    assert target_ids.shape == (4,), f"Expected (4,), got {target_ids.shape}"
    assert torch.equal(target_ids[:-1], input_ids[1:]), \
        "target_ids[:-1] should equal input_ids[1:]"


def test_dataset_stride():
    tok = BPETokenizer(vocab_size=64)
    tok.train("a b c d e f g h i j k l m n o p")

    ds_no_stride = TextDataset(tok, ["a b c d e f g h i j k l m n o p"], seq_len=4)
    ds_stride = TextDataset(tok, ["a b c d e f g h i j k l m n o p"], seq_len=4, stride=2)

    assert len(ds_stride) > len(ds_no_stride), \
        "Smaller stride should produce more sequences"


def test_dataset_empty_corpus():
    tok = BPETokenizer(vocab_size=64)
    tok.train("hello world")
    ds = TextDataset(tok, [""], seq_len=128)
    assert len(ds) == 0, "Dataset should be empty for too-short corpus"


def test_dataset_multiple_texts():
    tok = BPETokenizer(vocab_size=64)
    tok.train("one two three four five six seven eight nine ten")
    texts = ["one two three", "four five six", "seven eight nine ten"]
    ds = TextDataset(tok, texts, seq_len=4)
    assert len(ds) > 0, "Dataset should have sequences"
    input_ids, target_ids = ds[0]
    assert input_ids.shape == (4,)
    assert target_ids.shape == (4,)


def test_dataset_causal_structure():
    tok = BPETokenizer(vocab_size=64)
    tok.train("a sequence of words for testing causal lm structure")

    ds = TextDataset(tok, ["a sequence of words"], seq_len=4)
    for i in range(len(ds)):
        input_ids, target_ids = ds[i]
        assert torch.equal(target_ids[:-1], input_ids[1:]), \
            f"Sequence {i}: wrong causal structure"


def test_streaming_dataset():
    import tempfile, os
    tok = BPETokenizer(vocab_size=64)
    tok.train("test corpus for streaming dataset validation")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("hello world this is a streaming test")
        f_path = f.name

    try:
        ds = StreamingTextDataset(tok, [f_path], seq_len=4)
        count = 0
        for input_ids, target_ids in ds:
            assert input_ids.shape == (4,)
            assert target_ids.shape == (4,)
            count += 1
        assert count > 0, "Streaming dataset should yield sequences"
    finally:
        os.unlink(f_path)


def test_token_count():
    tok = BPETokenizer(vocab_size=64)
    tok.train("token counting test for verification of dataset properties")
    ds = TextDataset(tok, ["hello world"], seq_len=2)
    # encode adds BOS/EOS so we always have at least 2 tokens
    assert ds.num_tokens >= 2, "Should have at least BOS + EOS tokens"
