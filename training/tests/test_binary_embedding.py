"""Tests for BinaryEmbedding layer."""

import torch
import pytest

from blm.layers.binary_embedding import BinaryEmbedding
from blm.binary_ops import binarize_weight, binarize_activation, pack_bits, hamming_distance


class TestBinarizeFunctions:
    def test_binarize_weight_output_values(self):
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        y = binarize_weight(x)
        expected = torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0])
        assert torch.allclose(y, expected), f"Got {y}, expected {expected}"

    def test_binarize_activation_output_values(self):
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        y = binarize_activation(x)
        expected = torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0])
        assert torch.allclose(y, expected), f"Got {y}, expected {expected}"

    def test_binarize_weight_gradient_flows(self):
        x = torch.randn(10, requires_grad=True)
        y = binarize_weight(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None, "Gradient is None"
        assert torch.allclose(x.grad, torch.ones(10)), \
            f"STE gradient should be 1, got {x.grad}"

    def test_binarize_activation_gradient_flows(self):
        x = torch.randn(10, requires_grad=True)
        y = binarize_activation(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None, "Gradient is None"


class TestBinaryEmbedding:
    def test_output_shape(self):
        emb = BinaryEmbedding(vocab_size=256, hidden_dim=64)
        ids = torch.randint(0, 256, (2, 8))
        out = emb(ids)
        assert out.shape == (2, 8, 64), f"Shape mismatch: {out.shape}"

    def test_output_binary_values(self):
        emb = BinaryEmbedding(vocab_size=64, hidden_dim=64)
        ids = torch.tensor([[1, 2, 3]])
        out = emb(ids)
        unique_vals = out.unique()
        assert set(unique_vals.tolist()).issubset({-1.0, 1.0}), \
            f"Values not binary: {unique_vals.tolist()}"

    def test_gradient_flows_through_embedding(self):
        emb = BinaryEmbedding(vocab_size=64, hidden_dim=64)
        ids = torch.tensor([[1, 2, 3]])
        out = emb(ids)
        loss = out.sum()
        loss.backward()
        assert emb.weight.grad is not None, "Gradient is None"
        assert emb.weight.grad.shape == emb.weight.shape, \
            f"Gradient shape mismatch: {emb.weight.grad.shape}"

    def test_export_binary_shape(self):
        emb = BinaryEmbedding(vocab_size=256, hidden_dim=64)
        exported = emb.export_binary()
        assert exported.shape == (256, 8), \
            f"Export shape mismatch: {exported.shape}"
        assert exported.dtype == torch.uint8, \
            f"Export dtype: {exported.dtype}"

    def test_export_binary_values(self):
        emb = BinaryEmbedding(vocab_size=10, hidden_dim=64)
        exported = emb.export_binary()
        # Each byte should have bits 0-7
        assert (exported <= 0xFF).all(), "Byte values exceed 0xFF"
        assert (exported >= 0).all(), "Byte values negative"

    def test_different_tokens_different_embeddings(self):
        emb = BinaryEmbedding(vocab_size=64, hidden_dim=64)
        ids = torch.tensor([[1, 2]])
        out = emb(ids)
        # Token 1 and token 2 should (probably) have different embeddings
        diff = (out[0, 0] != out[0, 1]).sum()
        assert diff > 0, "Different tokens have identical embeddings"

    def test_stats(self):
        emb = BinaryEmbedding(vocab_size=64, hidden_dim=64)
        stats = emb.stats()
        assert stats["vocab_size"] == 64
        assert stats["hidden_dim"] == 64
        assert stats["total_params"] == 64 * 64
        assert 0 < stats["pos_ratio"] < 1


class TestPackBits:
    def test_pack_bits_basic(self):
        tensor = torch.tensor([[1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]])
        packed = pack_bits(tensor)
        # 1,1,0,0,1,1,0,0 = 0b00110011 = 0x33
        expected = torch.tensor([[0b00110011]], dtype=torch.uint8)
        assert torch.allclose(packed, expected), f"Got {packed}, expected {expected}"

    def test_pack_bits_dimension(self):
        tensor = torch.randn(5, 64)
        bits = (tensor > 0).float() * 2 - 1  # -> {-1, +1}
        packed = pack_bits(bits)
        assert packed.shape == (5, 8), f"Shape: {packed.shape}"


class TestHammingDistance:
    def test_identical_vectors(self):
        a = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
        dist = hamming_distance(a, a)
        assert dist.item() == 0.0, f"Identical vectors: {dist.item()}"

    def test_opposite_vectors(self):
        a = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
        b = torch.tensor([[-1.0, -1.0, 1.0, 1.0]])
        dist = hamming_distance(a, b)
        assert dist.item() == 4.0, f"Opposite vectors: {dist.item()}"

    def test_half_distance(self):
        a = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        b = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
        dist = hamming_distance(a, b)
        assert dist.item() == 2.0, f"Half distance: {dist.item()}"
