"""Tests for BSM layers: BinaryLinear, BinaryStateUpdate, BinaryMixer, BSMModel."""

import torch
import pytest
from blm.layers.binary_linear import BinaryLinear
from blm.layers.binary_state import BinaryStateUpdate, majority3
from blm.layers.binary_mixer import BinaryMixer
from blm.model import BSMModel, BSMConfig, BSMLayer
from blm.binary_ops import binarize_activation, binarize_weight


# =============================================================================
# BinaryLinear tests
# =============================================================================

class TestBinaryLinear:
    def test_output_range(self):
        """Output should be in [-in_features, +in_features]."""
        layer = BinaryLinear(64, 128)
        x = torch.ones(2, 64)  # all +1
        y = layer(x)
        assert y.shape == (2, 128)
        assert y.abs().max() <= 64, f"Output exceeds range: {y.abs().max()}"

    def test_gradient_flows(self):
        """Gradient should flow through BinaryLinear via STE."""
        layer = BinaryLinear(64, 64)
        x = torch.randn(2, 64)
        x_bin = binarize_activation(x)
        y = layer(x_bin)
        loss = y.sum()
        loss.backward()
        assert layer.weight.grad is not None, "Gradient is None"
        assert not torch.isnan(layer.weight.grad).any(), "NaN gradient"

    def test_identical_vectors_give_maximum(self):
        """Identical vectors should give maximum output."""
        layer = BinaryLinear(64, 64)
        with torch.no_grad():
            layer.weight.fill_(1.0)
        x = torch.ones(1, 64)
        y = layer(x)
        assert torch.allclose(y, torch.full((1, 64), 64.0)), \
            f"Expected all 64.0, got {y[0, :3].tolist()}..."

    def test_opposite_vectors_give_minimum(self):
        """Opposite vectors should give minimum output."""
        layer = BinaryLinear(64, 64)
        with torch.no_grad():
            layer.weight.fill_(-1.0)
        x = torch.ones(1, 64)
        y = layer(x)
        assert torch.allclose(y, torch.full((1, 64), -64.0)), \
            f"Expected all -64.0, got {y[0, :3].tolist()}..."

    def test_export_binary_shape(self):
        """export_binary should produce correct packed shape."""
        layer = BinaryLinear(128, 64)
        packed = layer.export_binary()
        assert packed.shape == (64, 16), f"Shape: {packed.shape}"
        assert packed.dtype == torch.uint8

    def test_half_match(self):
        """Half matching bits should give dot = 0."""
        layer = BinaryLinear(64, 64)
        with torch.no_grad():
            layer.weight[:, :32] = 1.0
            layer.weight[:, 32:] = -1.0
        x = torch.ones(1, 64)
        y = layer(x)
        assert torch.allclose(y, torch.zeros(1, 64), atol=1e-5), \
            f"Expected all 0, got {y[0, :3].tolist()}..."


# =============================================================================
# BinaryStateUpdate tests
# =============================================================================

class TestBinaryStateUpdate:
    def test_output_shape(self):
        """Output should have same shape as input state."""
        bsu = BinaryStateUpdate(hidden_dim=64)
        state = bsu.init_state(batch_size=4)
        x = torch.sign(torch.randn(4, 64))
        out = bsu(state, x)
        assert out.shape == (4, 64)

    def test_output_binary(self):
        """Output should be in {-1, +1}."""
        bsu = BinaryStateUpdate(hidden_dim=64)
        state = bsu.init_state(batch_size=2)
        x = torch.sign(torch.randn(2, 64))
        out = bsu(state, x)
        vals = out.unique()
        assert set(vals.tolist()).issubset({-1.0, 1.0}), f"Values: {vals}"

    def test_gradient_flows(self):
        """Gradient should flow through state update."""
        bsu = BinaryStateUpdate(hidden_dim=64)
        state = bsu.init_state(batch_size=2)
        x = torch.randn(2, 64)
        x_bin = torch.sign(x)
        out = bsu(state, x_bin)
        loss = out.sum()
        loss.backward()
        assert bsu.W_forget.weight.grad is not None
        assert bsu.W_input.weight.grad is not None

    def test_multiple_steps(self):
        """Multiple state updates should work sequentially."""
        bsu = BinaryStateUpdate(hidden_dim=64)
        state = bsu.init_state(batch_size=2)
        x = torch.sign(torch.randn(2, 64))

        for _ in range(10):
            state = bsu(state, x)
            assert state.shape == (2, 64)
            assert set(state.unique().tolist()).issubset({-1.0, 1.0})


# =============================================================================
# majority3 tests
# =============================================================================

class TestMajority3:
    def test_two_out_of_three(self):
        """Majority with 2 True inputs should be True."""
        a = torch.ones(4, 64)
        b = torch.ones(4, 64)
        c = -torch.ones(4, 64)
        out = majority3(a, b, c)
        assert (out == 1.0).all(), "2/3 majority should be all +1"

    def test_one_out_of_three(self):
        """Majority with 1 True input should be False."""
        a = torch.ones(4, 64)
        b = -torch.ones(4, 64)
        c = -torch.ones(4, 64)
        out = majority3(a, b, c)
        assert (out == -1.0).all(), "1/3 majority should be all -1"

    def test_three_same(self):
        """All same should preserve value."""
        a = torch.ones(4, 64)
        out = majority3(a, a, a)
        assert (out == 1.0).all()
        out2 = majority3(-a, -a, -a)
        assert (out2 == -1.0).all()


# =============================================================================
# BinaryMixer tests
# =============================================================================

class TestBinaryMixer:
    def test_output_shape(self):
        """Output should be [B, T, D]."""
        mixer = BinaryMixer(hidden_dim=64, window_size=4)
        x = torch.sign(torch.randn(2, 8, 64))
        out = mixer(x)
        assert out.shape == (2, 8, 64)

    def test_output_binary(self):
        """Output should be in {-1, +1}."""
        mixer = BinaryMixer(hidden_dim=64, window_size=4)
        x = torch.sign(torch.randn(2, 8, 64))
        out = mixer(x)
        vals = out.unique()
        assert set(vals.tolist()).issubset({-1.0, 1.0})

    def test_window_size_effect(self):
        """Larger window should change output."""
        mixer1 = BinaryMixer(hidden_dim=128, window_size=2)
        mixer2 = BinaryMixer(hidden_dim=128, window_size=8)
        x = torch.sign(torch.randn(1, 16, 128))
        out1 = mixer1(x)
        out2 = mixer2(x)
        # Different windows should (usually) give different outputs
        diff = (out1 != out2).sum().item()
        assert diff > 0, "Different windows should differ"


# =============================================================================
# BSMLayer tests
# =============================================================================

class TestBSMLayer:
    def test_forward_shape(self):
        """Forward should produce correct shapes."""
        config = BSMConfig(hidden_dim=64)
        layer = BSMLayer(config)
        B, T, D = 2, 8, 64
        x = torch.sign(torch.randn(B, T, D))
        state = torch.full((B, D), -1.0)
        out, new_state = layer(x, state)
        assert out.shape == (B, T, D)
        assert new_state.shape == (B, D)

    def test_single_step_shape(self):
        """Single step should produce [B, D]."""
        config = BSMConfig(hidden_dim=64)
        layer = BSMLayer(config)
        B, D = 2, 64
        x = torch.sign(torch.randn(B, D))
        state = torch.full((B, D), -1.0)
        out, new_state = layer(x, state, single_step=True)
        assert out.shape == (B, D)
        assert new_state.shape == (B, D)

    def test_single_step_matches_sequence(self):
        """Step-by-step should match full sequence forward."""
        torch.manual_seed(42)
        config = BSMConfig(hidden_dim=64)
        layer = BSMLayer(config)
        B, T, D = 2, 4, 64
        x = torch.sign(torch.randn(B, T, D))
        state = torch.full((B, D), -1.0)

        # Full sequence forward
        full_out, full_state = layer(x, state)

        # Step-by-step
        step_states = []
        cs = state.clone()
        for t in range(T):
            xt = x[:, t, :]
            out_t, cs = layer(xt, cs, single_step=True)
            step_states.append(out_t)
        step_out = torch.stack(step_states, dim=1)

        assert torch.allclose(full_out, step_out), "Step-by-step != full forward"
        assert torch.allclose(full_state, cs), "Final states differ"


# =============================================================================
# BSMModel tests
# =============================================================================

class TestBSMModel:
    @pytest.fixture
    def small_config(self):
        return BSMConfig(vocab_size=64, hidden_dim=64, num_layers=2,
                         window_size=4, seq_len=16)

    @pytest.fixture
    def model(self, small_config):
        return BSMModel(small_config)

    def test_forward_shape(self, model):
        """Forward should produce [B, T, vocab_size]."""
        B, T = 2, 8
        ids = torch.randint(0, 64, (B, T))
        logits, states = model(ids)
        assert logits.shape == (B, T, 64), f"Shape: {logits.shape}"
        assert len(states) == 2  # num_layers

    def test_states_shape(self, model):
        """States should be list of [B, D]."""
        B, T = 2, 8
        ids = torch.randint(0, 64, (B, T))
        _, states = model(ids)
        for s in states:
            assert s.shape == (B, 64)

    def test_step_matches_forward(self, model):
        """step() should match forward() token by token."""
        B, T = 2, 4
        ids = torch.randint(0, 64, (B, T))

        # Full forward
        logits_full, states_full = model(ids)

        # Step by step
        states = model.init_states(B)
        for t in range(T):
            logits_t, states = model.step(ids[:, t], states)

        # Compare last step logits
        assert torch.allclose(logits_t, logits_full[:, -1, :], atol=1e-6), \
            "step() logits != forward() last logits"

    def test_gradient_end_to_end(self, model):
        """Gradient should flow from loss to all parameters."""
        B, T = 2, 8
        ids = torch.randint(0, 64, (B, T))
        logits, _ = model(ids)
        loss = logits.mean()
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No parameters received gradients"

    def test_num_parameters(self, model):
        """Parameter count should be reasonable."""
        p = model.num_parameters()
        assert p["total_params"] > 0
        assert p["binary_params"] > p["float_params"], \
            "Should have more binary than float params"
        assert p["total_size_bytes"] > 0

    def test_tiny_model_generates(self, model):
        """Tiny model should generate without errors."""
        model.eval()
        ids = torch.randint(0, 64, (1, 4))
        output = model.generate(ids, max_new_tokens=5, temperature=0.8, top_k=5)
        assert output.shape == (1, 9), f"Shape: {output.shape}"

    def test_generation_deterministic_with_seed(self, model):
        """Same seed should give same generation."""
        model.eval()
        ids = torch.randint(0, 64, (1, 4))

        torch.manual_seed(42)
        out1 = model.generate(ids.clone(), max_new_tokens=5, temperature=0.0)

        torch.manual_seed(42)
        out2 = model.generate(ids.clone(), max_new_tokens=5, temperature=0.0)

        assert torch.allclose(out1, out2), "Non-deterministic generation"

    def test_model_summary(self, model):
        """Summary should be a formatted string."""
        s = model.summary()
        assert isinstance(s, str)
        assert "BSM" in s
        assert "64" in s
