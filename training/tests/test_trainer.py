"""Tests for Trainer."""

import tempfile
from pathlib import Path

import torch
import pytest

from blm.model import BSMModel, BSMConfig
from blm.tokenizer import BPETokenizer
from blm.data import TextDataset
from blm.trainer import Trainer, TrainerConfig


def make_tiny_model():
    cfg = BSMConfig(
        vocab_size=64,
        hidden_dim=64,
        num_layers=1,
        window_size=2,
        seq_len=8,
    )
    return BSMModel(cfg)


def make_tiny_data():
    tok = BPETokenizer(vocab_size=64)
    tok.train("the cat sat on the mat and the dog ran in the park")
    corpus = ["the cat sat on the mat and the dog ran in the park hello world"]
    ds = TextDataset(tok, corpus, seq_len=8)
    return ds, tok


class TestTrainerConfig:
    def test_default_config(self):
        cfg = TrainerConfig()
        assert cfg.batch_size == 32
        assert cfg.learning_rate == 3e-4
        assert cfg.max_steps == 10000

    def test_custom_config(self):
        cfg = TrainerConfig(batch_size=8, max_steps=50)
        assert cfg.batch_size == 8
        assert cfg.max_steps == 50


class TestTrainer:
    def test_trainer_initializes(self):
        model = make_tiny_model()
        cfg = TrainerConfig(max_steps=1)
        ds, _ = make_tiny_data()
        trainer = Trainer(model, cfg, ds)
        assert trainer.step == 0
        assert trainer.best_loss == float('inf')

    def test_train_one_step(self):
        model = make_tiny_model()
        cfg = TrainerConfig(max_steps=1, batch_size=2, log_interval=1)
        ds, _ = make_tiny_data()
        trainer = Trainer(model, cfg, ds)
        result = trainer.train()
        assert result["steps"] >= 1
        assert len(result["train_losses"]) >= 1
        assert result["final_loss"] > 0, "Loss should be positive"

    def test_train_few_steps(self):
        model = make_tiny_model()
        cfg = TrainerConfig(max_steps=5, batch_size=2, log_interval=5, save_interval=10)
        ds, _ = make_tiny_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            trainer = Trainer(model, cfg, ds)
            result = trainer.train()
            assert result["steps"] == 5
            # Check that checkpoint directory was created
            assert Path(tmpdir).exists()

    def test_loss_decreases(self):
        model = make_tiny_model()
        cfg = TrainerConfig(max_steps=10, batch_size=2, log_interval=10)
        ds, _ = make_tiny_data()
        trainer = Trainer(model, cfg, ds)
        result = trainer.train()
        losses = result["train_losses"]
        if len(losses) >= 2:
            # Loss should generally decrease (not strictly, but should trend down)
            assert losses[-1] < losses[0] * 1.2, \
                f"Loss should trend downward: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_eval_during_training(self):
        model = make_tiny_model()
        cfg = TrainerConfig(max_steps=5, batch_size=2, eval_interval=3)
        ds, _ = make_tiny_data()
        trainer = Trainer(model, cfg, ds, eval_dataset=ds)
        result = trainer.train()
        # Eval at step 3
        assert len(result["eval_losses"]) >= 1, "Should have at least one eval"

    def test_checkpoint_save_and_load(self):
        model = make_tiny_model()
        cfg = TrainerConfig(max_steps=5, batch_size=2, save_interval=5, log_interval=5)
        ds, _ = make_tiny_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            trainer = Trainer(model, cfg, ds)
            trainer.train()

            # Check that checkpoint files exist
            checkpoint_dir = Path(tmpdir)
            files = list(checkpoint_dir.glob("*.pt"))
            assert len(files) >= 1, f"No checkpoint files found in {tmpdir}"

            # Load a model into a new trainer
            model2 = make_tiny_model()
            trainer2 = Trainer(model2, cfg, ds)
            trainer2.load_checkpoint(checkpoint_dir / "checkpoint_final.pt")
            assert trainer2.step == 5

    def test_lr_schedule(self):
        model = make_tiny_model()
        cfg = TrainerConfig(max_steps=100, warmup_steps=20, batch_size=2, log_interval=100)
        ds, _ = make_tiny_data()
        trainer = Trainer(model, cfg, ds)

        # Before warmup: lr should be proportional
        trainer.step = 10
        lr_at_10 = trainer._get_lr()
        assert lr_at_10 < cfg.learning_rate, f"Warmup LR {lr_at_10} should be < {cfg.learning_rate}"
        assert lr_at_10 > 0, "LR should be positive"

        # At full warmup: should be at max
        trainer.step = 20
        lr_at_20 = trainer._get_lr()
        assert abs(lr_at_20 - cfg.learning_rate) < 1e-6, \
            f"After warmup LR should be {cfg.learning_rate}, got {lr_at_20}"

        # After warmup: should decay
        trainer.step = 80
        lr_at_80 = trainer._get_lr()
        assert lr_at_80 < cfg.learning_rate, \
            f"After warmup LR should decay: {lr_at_80} < {cfg.learning_rate}"
