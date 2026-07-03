"""Trainer for Binary State Machine — pure CPU, no CUDA dependencies."""

import json
import time
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import BinaryStateMachine


@dataclass
class TrainerConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    warmup_steps: int = 500
    max_steps: int = 10000
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    gradient_clip: float = 1.0
    seed: int = 42
    output_dir: str = "checkpoints/bsm"


class BSMTrainer:
    """
    Trains BinaryStateMachine with STE + AdamW.

    Loss: binary cross-entropy on the logV output bits.
    No softmax over vocabulary — O(log V) loss computation.
    """

    def __init__(
        self,
        model: BinaryStateMachine,
        config: TrainerConfig,
        train_dataset,
        eval_dataset=None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )

        self.step = 0
        self.best_accuracy = 0.0
        self.train_losses = []
        self.train_accuracies = []
        self.eval_accuracies = []
        self.state_metrics = []  # Persistence, Plasticity, Capacity

    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m"
        else:
            return f"{seconds/3600:.1f}h"

    def _measure_state_properties(self, states: torch.Tensor) -> dict:
        """
        Measure state properties for diagnostics.

        states: [n_positions, D]  — sequence of states
        """
        D = states.shape[-1]

        # Plasticity: fraction of bits that change per step
        if states.shape[0] > 1:
            flips = (states[1:] != states[:-1]).float().mean().item()
        else:
            flips = 0.0

        # Capacity: number of unique states in window
        n_unique = len(states.unique(dim=0))

        # Persistence: avg time until a bit flips (estimate)
        bit_sequences = states.float().t()  # [D, T]
        # For each bit, count consecutive same values
        persistence_per_bit = []
        for d in range(min(D, 64)):  # Sample 64 bits
            seq = bit_sequences[d]
            if seq.shape[0] > 1:
                changes = (seq[1:] != seq[:-1]).float()
                if changes.sum() > 0:
                    avg_run = (1.0 / (changes.mean() + 1e-8))
                else:
                    avg_run = float(seq.shape[0])
                persistence_per_bit.append(avg_run)
        avg_persistence = sum(persistence_per_bit) / len(persistence_per_bit) if persistence_per_bit else 0.0

        # Entropy estimate
        freq = {}
        for s in states:
            key = s[:16].tolist()  # Sample first 16 bits as hash
            key = tuple(key)
            freq[key] = freq.get(key, 0) + 1
        total = sum(freq.values())
        entropy = -sum(c/total * math.log2(c/total) for c in freq.values()) if total > 0 else 0

        return {
            "plasticity": round(float(flips), 4),
            "capacity": n_unique,
            "persistence": round(float(avg_persistence), 1),
            "entropy_16bit": round(float(entropy), 2),
        }

    def train(self) -> dict:
        model = self.model
        config = self.config

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        eval_loader = None
        if self.eval_dataset is not None:
            eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
            )

        model.train()
        data_iter = iter(train_loader)
        t_start = time.time()

        print(f"[BSM] Training {model.summary()}")
        print(f"[BSM] Steps: {config.max_steps}  Batch: {config.batch_size}  LR: {config.learning_rate}")
        print()

        while self.step < config.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids, target_ids = batch
            B, T = input_ids.shape

            # Forward
            decode_logits, _ = model(input_ids)
            loss = model.compute_loss(decode_logits, target_ids)

            # Metrics
            with torch.no_grad():
                acc = model.accuracy(decode_logits, target_ids)
                bit_acc = model.bit_accuracy(decode_logits, target_ids)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            if config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

            self.optimizer.step()
            self.model._clamp_params()
            self.step += 1

            self.train_losses.append(loss.item())
            self.train_accuracies.append(acc.item())

            # Logging
            if self.step % config.log_interval == 0:
                elapsed = time.time() - t_start
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  step {self.step:>6d} | "
                    f"loss {loss.item():.4f} | "
                    f"acc {acc.item()*100:.1f}% | "
                    f"bit_acc {bit_acc.item()*100:.1f}% | "
                    f"lr {lr:.2e} | "
                    f"{self._format_time(elapsed)}"
                )
                t_start = time.time()

            # Evaluation
            if self.step % config.eval_interval == 0 and eval_loader is not None:
                eval_acc, eval_loss, state_metrics = self._evaluate(eval_loader)
                self.eval_accuracies.append(eval_acc)
                print(
                    f"  [eval] acc {eval_acc*100:.1f}%  "
                    f"loss {eval_loss:.4f}  "
                    f"cap={state_metrics['capacity']}  "
                    f"plas={state_metrics['plasticity']:.2f}  "
                    f"pers={state_metrics['persistence']:.1f}"
                )

                if eval_acc > self.best_accuracy:
                    self.best_accuracy = eval_acc
                    self._save_checkpoint("best")

            # Checkpoint
            if self.step % config.save_interval == 0:
                self._save_checkpoint(f"step_{self.step}")

        self._save_checkpoint("final")
        self._save_metrics()
        print(f"\n[BSM] Training complete. Best accuracy: {self.best_accuracy*100:.1f}%")
        return {
            "final_loss": self.train_losses[-1] if self.train_losses else 0.0,
            "best_accuracy": self.best_accuracy,
            "steps": self.step,
        }

    def _evaluate(self, eval_loader) -> tuple:
        """Evaluate on validation set with state metrics."""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        all_states = []

        with torch.no_grad():
            for input_ids, target_ids in eval_loader:
                decode_logits, final_state = self.model(input_ids)
                loss = self.model.compute_loss(decode_logits, target_ids)
                acc = self.model.accuracy(decode_logits, target_ids)
                total_loss += loss.item()
                total_acc += acc.item()
                num_batches += 1

                if num_batches <= 5:
                    # Collect state trajectories for metrics
                    B = input_ids.shape[0]
                    state = torch.full((B, self.model.hidden_dim), -1.0)
                    for t in range(input_ids.shape[1]):
                        state, _ = self.model.step(state, input_ids[:, t])
                        all_states.append(state[0].cpu())

        self.model.train()

        avg_loss = total_loss / max(num_batches, 1)
        avg_acc = total_acc / max(num_batches, 1)

        state_metrics = {}
        if all_states:
            states_tensor = torch.stack(all_states)  # [T, D]
            state_metrics = self._measure_state_properties(states_tensor)

        return avg_acc, avg_loss, state_metrics

    def _save_checkpoint(self, tag: str) -> None:
        path = self.output_dir / f"checkpoint_{tag}.pt"
        data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": {
                "vocab_size": self.model.vocab_size,
                "hidden_dim": self.model.hidden_dim,
            },
            "trainer_config": {
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "max_steps": self.config.max_steps,
            },
            "step": self.step,
            "best_accuracy": self.best_accuracy,
            "train_losses": self.train_losses,
        }
        torch.save(data, path)
        print(f"    saved: {path}")

    def _save_metrics(self) -> None:
        path = self.output_dir / "metrics.json"
        data = {
            "final_loss": self.train_losses[-1] if self.train_losses else None,
            "best_accuracy": self.best_accuracy,
            "steps": self.step,
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
            "eval_accuracies": self.eval_accuracies,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
