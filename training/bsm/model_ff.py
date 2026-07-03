"""
Feedforward Binary Neural Network — sliding window next-token prediction.
Supports: sign activation, learnable threshold, ternary weights.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def ste_binarize(x: Tensor) -> Tensor:
    x_bin = torch.where(x >= 0, 1.0, -1.0)
    return x_bin.detach() + x - x.detach()


def ste_binarize_thresh(x: Tensor, tau: Tensor) -> Tensor:
    """sign(x - tau) with STE. tau receives gradient via x - tau."""
    x_shifted = x - tau
    x_bin = torch.where(x >= tau, 1.0, -1.0)
    return x_bin.detach() + x_shifted - x_shifted.detach()


def ste_ternary(x: Tensor, tau: Tensor) -> Tensor:
    """Ternary {-1, 0, +1} with STE for x, identity for tau."""
    x_ter = torch.where(x > tau, 1.0, torch.where(x < -tau, -1.0, 0.0))
    return x_ter.detach() + x - x.detach()


def int_to_bits(tokens: Tensor, num_bits: int = 12) -> Tensor:
    mask = 2 ** torch.arange(num_bits - 1, -1, -1, device=tokens.device)
    bits = (tokens.unsqueeze(-1) & mask).ne(0).float() * 2 - 1
    return bits


def bits_to_int(bits: Tensor, num_bits: int = 12) -> Tensor:
    mask = 2 ** torch.arange(num_bits - 1, -1, -1, device=bits.device)
    bits_01 = (bits > 0).float()
    return (bits_01 * mask).sum(dim=-1).long()


class FeedforwardBinary(nn.Module):
    """
    Feedforward binary neural network.
    
    Modes:
      'sign':    sign(W) — standard ±1 with STE
      'thresh':  sign(W - τ) — learnable binarization threshold
      'ternary': ternary {-1, 0, +1} with threshold τ
    """

    def __init__(self, vocab_size: int = 4096, context_size: int = 4,
                 hidden_dims: list = None, num_bits: int = 12,
                 mode: str = "sign"):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.num_bits = num_bits
        self.mode = mode
        if hidden_dims is None:
            hidden_dims = [384, 128, 64]

        dims = [context_size * num_bits] + hidden_dims + [num_bits]

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.tau = nn.ParameterList()

        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            w = nn.Parameter(torch.zeros(out_dim, in_dim))
            b = nn.Parameter(torch.zeros(out_dim))
            nn.init.normal_(w, mean=0.0, std=0.01)
            nn.init.normal_(b, mean=0.0, std=0.01)
            self.weights.append(w)
            self.biases.append(b)

            if i < len(dims) - 1:
                if mode == "thresh":
                    tau = nn.Parameter(torch.zeros(out_dim, 1))
                elif mode == "ternary":
                    tau = nn.Parameter(torch.ones(out_dim, 1) * 0.1)
                else:
                    tau = None
                self.tau.append(tau)

        self.binarized = False

    def _forward_binary(self, h: Tensor, i: int) -> Tensor:
        w = self.weights[i]
        b = self.biases[i]
        is_last = (i == len(self.weights) - 1)

        if is_last:
            h = F.linear(h, w, b)
        elif self.mode == "sign":
            w_b = ste_binarize(w)
            b_b = ste_binarize(b)
            h = F.linear(h, w_b, b_b)
            h = torch.tanh(h)
        elif self.mode == "thresh":
            tau = self.tau[i].to(w.device)
            w_b = ste_binarize_thresh(w, tau)
            b_b = ste_binarize(b)
            h = F.linear(h, w_b, b_b)
            h = torch.tanh(h)
        elif self.mode == "ternary":
            tau = self.tau[i].to(w.device)
            w_t = ste_ternary(w, tau)
            b_t = ste_ternary(b, tau.squeeze())
            h = F.linear(h, w_t, b_t)
            h = torch.tanh(h)
        return h

    def forward(self, bits: Tensor) -> Tensor:
        if not self.binarized:
            return self._forward_fp32(bits)
        h = bits
        for i in range(len(self.weights)):
            h = self._forward_binary(h, i)
        return h

    def _forward_fp32(self, bits: Tensor) -> Tensor:
        h = bits
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            h = F.linear(h, w, b)
            if i < len(self.weights) - 1:
                h = torch.tanh(h)
        return h

    def binarize_weights(self):
        for i in range(len(self.weights)):
            with torch.no_grad():
                w, b = self.weights[i].data, self.biases[i].data
                if self.mode == "sign":
                    w.copy_(torch.sign(w))
                    b.copy_(torch.sign(b))
                elif self.mode == "thresh":
                    tau = self.tau[i]
                    w.copy_(torch.sign(w - tau))
                    b.copy_(torch.sign(b))
                elif self.mode == "ternary":
                    tau = self.tau[i]
                    w.copy_(torch.where(w > tau, 1.0,
                             torch.where(w < -tau, -1.0, 0.0)))
                    b.copy_(torch.sign(b))
        self.binarized = True

    def predict(self, logits: Tensor) -> Tensor:
        return bits_to_int(torch.sign(logits), self.num_bits)

    def compute_loss(self, logits: Tensor, targets: Tensor,
                     label_smoothing: float = 0.0) -> Tensor:
        target_bits = int_to_bits(targets, self.num_bits)
        t01 = (target_bits.reshape(-1, self.num_bits) + 1) / 2
        if label_smoothing > 0.0:
            t01 = t01 * (1 - label_smoothing) + label_smoothing / 2
        return F.binary_cross_entropy_with_logits(
            logits.reshape(-1, self.num_bits), t01
        )

    def accuracy(self, logits: Tensor, targets: Tensor) -> Tensor:
        return (self.predict(logits) == targets).float().mean()

    def clamp_binary(self):
        for i in range(len(self.weights) - 1):
            self.weights[i].data.clamp_(-1.0, 1.0)
            self.biases[i].data.clamp_(-1.0, 1.0)

    def summary(self) -> str:
        phase = "BINARIZED" if self.binarized else "FP32"
        binary_bits = sum(w.numel() + b.numel()
                          for w, b in zip(self.weights[:-1], self.biases[:-1]))
        fp32_bytes = (self.weights[-1].numel() + self.biases[-1].numel()) * 4
        total_bytes = binary_bits // 8 + fp32_bytes
        lines = [f"FeedforwardBinary ({self.mode}, {phase})"]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            if i == len(self.weights) - 1:
                tag = "FP32→logit"
            elif self.mode == "sign":
                tag = f"sign(±1)→tanh"
            elif self.mode == "thresh":
                tag = f"sign(W-τ)→tanh"
            elif self.mode == "ternary":
                tag = f"ternary{{-1,0,+1}}→tanh"
            lines.append(f"  L{i}: {w.shape[1]}→{w.shape[0]} {tag}")
        lines.append(f"  Binary: {binary_bits:,} bits ({binary_bits//8:,} bytes)")
        lines.append(f"  FP32:   {fp32_bytes:,} bytes")
        lines.append(f"  Total:  {total_bytes:,} bytes ({total_bytes/1024:.2f} KB)")
        return "\n".join(lines)
