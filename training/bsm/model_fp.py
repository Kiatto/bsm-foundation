"""
Binary State Machine — BSM v2.6n (noise + label smoothing)

v2.6fp + state noise (training only) + label smoothing.

State noise: Gaussian noise before state binarization.
  Forces state to encode information robustly against perturbation.
  Zero parameters, zero inference cost.

Label smoothing: BCE targets {0,1} → {ε, 1-ε}.
  Prevents confident-wrong predictions from dominating gradients.
  Keeps decoder from collapsing.

68 KB T + 2 KB W1 + 2 KB W2 + 0.5 KB D_dec + 1.5 KB B_dec (FP32) = ~74 KB
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def ste_binarize(x: Tensor) -> Tensor:
    x_bin = torch.where(x >= 0, 1.0, -1.0)
    return x_bin.detach() + x - x.detach()


def binary_matmul(x: Tensor, w: Tensor) -> Tensor:
    return torch.matmul(x, w.t())


def int_to_bits(tokens: Tensor, num_bits: int = 12) -> Tensor:
    mask = 2 ** torch.arange(num_bits - 1, -1, -1, device=tokens.device)
    bits = (tokens.unsqueeze(-1) & mask).ne(0).float() * 2 - 1
    return bits


def bits_to_int(bits: Tensor, num_bits: int = 12) -> Tensor:
    mask = 2 ** torch.arange(num_bits - 1, -1, -1, device=bits.device)
    bits_01 = (bits > 0).float()
    return (bits_01 * mask).sum(dim=-1).long()


class BinaryStateMachine(nn.Module):
    """
    BSM v2.6fp — two-layer correction + FP32 decoder.

    Same as v2.6 but B_dec is FP32 (no binarization, no clamping).
    """

    def __init__(self, vocab_size: int = 4096, hidden_dim: int = 128,
                 dec_dim: int = 32, noise_std: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.dec_dim = dec_dim
        self.num_bits = int(math.log2(vocab_size))  # 12
        self.noise_std = noise_std

        # State transition (all binary)
        self.T = nn.Parameter(torch.zeros(vocab_size, hidden_dim))
        self.W1 = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.W2 = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))

        # Decoder: D_dec binary, B_dec FP32 (the controlled variable)
        self.D_dec = nn.Parameter(torch.zeros(dec_dim, hidden_dim))
        self.B_dec = nn.Parameter(torch.zeros(self.num_bits, dec_dim))  # FP32!

        for p in [self.T, self.W1, self.W2, self.D_dec, self.B_dec]:
            nn.init.normal_(p, mean=0.0, std=0.01)

    def init_state(self, batch_size: int) -> Tensor:
        return torch.full((batch_size, self.hidden_dim), -1.0)

    def _clamp_params(self):
        # Only clamp binary params; B_dec stays FP32
        for p in [self.T, self.W1, self.W2, self.D_dec]:
            p.data.clamp_(-1.0, 1.0)

    def step(self, state: Tensor, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        T_bin = ste_binarize(self.T)
        W1b = ste_binarize(self.W1)
        W2b = ste_binarize(self.W2)
        Db = ste_binarize(self.D_dec)
        # B_dec is NOT binarized — used as FP32 directly

        emb = T_bin[token_ids]

        a = ste_binarize(state.detach() * emb)
        h1 = ste_binarize(binary_matmul(a, W1b))
        h2 = ste_binarize(binary_matmul(h1, W2b)) * a
        new_state_raw = state * emb * h2
        if self.training and self.noise_std > 0:
            new_state_raw = new_state_raw + torch.randn_like(new_state_raw) * self.noise_std
        new_state = ste_binarize(new_state_raw)

        h_dec = ste_binarize(binary_matmul(new_state, Db))
        # FP32 matmul: h_dec (binary) @ B_dec.t() (FP32) → FP32 logits
        bit_logits = torch.matmul(h_dec, self.B_dec.t())

        return new_state, bit_logits

    def forward(self, token_ids: Tensor, state: Tensor = None,
                chunk_size: int = 0) -> tuple[Tensor, Tensor]:
        self._clamp_params()
        B, T = token_ids.shape
        device = token_ids.device
        if state is None:
            state = torch.full((B, self.hidden_dim), -1.0, device=device)
        logits_list = []
        for t in range(T):
            state, logits = self.step(state, token_ids[:, t])
            logits_list.append(logits)
            if chunk_size > 0 and (t + 1) % chunk_size == 0 and t < T - 1:
                state = state.detach()
        return torch.stack(logits_list, dim=1), state

    def decode_to_tokens(self, bit_logits: Tensor) -> Tensor:
        bits = torch.where(bit_logits > 0, 1.0, -1.0)
        return bits_to_int(bits, self.num_bits)

    def compute_loss(self, bit_logits: Tensor, targets: Tensor,
                     label_smoothing: float = 0.0) -> Tensor:
        target_bits = int_to_bits(targets, self.num_bits)  # {-1, +1}
        # Convert to {0, 1}
        targets_01 = (target_bits.reshape(-1, self.num_bits) + 1) / 2
        if label_smoothing > 0.0:
            # targets_01: {0 → ε, 1 → 1-ε} where ε = label_smoothing/2
            targets_01 = targets_01 * (1 - label_smoothing) + label_smoothing / 2
        return F.binary_cross_entropy_with_logits(
            bit_logits.reshape(-1, self.num_bits), targets_01,
        )

    def accuracy(self, bit_logits: Tensor, targets: Tensor) -> Tensor:
        preds = self.decode_to_tokens(bit_logits)
        return (preds == targets).float().mean()

    def count_params(self) -> dict:
        V, D = self.vocab_size, self.hidden_dim
        H = self.dec_dim
        L = self.num_bits
        binary_bits = V * D + 2 * D * D + H * D   # binary params in bits
        fp32_bytes = L * H * 4                     # B_dec in bytes
        total_bytes = binary_bits // 8 + fp32_bytes
        return {
            "T (bin)": V * D,
            "W1-2 (bin)": 2 * D * D,
            "D_dec (bin)": H * D,
            "B_dec (FP32)": f"{L*H} floats ({fp32_bytes} bytes)",
            "total_bytes": total_bytes,
        }

    def summary(self) -> str:
        info = self.count_params()
        return (
            f"BSM-{self.hidden_dim} v2.6n (noise={self.noise_std})\n"
            f"  Vocab:    {self.vocab_size}\n"
            f"  State:    {self.hidden_dim} bits\n"
            f"  T (bin):        {info['T (bin)']:,} bits ({info['T (bin)']//8:,} bytes)\n"
            f"  W1-2 (bin):     {info['W1-2 (bin)']:,} bits ({info['W1-2 (bin)']//8:,} bytes)\n"
            f"  D_dec (bin):    {info['D_dec (bin)']:,} bits ({info['D_dec (bin)']//8:,} bytes)\n"
            f"  {info['B_dec (FP32)']}\n"
            f"  Total:          {info['total_bytes']:,} bytes "
            f"({info['total_bytes']/1024:.1f} KB)\n"
        )
