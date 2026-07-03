"""
Binary State Machine — BSM v2.6

Two-layer correction + binary 12-bit decoder.
Predicts next token as a 12-bit binary code (direct token ID).

    S' = S ⊕ T[x] ⊕ F(S ⊕ T[x])
    bits = sign(h · bit_decoder)      where h = sign(D_dec · S')
    token = bits_to_int(bits)

68 KB T + 2 KB W1 + 2 KB W2 + 0.5 KB D_dec + 48 B bit_decoder = ~72.5 KB
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
    """Convert token IDs (int) to binary ±1 vectors."""
    mask = 2 ** torch.arange(num_bits - 1, -1, -1, device=tokens.device)
    bits = (tokens.unsqueeze(-1) & mask).ne(0).float() * 2 - 1
    return bits


def bits_to_int(bits: Tensor, num_bits: int = 12) -> Tensor:
    """Convert binary ±1 vectors to token IDs."""
    mask = 2 ** torch.arange(num_bits - 1, -1, -1, device=bits.device)
    bits_01 = (bits > 0).float()
    return (bits_01 * mask).sum(dim=-1).long()


class BinaryStateMachine(nn.Module):
    """
    BSM v2.6 — two-layer correction + 12-bit binary decoder.

    State transitions:
        a = S ⊕ T[x]
        h₁ = σ(W₁ · a)
        h₂ = σ(W₂ · h₁ ⊕ a)
        S' = S ⊕ T[x] ⊕ h₂

    Prediction:
        h_dec = σ(D_dec · S')          # 32-bit decoder state
        bit[i] = σ(h_dec · B_dec[i])   # 12-bit binary prediction
        token_id = bits_to_int(bits)
    """

    def __init__(self, vocab_size: int = 4096, hidden_dim: int = 128,
                 dec_dim: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.dec_dim = dec_dim
        self.num_bits = int(math.log2(vocab_size))  # 12

        # State transition
        self.T = nn.Parameter(torch.zeros(vocab_size, hidden_dim))
        self.W1 = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.W2 = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))

        # Binary decoder: 128 → 32 → 12 bits
        self.D_dec = nn.Parameter(torch.zeros(dec_dim, hidden_dim))
        self.B_dec = nn.Parameter(torch.zeros(self.num_bits, dec_dim))

        for p in [self.T, self.W1, self.W2, self.D_dec, self.B_dec]:
            nn.init.normal_(p, mean=0.0, std=0.01)

    def init_state(self, batch_size: int) -> Tensor:
        return torch.full((batch_size, self.hidden_dim), -1.0)

    def _clamp_params(self):
        pass  # params grow unbounded; binarization via sign() handles quantization

    def step(self, state: Tensor, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        T_bin = ste_binarize(self.T)
        W1b = ste_binarize(self.W1)
        W2b = ste_binarize(self.W2)
        Db = ste_binarize(self.D_dec)
        Bb = ste_binarize(self.B_dec)

        emb = T_bin[token_ids]

        # Two-layer correction (state detached for stable gradients)
        a = ste_binarize(state.detach() * emb)
        h1 = ste_binarize(binary_matmul(a, W1b))
        h2 = ste_binarize(binary_matmul(h1, W2b)) * a

        new_state = ste_binarize(state * emb * h2)

        # Binary decoder: 128 → 32 → 12 bits
        h_dec = ste_binarize(binary_matmul(new_state, Db))
        bit_logits = binary_matmul(h_dec, Bb)  # [B, 12]

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

    def compute_loss(self, bit_logits: Tensor, targets: Tensor) -> Tensor:
        target_bits = int_to_bits(targets, self.num_bits)
        return F.binary_cross_entropy_with_logits(
            bit_logits.reshape(-1, self.num_bits),
            (target_bits.reshape(-1, self.num_bits) + 1) / 2,
        )

    def accuracy(self, bit_logits: Tensor, targets: Tensor) -> Tensor:
        preds = self.decode_to_tokens(bit_logits)
        return (preds == targets).float().mean()

    def count_params(self) -> dict:
        V, D = self.vocab_size, self.hidden_dim
        H = self.dec_dim
        L = self.num_bits
        total = V * D + 2 * D * D + H * D + L * H
        return {
            "T": V * D, "W1-2": 2 * D * D,
            "D_dec": H * D, "B_dec": L * H,
            "total_bits": total, "total_bytes": total // 8,
        }

    def summary(self) -> str:
        info = self.count_params()
        return (
            f"BSM-{self.hidden_dim} v2.6\n"
            f"  Vocab:    {self.vocab_size}\n"
            f"  State:    {self.hidden_dim} bits\n"
            f"  Decoder:  {self.dec_dim} bits → {self.num_bits} bits\n"
            f"  T:        {info['T']:,} bits ({info['T']//8:,} bytes)\n"
            f"  W1-2:     {info['W1-2']:,} bits ({info['W1-2']//8:,} bytes)\n"
            f"  D_dec:    {info['D_dec']:,} bits ({info['D_dec']//8:,} bytes)\n"
            f"  B_dec:    {info['B_dec']:,} bits ({info['B_dec']//8:,} bytes)\n"
            f"  Total:    {info['total_bits']:,} bits "
            f"({info['total_bytes']:,} bytes "
            f"≈ {info['total_bytes']/1024:.1f} KB)\n"
        )
