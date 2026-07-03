"""
Binary-Weight RNN — BSM vF (FP32 state)

Weights: binary (±1 via STE)
State:   FP32, continuous (tanh activation)
Decoder: FP32 logits

State update:
    emb = W_hash · bits(token)          # binary proj: 12 → 128
    S'  = tanh((W1·S + W2·emb) / √D)   # FP32 state, binary weights

Parametri (tutti binari tranne B_dec):
    W_hash:  1.5K bit  (128×12)
    W1:     16.4K bit  (128×128)
    W2:     16.4K bit  (128×128)
    D_dec:   4.1K bit  (32×128)
    B_dec:   1.5KB     (12×32 float32)
    Totale: ~6.2KB
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
    """x @ w.t() / sqrt(D) — normalized to prevent tanh saturation."""
    return torch.matmul(x, w.t()) / math.sqrt(w.shape[0])


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
    BSM vF — Binary-Weight RNN with FP32 state.
    """

    def __init__(self, vocab_size: int = 4096, hidden_dim: int = 128,
                 dec_dim: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.dec_dim = dec_dim
        self.num_bits = int(math.log2(vocab_size))

        # All binary parameters
        self.W_hash = nn.Parameter(torch.zeros(hidden_dim, self.num_bits))
        self.W1 = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.W2 = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.D_dec = nn.Parameter(torch.zeros(dec_dim, hidden_dim))
        self.B_dec = nn.Parameter(torch.zeros(self.num_bits, dec_dim))

        for p in [self.W_hash, self.W1, self.W2, self.D_dec, self.B_dec]:
            nn.init.normal_(p, mean=0.0, std=0.01)

    def init_state(self, batch_size: int) -> Tensor:
        return torch.zeros(batch_size, self.hidden_dim)

    def _clamp_params(self):
        for p in [self.W_hash, self.W1, self.W2, self.D_dec]:
            p.data.clamp_(-1.0, 1.0)

    def step(self, state: Tensor, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        Wh = ste_binarize(self.W_hash)
        W1b = ste_binarize(self.W1)
        W2b = ste_binarize(self.W2)
        Db = ste_binarize(self.D_dec)

        # Token → 128D hypervector (binary projection)
        bits = int_to_bits(token_ids, self.num_bits)
        emb = binary_matmul(bits, Wh)  # [B, 128], FP32

        # State update: FP32 state, binary weights
        s_w1 = binary_matmul(state, W1b)  # [B, 128]
        s_w2 = binary_matmul(emb, W2b)    # [B, 128]
        new_state = torch.tanh(s_w1 + s_w2)  # [B, 128], FP32

        # Decoder: FP32 state → FP32 logits
        h_dec = binary_matmul(new_state, Db)  # [B, 32], FP32
        bit_logits = torch.matmul(h_dec, self.B_dec.t())  # [B, 12]

        return new_state, bit_logits

    def forward(self, token_ids: Tensor, state: Tensor = None,
                chunk_size: int = 0) -> tuple[Tensor, Tensor]:
        self._clamp_params()
        B, T = token_ids.shape
        device = token_ids.device
        if state is None:
            state = torch.zeros(B, self.hidden_dim, device=device)
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
        target_bits = int_to_bits(targets, self.num_bits)
        targets_01 = (target_bits.reshape(-1, self.num_bits) + 1) / 2
        if label_smoothing > 0.0:
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
        binary_bits = D * L + 2 * D * D + H * D
        fp32_bytes = L * H * 4
        total_bytes = binary_bits // 8 + fp32_bytes
        return {
            "W_hash": D * L,
            "W1": D * D,
            "W2": D * D,
            "D_dec": H * D,
            "B_dec (FP32)": f"{L*H} floats ({fp32_bytes} bytes)",
            "total_bytes": total_bytes,
        }

    def summary(self) -> str:
        info = self.count_params()
        return (
            f"BSM-{self.hidden_dim} vF (FP32 state, binary weights)\n"
            f"  Vocab:    {self.vocab_size}\n"
            f"  State:    {self.hidden_dim} FP32 ({self.hidden_dim*4:,} bytes)\n"
            f"  W_hash:   {info['W_hash']:,} bits ({info['W_hash']//8:,} bytes)\n"
            f"  W1:       {info['W1']:,} bits ({info['W1']//8:,} bytes)\n"
            f"  W2:       {info['W2']:,} bits ({info['W2']//8:,} bytes)\n"
            f"  D_dec:    {info['D_dec']:,} bits ({info['D_dec']//8:,} bytes)\n"
            f"  {info['B_dec (FP32)']}\n"
            f"  Total:    {info['total_bytes']:,} bytes "
            f"({info['total_bytes']/1024:.1f} KB)\n"
        )
