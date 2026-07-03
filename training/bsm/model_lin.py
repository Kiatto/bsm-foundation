"""
Linear Binary RNN — BSM vL

No embedding table. No correction layers.
State evolves by adding token hypervector:

    emb = W_hash · bits(token)     # 12-bit → 128D hypervector
    S'  = sign(S + emb)            # add & binarize

Parametri:
    W_hash:        1.5K bit  (128×12)
    D_dec:         4K bit    (32×128)
    B_dec (FP32):  1.5KB     (12×32 float32)
    Totale:        ~2KB

Gradiente: 6144 contributi per parametro per batch (vs 0.5 con T).
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
    BSM vL — Linear Binary RNN.

    No embedding table. State is a sum of token hypervectors.
    """

    def __init__(self, vocab_size: int = 4096, hidden_dim: int = 128,
                 dec_dim: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.dec_dim = dec_dim
        self.num_bits = int(math.log2(vocab_size))  # 12

        # The ONLY parameters:
        self.W_hash = nn.Parameter(torch.zeros(hidden_dim, self.num_bits))
        self.D_dec = nn.Parameter(torch.zeros(dec_dim, hidden_dim))
        self.B_dec = nn.Parameter(torch.zeros(self.num_bits, dec_dim))

        for p in [self.W_hash, self.D_dec, self.B_dec]:
            nn.init.normal_(p, mean=0.0, std=0.01)

    def init_state(self, batch_size: int) -> Tensor:
        return torch.full((batch_size, self.hidden_dim), -1.0)

    def _clamp_params(self):
        # Only binary params; B_dec stays FP32
        for p in [self.W_hash, self.D_dec]:
            p.data.clamp_(-1.0, 1.0)

    def step(self, state: Tensor, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        # Token → 12-bit binary code
        bits = int_to_bits(token_ids, self.num_bits)  # [B, 12]

        # Project to 128D hypervector (binary)
        Wh = ste_binarize(self.W_hash)
        emb = binary_matmul(bits, Wh)  # [B, 128], range [-12, 12]

        # State update: add hypervector & binarize
        new_state = ste_binarize(state + emb)  # [B, 128]

        # Decoder (same as v2.6fp)
        Db = ste_binarize(self.D_dec)
        h_dec = ste_binarize(binary_matmul(new_state, Db))
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
        binary_bits = D * L + H * D   # W_hash, D_dec
        fp32_bytes = L * H * 4        # B_dec
        total_bytes = binary_bits // 8 + fp32_bytes
        return {
            "W_hash (bin)": D * L,
            "D_dec (bin)": H * D,
            "B_dec (FP32)": f"{L*H} floats ({fp32_bytes} bytes)",
            "total_bytes": total_bytes,
        }

    def summary(self) -> str:
        info = self.count_params()
        return (
            f"BSM-{self.hidden_dim} vL (Linear RNN)\n"
            f"  Vocab:    {self.vocab_size}\n"
            f"  State:    {self.hidden_dim} bits\n"
            f"  W_hash:   {info['W_hash (bin)']:,} bits ({info['W_hash (bin)']//8:,} bytes)\n"
            f"  D_dec:    {info['D_dec (bin)']:,} bits ({info['D_dec (bin)']//8:,} bytes)\n"
            f"  {info['B_dec (FP32)']}\n"
            f"  Total:    {info['total_bytes']:,} bytes "
            f"({info['total_bytes']/1024:.2f} KB)\n"
        )
