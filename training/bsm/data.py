"""BSM Dataset — reuses the tokenizer from blm."""

import torch
from torch.utils.data import Dataset
from pathlib import Path


class TextDataset(Dataset):
    """
    Tokenize once, chunk into fixed-length sequences.

    Each sample: (input_ids, target_ids)
        input_ids:  [seq_len]
        target_ids: [seq_len]  (shifted by 1)
    """

    def __init__(self, tokenizer, texts: list[str], seq_len: int = 128):
        self.seq_len = seq_len

        all_tokens: list[int] = []
        for text in texts:
            all_tokens.extend(tokenizer.encode(text))
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)

        self._sequences = []
        stride = seq_len
        for start in range(0, len(self.tokens) - seq_len, stride):
            chunk = self.tokens[start:start + seq_len + 1]
            self._sequences.append((chunk[:-1], chunk[1:]))

    def __len__(self):
        return len(self._sequences)

    def __getitem__(self, idx):
        return self._sequences[idx]

    @property
    def num_tokens(self):
        return len(self.tokens)
