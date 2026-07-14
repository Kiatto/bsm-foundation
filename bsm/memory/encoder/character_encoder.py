"""
character_encoder.py — Character n-gram encoder per BSM.

Proietta il testo in un vettore binario {-1, +1} usando caratteri n-gram
(2-5 caratteri). Cattura similarità di stringa: "Seattle" e "Seattle,
Washington" condividono n-gram, mentre "United" no.

Questo encoder NON viene ingannato dalla sovrapposizione lessicale casuale.
"""

import hashlib
import numpy as np


class CharacterEncoder:
    """Character n-gram encoder.

    Fissa un vocabolario di n-gram dal corpus di fit, poi proietta ogni
    testo in un vettore binario D-dimensionale. Similarità basata su
    pattern di stringa, non su semantica bag-of-words.
    """

    def __init__(self, state_dim: int = 256, ng_min: int = 2, ng_max: int = 5):
        self.state_dim = state_dim
        self.ng_min = ng_min
        self.ng_max = ng_max
        self._vocab_size = 0
        self._name = "character"

    def _extract_ngrams(self, text: str) -> set:
        """Restituisce tutti gli n-gram di caratteri (2-5) dal testo."""
        lower = text.lower()
        ngrams = set()
        for n in range(self.ng_min, self.ng_max + 1):
            for i in range(len(lower) - n + 1):
                ngrams.add(lower[i:i + n])
        return ngrams

    def fit(self, texts):
        """Build n-gram vocabulary (determina il numero di feature)."""
        all_ngrams = set()
        for t in texts:
            all_ngrams.update(self._extract_ngrams(t))
        self._vocab_size = len(all_ngrams)
        return self

    def encode(self, text_or_texts):
        """Text → (N, D) int8 array in {-1, +1}.

        Ogni n-gram viene hashato a una posizione nel vettore.
        Il segno è determinato dalla parità del conteggio.
        """
        if isinstance(text_or_texts, str):
            texts = [text_or_texts]
        else:
            texts = list(text_or_texts)

        results = []
        for text in texts:
            ngrams = self._extract_ngrams(text)
            raw = np.zeros(self.state_dim, dtype=np.float32)
            for ng in ngrams:
                h = int(hashlib.md5(ng.encode()).hexdigest(), 16) % self.state_dim
                raw[h] += 1.0
            binary = np.where(raw > 0, 1, -1).astype(np.int8)
            results.append(binary)

        out = np.array(results, dtype=np.int8)
        return out[0] if isinstance(text_or_texts, str) else out

    def __repr__(self):
        return f"CharacterEncoder(D={self.state_dim}, ngram={self.ng_min}-{self.ng_max})"
