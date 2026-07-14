"""
entity_encoder.py — Entity encoder via one-bit MinHash.

Estrae le entità (nomi propri, camelCase, sigle) dal testo e codifica
l'INSIEME di entità come sketch binario MinHash a state_dim bit.

Proprietà chiave (che rende l'encoder un cittadino legittimo dello
spazio di Hamming): per due insiemi di entità A e B con similarità di
Jaccard J, ogni bit dello sketch coincide con probabilità (1+J)/2,
quindi

    E[hamming(sketch_A, sketch_B)] = state_dim * (1 - J) / 2

La distanza di Hamming tra sketch È una stima della distanza di
Jaccard — nessuna metrica travestita, nessuna memoria dedicata.

"Amazon.com Inc. was founded by Jeff Bezos in 1994."
  → entities: {Amazon, Inc, Jeff, Bezos}
  → sketch MinHash a 256 bit dell'insieme

Testi senza entità vengono codificati con MinHash sulle parole
(fallback deterministico, mai vettore degenere).
"""

import re
import hashlib
import numpy as np

from bsm.memory.store.memory_store import MemoryStore


STOP_WORDS = frozenset({
    "the", "a", "an", "in", "of", "to", "for", "and", "or",
    "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can",
    "that", "this", "these", "those", "it", "its",
    "at", "by", "with", "from", "as", "on", "not",
    "who", "what", "where", "when", "why", "how", "which",
})


def _minhash_sketch(items: set, state_dim: int) -> np.ndarray:
    """One-bit MinHash: per ogni bit j, min dell'hash h_j sugli item,
    tenendo il bit meno significativo del minimo.

    Serve un hash con bit indipendenti: CRC32 è lineare su GF(2) e
    produce parità correlate tra insiemi diversi (distanze lontane
    dalla teoria D*(1-J)/2).  md5 non ha questo problema.
    Deterministico, zero training.
    """
    sketch = np.empty(state_dim, dtype=np.int8)
    hashes = {it: hashlib.md5(it.encode()).digest() for it in items}
    for j in range(state_dim):
        salt = f"{j}|".encode()
        m = min(int.from_bytes(
            hashlib.md5(salt + h).digest()[:8], "little")
            for h in hashes.values())
        sketch[j] = 1 if (m & 1) else -1
    return sketch


class EntityEncoder:
    """Entity-set encoder: testo → sketch MinHash binario in {-1, +1}.

    Compatibile con il contratto degli altri encoder BSM (encode →
    int8 {-1,+1} a state_dim), quindi utilizzabile in un BSM standard
    con ricerca Hamming.
    """

    def __init__(self, state_dim: int = 256):
        self.state_dim = state_dim
        self._vocab_size = 0
        self._name = "entity"
        self._store = MemoryStore(state_dim=state_dim)

    def _extract_entities(self, text: str) -> list:
        """Estrae parole con iniziale maiuscola (nomi propri) e camelCase."""
        entities = set()
        # Parole con iniziale maiuscola
        for w in re.findall(r"\b[A-Z][a-zA-Z]*\b", text):
            if len(w) > 1 and w.lower() not in STOP_WORDS:
                entities.add(w)
        # camelCase (almeno una maiuscola interna, prima lettera minuscola)
        for w in re.findall(r"\b[a-z]+[A-Z][a-zA-Z]*\b", text):
            if len(w) > 1 and w.lower() not in STOP_WORDS:
                entities.add(w)
        # Parole TUTTE MAIUSCOLE (sigle come LLC, Inc.)
        for w in re.findall(r"\b[A-Z]{2,}\b", text):
            if len(w) > 1 and w.lower() not in STOP_WORDS:
                entities.add(w)
        return list(entities)

    def fit(self, texts):
        """Statistiche sul vocabolario (nessun training necessario)."""
        all_entities = set()
        for t in texts:
            all_entities.update(self._extract_entities(t))
        self._vocab_size = len(all_entities)
        return self

    def encode(self, text_or_texts):
        """Text → (N, D) int8 array in {-1, +1} (sketch MinHash)."""
        if isinstance(text_or_texts, str):
            texts = [text_or_texts]
        else:
            texts = list(text_or_texts)

        results = []
        for text in texts:
            items = {e.lower() for e in self._extract_entities(text)}
            if not items:
                # Fallback: MinHash sulle parole non-stop (deterministico)
                items = {w for w in text.lower().split()
                         if w and w not in STOP_WORDS}
            if not items:
                items = {text.lower() or "∅"}
            results.append(_minhash_sketch(items, self.state_dim))

        out = np.array(results, dtype=np.int8)
        return out[0] if isinstance(text_or_texts, str) else out

    def observe(self, text: str, payload: dict):
        """Osserva un documento nella memoria interna (Hamming su sketch)."""
        self._store.put(self.encode(text), payload)

    def recall(self, query: str, k: int = 10):
        """Ricerca per distanza di Hamming tra sketch MinHash.

        La distanza restituita stima D*(1-J)/2 rispetto al Jaccard J
        tra gli insiemi di entità.
        """
        return self._store.search(self.encode(query), k=k)

    def __repr__(self):
        return f"EntityEncoder(D={self.state_dim}, minhash)"
