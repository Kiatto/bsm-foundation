"""
ensemble.py — Retrieval Ensemble con Reciprocal Rank Fusion.

Tre spazi di proiezione indipendenti, ognuno con il proprio encoder:

   1. ProjectionEncoder  — similarità semantica globale (bag-of-words)
   2. HashEncoder        — similarità di stringa (hash-based)
   3. EntityEncoder      — similarità tra entità (Jaccard su nomi propri)

Ogni encoder osserva lo STESSO KB in un proprio spazio geometrico.
La query viene codificata in tutti e tre gli spazi, recuperata
separatamente, e i risultati vengono fusi via Reciprocal Rank Fusion (RRF).

RRF non richiede normalizzazione tra metriche diverse ed è robusto a
encoder con distribuzioni di distanza molto diverse tra loro.
"""

import numpy as np
from bsm.memory.encoder.bsm_encoder import ProjectionEncoder, HashEncoder
from bsm.memory.encoder.entity_encoder import EntityEncoder


class EnsembleRetriever:
    """Insieme di retrievers con Reciprocal Rank Fusion.

    Usage:
        ensemble = EnsembleRetriever(state_dim=256)
        ensemble.fit(knowledge_base)
        for doc in knowledge_base:
            ensemble.observe(doc, {"text": doc, "source": "kb"})

        results = ensemble.recall(query, k=10)
        # results: [(payload, rrf_distance, meta), ...]
    """

    def __init__(self, state_dim: int = 256):
        self.state_dim = state_dim
        self.encoders = {}
        self.bsms = {}
        self.weights = {}     # peso RRF per encoder, adattato dal feedback

    def _lazy_init(self):
        """Inizializza encoder e BSM al primo uso (dopo che state_dim è noto)."""
        if self.bsms:
            return
        from bsm import BSM  # lazy import per evitare circular dependency
        proj = ProjectionEncoder(state_dim=self.state_dim)
        hash_enc = HashEncoder(state_dim=self.state_dim)
        ent = EntityEncoder(state_dim=self.state_dim)
        self.encoders = {
            "projection": proj,
            "hash": hash_enc,
            "entity": ent,
        }
        self.bsms = {
            name: BSM(encoder=enc, state_dim=self.state_dim)
            for name, enc in self.encoders.items()
        }
        self.weights = {name: 1.0 for name in self.encoders}

    def fit(self, texts: list):
        """Fitta tutti gli encoder sui testi del KB."""
        self._lazy_init()
        for name, enc in self.encoders.items():
            if hasattr(enc, 'fit'):
                enc.fit(texts)

    def observe(self, text: str, payload: dict):
        """Osserva un documento in TUTTI gli spazi.

        L'EntityEncoder usa una memoria Jaccard dedicata (non BSM),
        gli altri encoder usano il BSM classico con Hamming distance.
        """
        self._lazy_init()
        for name, bsm in self.bsms.items():
            state = bsm.encode(text)
            bsm.observe(state, payload)

    def _recall_encoder(self, name: str, query: str, k: int) -> list:
        """Recupera da un singolo encoder (Hamming in tutti gli spazi:
        l'EntityEncoder MinHash mappa il Jaccard su Hamming)."""
        bsm = self.bsms[name]
        state = bsm.encode(query)
        return bsm.recall(state, k=k)

    def recall(self, query: str, k: int = 10, oversample: int = 3,
               rrf_k: int = 60):
        """Recupera da tutti i BSMs e fonde via RRF.

        Args:
            query: Testo della query.
            k: Numero di risultati da restituire.
            oversample: Quanto oversample per encoder (k * oversample).
            rrf_k: Costante RRF (tipicamente 60).

        Returns:
            Lista di (payload, rrf_distance, meta) dove rrf_distance è
            normalizzata in 0..state_dim (compatibile con i vecchi client).
        """
        self._lazy_init()

        if not self.bsms:
            return []

        all_rankings = []
        for name in self.encoders:
            results = self._recall_encoder(name, query, k=k * oversample)
            all_rankings.append(results)

        # ── Reciprocal Rank Fusion ──
        rrf_scores = {}     # doc_key → cumulative RRF score
        payloads = {}       # doc_key → payload

        for name, rankings in zip(self.encoders, all_rankings):
            w = self.weights.get(name, 1.0)
            for rank, (payload, dist, meta) in enumerate(rankings, 1):
                doc_key = (payload.get("text", str(payload))
                           if isinstance(payload, dict) else str(payload))
                if doc_key not in rrf_scores:
                    rrf_scores[doc_key] = 0.0
                    payloads[doc_key] = payload
                rrf_scores[doc_key] += w / (rrf_k + rank)

        # Normalizza RRF score in distanza 0..state_dim
        # RRF score massimo = sum(weights) * 1/(rrf_k+1)
        max_rrf = sum(self.weights.values()) / (rrf_k + 1)
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: -x[1])

        results = []
        for doc_key, score in sorted_docs[:k]:
            norm_score = score / max_rrf if max_rrf > 0 else 0.0
            rrf_dist = (1.0 - norm_score) * self.state_dim
            results.append((payloads[doc_key], rrf_dist, {}))

        return results

    def recall_raw(self, query: str, k: int = 10):
        """Come recall() ma restituisce risultati separati per encoder.

        Returns:
            dict: {encoder_name: [(payload, dist, meta), ...]}
        """
        self._lazy_init()
        raw = {}
        for name in self.encoders:
            raw[name] = self._recall_encoder(name, query, k=k)
        return raw

    def reward(self, query: str, answer_text: str, correct: bool,
               lr: float = 0.15, k: int = 10):
        """Adatta i pesi RRF dal feedback: gli encoder che avevano
        rankato in alto la risposta premiata guadagnano peso (o lo
        perdono, se la risposta era sbagliata).

        Chiude il loop di apprendimento verso il layer geometrico:
        l'esperienza non aggiorna solo il grafo simbolico, ma anche
        come i tre spazi binari vengono fusi.
        """
        self._lazy_init()
        raw = self.recall_raw(query, k=k)
        sign = 1.0 if correct else -1.0
        for name, results in raw.items():
            rank = None
            for i, (payload, dist, meta) in enumerate(results, 1):
                text = (payload.get("text", str(payload))
                        if isinstance(payload, dict) else str(payload))
                if text == answer_text:
                    rank = i
                    break
            if rank is None:
                continue
            credit = 1.0 / rank
            self.weights[name] = max(
                0.1, self.weights[name] + lr * credit * sign)
        # Rinormalizza a media 1 (i pesi sono relativi)
        mean = sum(self.weights.values()) / len(self.weights)
        if mean > 0:
            self.weights = {n: w / mean for n, w in self.weights.items()}

    def info(self):
        """Stato degli encoder."""
        return {name: repr(enc) for name, enc in self.encoders.items()}
