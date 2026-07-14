"""
algebraic.py — Percorso di ragionamento algebrico (VSA) per il
ReasoningEngine: il paradigma adottato, non solo dimostrato.

Pipeline:
    testo KB ──TripleExtractor──▶ triple (s, rel, o)
                                     │ grounding: sketch MinHash
                                     ▼
                              WorkingMemory (XOR olografico)
    query ──QueryPlanner──▶ catena di relazioni
                                     │ hop = T ⊕ chiave → cleanup
                                     ▼
                              risposta + confidence calibrata

Divisione dei ruoli:
  - il simbolico sta AI CONFINI (estrazione triple all'ingresso,
    pattern della query, lookup di provenienza all'uscita);
  - il RAGIONAMENTO (concatenare i fatti attraverso le entità ponte)
    è XOR + cleanup nello spazio di Hamming: nessuna stop-word,
    nessun boost di keyword, nessun indice testuale nel loop.
"""

import re
from typing import List, Optional, Tuple

import numpy as np

from bsm.memory.vsa import WorkingMemory, hamming
from bsm.memory.encoder.entity_encoder import _minhash_sketch


# ---------------------------------------------------------------------------
# Estrazione triple (simbolico, all'ingresso)
# ---------------------------------------------------------------------------

# pattern → (relazione, inversa)
_FACT_PATTERNS = [
    (re.compile(r"^(.*?) (?:manufactures|develops|developed|creates|"
                r"created|provides|owns) (.+?)\.?$", re.I),
     "makes", "made_by"),
    (re.compile(r"^(.*?) was founded by (.+?)(?: in .+)?\.?$", re.I),
     "founded_by", "founder_of"),
    (re.compile(r"^(.*?) is headquartered in (.+?)\.?$", re.I),
     "hq", "hq_of"),
]


def _norm(entity: str) -> str:
    e = entity.strip().strip(".,;: ").lower()
    return e[4:] if e.startswith("the ") else e


def _tokens(name: str) -> set:
    """Token normalizzati per lo sketch di grounding (via punteggiatura)."""
    return {w.strip(".,;:()'\"") for w in name.lower().split()
            if w.strip(".,;:()'\"")}


class TripleExtractor:
    """Estrae triple (soggetto, relazione, oggetto) da frasi di KB
    con pattern SVO minimali.  Restituisce anche la frase sorgente
    (provenienza della risposta)."""

    def extract(self, text: str) -> List[Tuple[str, str, str, str]]:
        triples = []
        # split solo prima di una maiuscola: "Apple Inc. manufactures"
        # non è un confine di frase, "…France. It was…" sì
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip()):
            if not sentence:
                continue
            for pattern, rel, inverse in _FACT_PATTERNS:
                m = pattern.match(sentence.strip())
                if m:
                    s, o = _norm(m.group(1)), _norm(m.group(2))
                    if s and o:
                        triples.append((s, rel, o, sentence))
                        triples.append((o, inverse, s, sentence))
                    break
        return triples


# ---------------------------------------------------------------------------
# Query planner (simbolico, all'ingresso)
# ---------------------------------------------------------------------------

# pattern → catena di relazioni da percorrere partendo dall'ancora
_QUERY_PATTERNS = [
    (re.compile(r"who founded the company (?:that|which) "
                r"(?:makes|develops|created|owns|provides) (.+)\?", re.I),
     ["made_by", "founded_by"]),
    (re.compile(r"who founded the company based in (.+)\?", re.I),
     ["hq_of", "founded_by"]),
    (re.compile(r"where is the company based that was founded by (.+)\?",
                re.I),
     ["founder_of", "hq"]),
    (re.compile(r"where is the company based that "
                r"(?:makes|develops|provides) (.+)\?", re.I),
     ["made_by", "hq"]),
    (re.compile(r"what does the company founded by (.+) make\?", re.I),
     ["founder_of", "makes"]),
    (re.compile(r"what does the company based in (.+) make\?", re.I),
     ["hq_of", "makes"]),
    # single-hop
    (re.compile(r"who founded (.+)\?", re.I), ["founded_by"]),
    (re.compile(r"where is (.+) (?:headquartered|based)\?", re.I), ["hq"]),
    (re.compile(r"what does (.+) (?:make|develop|provide)\?", re.I),
     ["makes"]),
]


class QueryPlanner:
    """Domanda → (ancora, catena di relazioni), o None se il pattern
    non è riconosciuto (→ fallback alle euristiche)."""

    def plan(self, query: str) -> Optional[Tuple[str, List[str]]]:
        q = query.strip()
        for pattern, chain in _QUERY_PATTERNS:
            m = pattern.search(q)
            if m:
                return _norm(m.group(1)), list(chain)
        return None


# ---------------------------------------------------------------------------
# Reasoner algebrico
# ---------------------------------------------------------------------------

class AlgebraicReasoner:
    """Multi-hop come XOR: i fatti vivono in una WorkingMemory
    olografica, il grounding testo→entità passa dagli sketch MinHash.

    answer(query) → (payload, confidence, catena) oppure None se la
    query non è pianificabile o il cleanup non è affidabile.
    """

    # J stimato minimo perché un'ancora testuale agganci un'entità nota
    GROUND_MIN_JACCARD = 0.15

    def __init__(self, state_dim: int = 2048, sketch_dim: int = 256,
                 min_hop_confidence: float = 0.60):
        self.state_dim = state_dim
        self.sketch_dim = sketch_dim
        self.min_hop_confidence = min_hop_confidence
        self.wm = WorkingMemory(state_dim)
        self.extractor = TripleExtractor()
        self.planner = QueryPlanner()
        self._entities: List[str] = []           # nomi canonici
        self._sketches: List[np.ndarray] = []    # grounding MinHash
        self._provenance: dict = {}              # (s, rel) → frase sorgente
        self.n_triples = 0

    # -- ingest ---------------------------------------------------------

    def learn(self, texts: List[str]) -> int:
        """Estrae le triple dai testi e le scrive nell'algebra."""
        for text in texts:
            for s, rel, o, source in self.extractor.extract(text):
                self.wm.store(s, rel, o)
                self._provenance[(s, rel)] = source
                for name in (s, o):
                    if name not in self._entities:
                        self._entities.append(name)
                        self._sketches.append(
                            _minhash_sketch(_tokens(name),
                                            self.sketch_dim))
                self.n_triples += 1
        return self.n_triples

    # -- grounding (sketch MinHash, al confine) --------------------------

    def _ground(self, anchor: str) -> Optional[str]:
        """Testo libero → nome canonico via Hamming tra sketch."""
        if not self._entities:
            return None
        probe = _minhash_sketch(_tokens(anchor), self.sketch_dim)
        dists = [hamming(probe, s) for s in self._sketches]
        i = int(np.argmin(dists))
        est_j = 1.0 - 2.0 * dists[i] / self.sketch_dim
        return self._entities[i] if est_j >= self.GROUND_MIN_JACCARD else None

    # -- reasoning (XOR, il cuore) ---------------------------------------

    def answer(self, query: str) -> Optional[tuple]:
        """(payload, confidence, catena_descritta) o None."""
        from bsm.memory.reasoning_engine import calibrated_confidence

        plan = self.planner.plan(query)
        if plan is None:
            return None
        anchor, chain = plan
        node = self._ground(anchor)
        if node is None:
            return None

        confidence = 1.0
        steps = [node]
        for rel in chain:
            name, dist = self.wm.query(node, rel)      # T ⊕ chiave → cleanup
            hop_conf = calibrated_confidence(dist, self.state_dim)
            if hop_conf < self.min_hop_confidence:
                return None                             # cleanup nel rumore
            confidence *= hop_conf
            node = name
            steps.append(node)

        source = self._provenance.get((steps[-2], chain[-1]))
        payload = {"text": source or node, "entity": node,
                   "source": "algebraic"}
        chain_desc = "→".join(steps)
        return payload, confidence, chain_desc

    def __repr__(self):
        return (f"AlgebraicReasoner(D={self.state_dim}, "
                f"triples={self.n_triples}, "
                f"entities={len(self._entities)})")
