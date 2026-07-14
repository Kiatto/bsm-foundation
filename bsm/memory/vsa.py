"""
vsa.py — Algebra binaria per il ragionamento in spazio di Hamming.

Due algebre di binding, una per sottosistema di memoria:

  WORKING MEMORY — binding XOR (invertibile, esatto)
      In {-1,+1} il prodotto elementwise È lo XOR: auto-inverso,
      distanze preservate.  I fatti (s, r, o) diventano
          fatto = chiave ⊕ oggetto,   chiave = s ⊕ ρ(r)
      e l'intera memoria è UNA traccia olografica T = maj(fatti):
      la query è algebra, T ⊕ chiave ≈ oggetto + rumore, ripulito
      dalla item memory.  Manipolazione esatta, capacità limitata
      dal rumore di sovrapposizione.

  SEMANTIC MEMORY — binding proiettivo lossy (astrattivo)
      bind_role(x) = sign(P_role · x) con P_role a rango ridotto
      FITTATA sul corpus (SVD): proiettando si scarta il dettaglio
      idiosincratico e si tiene la struttura condivisa.  Varianti
      della stessa entità collassano sullo stesso sottospazio: la
      generalizzazione non è un processo a valle, è codificata
      nell'operatore di scrittura.  Non invertibile: si decodifica
      per nearest-fact (verifica in avanti).

  bundle(x1..xn) = majority vote bit a bit (sovrapposizione olografica)

Tesi (docs/vsa_report.md): l'operatore di binding è una proprietà del
sottosistema di memoria, non dell'algebra — il consolidamento
Working→Semantic è un cambio di operatore.
"""

import zlib
import numpy as np
from typing import List, Optional

# ---------------------------------------------------------------------------
# Primitive dell'algebra
# ---------------------------------------------------------------------------


def random_hv(name: str, state_dim: int) -> np.ndarray:
    """Hypervector deterministico in {-1,+1} dal nome (item memory)."""
    seed = zlib.crc32(name.lower().encode()) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    return np.where(rng.rand(state_dim) > 0.5, 1, -1).astype(np.int8)


def bind_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Binding XOR: in {-1,+1} è il prodotto elementwise. Auto-inverso."""
    return (a.astype(np.int16) * b.astype(np.int16)).astype(np.int8)


def permute(x: np.ndarray, k: int = 1) -> np.ndarray:
    """Permutazione ciclica: marca il ruolo/posizione (ρ di Kanerva)."""
    return np.roll(x, k)


def bundle(states: List[np.ndarray]) -> np.ndarray:
    """Sovrapposizione olografica: majority vote bit a bit.

    Con numero pari di stati i pareggi si rompono con un hypervector
    di rumore fisso (deterministico, nessun bias sistematico).
    """
    total = np.sum(np.stack(states).astype(np.int32), axis=0)
    if len(states) % 2 == 0:
        tie = random_hv("__tie_breaker__", states[0].shape[0])
        total = total * 2 + tie
    return np.where(total >= 0, 1, -1).astype(np.int8)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


# ---------------------------------------------------------------------------
# Item memory / cleanup
# ---------------------------------------------------------------------------

class ItemMemory:
    """Codebook nome ↔ hypervector con cleanup (nearest in Hamming)."""

    def __init__(self, state_dim: int):
        self.state_dim = state_dim
        self._names: List[str] = []
        self._states: List[np.ndarray] = []
        self._index: dict = {}

    def add(self, name: str, state: Optional[np.ndarray] = None) -> np.ndarray:
        if name not in self._index:
            self._index[name] = len(self._names)
            self._names.append(name)
            self._states.append(state if state is not None
                                else random_hv(name, self.state_dim))
        return self._states[self._index[name]]

    def get(self, name: str) -> np.ndarray:
        return self._states[self._index[name]]

    def cleanup(self, noisy: np.ndarray) -> tuple:
        """(nome, stato, dist) dell'item più vicino allo stato rumoroso."""
        dists = [hamming(noisy, s) for s in self._states]
        i = int(np.argmin(dists))
        return self._names[i], self._states[i], dists[i]

    def names(self) -> List[str]:
        return list(self._names)


# ---------------------------------------------------------------------------
# Proiezione di ruolo (binding lossy)
# ---------------------------------------------------------------------------

class RoleProjection:
    """Operatore di binding proiettivo per un ruolo.

    Default: matrice casuale deterministica (seed dal nome del ruolo) —
    binding non invertibile ma full-rank.  Con fit(corpus, rank)
    diventa lossy e astrattiva: tiene i top-`rank` componenti SVD del
    corpus (struttura condivisa), scarta il resto (dettaglio), poi
    ruota nello spazio del ruolo.
    """

    def __init__(self, role: str, state_dim: int):
        self.role = role
        self.state_dim = state_dim
        self.rank = state_dim
        seed = zlib.crc32(f"role:{role}".encode()) & 0xFFFFFFFF
        rng = np.random.RandomState(seed)
        self._P = rng.randn(state_dim, state_dim).astype(np.float32)
        self._fitted = False

    def fit(self, states: List[np.ndarray], rank: int) -> "RoleProjection":
        X = np.stack(states).astype(np.float32)
        X -= X.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        r = min(rank, Vt.shape[0])
        basis = Vt[:r]                                   # (r, D)
        seed = zlib.crc32(f"rot:{self.role}".encode()) & 0xFFFFFFFF
        rng = np.random.RandomState(seed)
        rot = rng.randn(self.state_dim, r).astype(np.float32)
        rot /= np.linalg.norm(rot, axis=0, keepdims=True) + 1e-8
        self._P = rot @ basis                            # (D, D), rango r
        self.rank = r
        self._fitted = True
        return self

    def apply(self, x: np.ndarray) -> np.ndarray:
        """bind_role(x) = sign(P·x): binario → binario, lossy."""
        raw = self._P @ x.astype(np.float32)
        return np.where(raw >= 0, 1, -1).astype(np.int8)

    def __repr__(self):
        kind = f"fitted(rank={self.rank})" if self._fitted else "random"
        return f"RoleProjection({self.role}, {kind})"


# ---------------------------------------------------------------------------
# Working Memory — algebra XOR, traccia olografica
# ---------------------------------------------------------------------------

class WorkingMemory:
    """Memoria di lavoro: fatti (s, r, o) in una traccia olografica.

        chiave(s, r) = item(s) ⊕ ρ(item(r))
        fatto        = chiave ⊕ item(o)
        T            = maj(fatti)                (UN vettore per tutto)
        query(s, r)  = cleanup( T ⊕ chiave(s,r) )

    Il ragionamento è XOR: nessuna euristica testuale, nessun indice.
    """

    def __init__(self, state_dim: int, items: Optional[ItemMemory] = None):
        self.state_dim = state_dim
        self.items = items or ItemMemory(state_dim)
        self._facts: List[np.ndarray] = []
        self._trace: Optional[np.ndarray] = None

    def _key(self, s: str, r: str) -> np.ndarray:
        return bind_xor(self.items.add(s), permute(self.items.add(r), 1))

    def store(self, s: str, r: str, o: str):
        self._facts.append(bind_xor(self._key(s, r), self.items.add(o)))
        self._trace = bundle(self._facts)

    def query(self, s: str, r: str) -> tuple:
        """(nome_oggetto, dist_cleanup) dalla SOLA traccia olografica."""
        noisy_o = bind_xor(self._trace, self._key(s, r))
        name, _, d = self.items.cleanup(noisy_o)
        return name, d

    def query_exact(self, s: str, r: str) -> tuple:
        """Decodifica dal fatto singolo più vicino (non olografica)."""
        key = self._key(s, r)
        # il fatto giusto è quello per cui fatto ⊕ chiave è più vicino
        # a un item noto (minima distanza di cleanup)
        best = (None, self.state_dim + 1)
        for f in self._facts:
            name, _, d = self.items.cleanup(bind_xor(f, key))
            if d < best[1]:
                best = (name, d)
        return best

    def n_facts(self) -> int:
        return len(self._facts)


# ---------------------------------------------------------------------------
# Semantic Memory — addressing con binding configurabile
# ---------------------------------------------------------------------------

class SemanticMemory:
    """Memoria semantica: fatti indirizzati da una chiave il cui
    operatore di binding è configurabile — il cuore dell'esperimento.

        binding='xor'   chiave = ruolo_s ⊕ sketch(s)  ⊕-bundled con rel
                        (invertibile, preserva le distanze grezze)
        binding='proj'  chiave = P_subj(sketch(s)) bundled con rel
                        (lossy: se fittata, le varianti collassano)

    store/query lavorano su STATI di entità (es. sketch MinHash del
    nome), quindi la memoria è agnostica rispetto alla lingua.
    query = nearest-fact sulla chiave → payload oggetto del fatto.
    """

    def __init__(self, state_dim: int, binding: str = "proj"):
        assert binding in ("xor", "proj")
        self.state_dim = state_dim
        self.binding = binding
        self._proj_s = RoleProjection("subj", state_dim)
        self._role_s = random_hv("__role_subj__", state_dim)
        self._facts: List[tuple] = []    # (key_trace, s_name, rel, o_name)

    def fit(self, entity_states: List[np.ndarray], rank: int):
        """Fitta la proiezione di soggetto sul corpus (solo 'proj')."""
        self._proj_s.fit(entity_states, rank)
        return self

    def _key(self, s_state: np.ndarray) -> np.ndarray:
        """Chiave geometrica del soggetto.  La relazione resta un
        simbolo esatto (filtrata in query): la geometria decide SOLO
        la similarità tra entità — è l'operatore sotto esame."""
        if self.binding == "xor":
            return bind_xor(self._role_s, s_state)
        return self._proj_s.apply(s_state)

    def store(self, s_state: np.ndarray, rel: str, o_name: str,
              s_name: str = ""):
        self._facts.append((self._key(s_state), s_name, rel, o_name))

    def query(self, s_state: np.ndarray, rel: str) -> tuple:
        """(nome_oggetto, dist, margine).  Nearest-fact sulla chiave tra
        i fatti della relazione; il margine (secondo - primo) misura la
        robustezza dell'addressing."""
        probe = self._key(s_state)
        dists = sorted(((hamming(probe, f[0]), f) for f in self._facts
                        if f[2] == rel), key=lambda x: x[0])
        best_d, best_f = dists[0]
        margin = (dists[1][0] - best_d) if len(dists) > 1 else best_d
        return best_f[3], best_d, margin

    def n_facts(self) -> int:
        return len(self._facts)
