"""
graph_cache.py — Graph Cache vivente con Memory Confidence.

Ogni arco è un oggetto con ciclo di vita: la confidenza cresce con
l'esperienza (support/success), decade col tempo, e può essere
promossa (hot edge) o dimenticata (forget).

    Android ──[conf=0.92, sup=41]──→ Google ──→ Mountain View

Il Core (BSM) resta geometrico e deterministico.  Il GraphCache
aggiunge un layer di apprendimento dall'esperienza sopra.
"""

import json
import math
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Any


def _payload_text(p: Any) -> str:
    """Testo di un payload arbitrario (dict con 'text', dict, o altro)."""
    if isinstance(p, dict):
        return str(p.get("text", p))
    return str(p)


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

@dataclass
class Edge:
    """Arco tra due entità, con confidenza basata sull'esperienza.

    La confidenza è la media della posterior Beta(1+success, 1+failure):
        confidence = (1 + success) / (2 + support)

    Ogni uso (cache hit) incrementa support e, se il retrieval lo
    conferma, incrementa anche success.
    """
    source: str
    target: str
    answer_payload: Any
    bridge_chunk: str = ""

    confidence: float = 0.50
    support: int = 0
    success: int = 0
    failure: int = 0

    first_seen: float = 0.0
    last_seen: float = 0.0

    provenance: str = "retrieval"
    version: int = 1

    def __post_init__(self):
        if self.first_seen == 0.0:
            self.first_seen = time.time()
        if self.last_seen == 0.0:
            self.last_seen = time.time()

    def _rebuild_confidence(self):
        s = self.support
        if s == 0:
            self.confidence = 0.50
        else:
            self.confidence = (1.0 + self.success) / (2.0 + s)
        self.confidence = max(0.01, min(0.99, self.confidence))

    def record_hit(self, verified: bool = False):
        self.support += 1
        self.last_seen = time.time()
        if verified:
            self.success += 1
        else:
            self.failure += 1
        self._rebuild_confidence()

    def record_feedback(self, correct: bool):
        self.support += 1
        self.last_seen = time.time()
        if correct:
            self.success += 1
        else:
            self.failure += 1
        self._rebuild_confidence()

    def is_hot(self) -> bool:
        return self.support > 10 and self.confidence > 0.90

    def is_stale(self, now: float, max_age_days: float = 30.0) -> bool:
        age_days = (now - self.last_seen) / 86400
        return age_days > max_age_days

    def key(self) -> str:
        return f"{self.source.lower()}→{self.target.lower()}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["answer_payload"] = self._serialize_payload(self.answer_payload)
        return d

    @staticmethod
    def _serialize_payload(p: Any) -> Any:
        if isinstance(p, dict):
            return {k: Edge._serialize_payload(v) for k, v in p.items()}
        if isinstance(p, (list, tuple)):
            return [Edge._serialize_payload(v) for v in p]
        if hasattr(p, "__dict__"):
            return str(p)
        return p

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        return cls(**d)

    def __repr__(self):
        return (f"Edge({self.source}→{self.target}, "
                f"conf={self.confidence:.2f}, sup={self.support}, "
                f"provenance={self.provenance})")


# ---------------------------------------------------------------------------
# GraphCache
# ---------------------------------------------------------------------------

DECAY_LAMBDA = 0.01            # decay giornaliero: conf *= exp(-λ * Δt)
HOT_MIN_SUPPORT = 10
HOT_MIN_CONFIDENCE = 0.90
FORGET_CONFIDENCE = 0.15
FORGET_MAX_AGE_DAYS = 30.0
MERGE_MIN_OVERLAP = 0.50       # Jaccard minimo per fondere due target


class GraphCache:
    """Cache di percorsi di ragionamento con ciclo di vita.

    Struttura:
        self._graph[source_entity] = [Edge, ...]
        self._hot[source_entity] = [Edge, ...]     (hot edges, cache veloce)
        self._source_queries[query_lower] = entity
    """

    def __init__(self, path: Optional[str] = None,
                 encoder: Optional[Any] = None,
                 hamming_radius_frac: float = 0.30):
        """Args:
            path: file di persistenza (opzionale).
            encoder: encoder binario (tipicamente EntityEncoder MinHash).
                Se presente, lookup e merge diventano geometrici: entità
                simili ("Google" / "Google LLC" / "googel") condividono
                la stessa regione dello spazio di Hamming invece di
                essere stringhe distinte.
            hamming_radius_frac: raggio (frazione di state_dim) entro cui
                due entità sono considerate la stessa nel lookup.
        """
        self._path = Path(path) if path else None
        self._encoder = encoder
        self._radius_frac = hamming_radius_frac
        self._entity_states: dict = {}     # nome → stato int8 (memoizzato)
        self._graph: dict = {}
        self._hot: dict = {}
        self._source_queries: dict = {}
        self._metrics: dict = {
            "forgotten": 0,
            "merged": 0,
            "promoted": 0,
        }
        if self._path and self._path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Geometria delle entità (opzionale, richiede encoder)
    # ------------------------------------------------------------------

    def _entity_state(self, name: str):
        """Stato binario memoizzato di un nome di entità."""
        key = name.lower()
        if key not in self._entity_states:
            self._entity_states[key] = self._encoder.encode(key)
        return self._entity_states[key]

    def _hamming(self, a: str, b: str) -> int:
        sa, sb = self._entity_state(a), self._entity_state(b)
        return int(np.count_nonzero(sa != sb))

    def _nearest_entity(self, entity: str) -> Optional[str]:
        """Sorgente in _graph più vicina in spazio di Hamming, se entro
        il raggio.  Rende il lookup robusto ad alias e varianti."""
        if not self._encoder:
            return None
        radius = self._radius_frac * self._encoder.state_dim
        best, best_d = None, radius + 1
        for src in self._graph:
            if src.startswith("_"):
                continue
            d = self._hamming(entity, src)
            if d < best_d:
                best, best_d = src, d
        return best if best_d <= radius else None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup_query(self, query: str) -> Optional[List[Edge]]:
        q = query.lower().strip()
        entity = self._source_queries.get(q)
        if entity:
            return self.lookup_entity(entity)
        return None

    def lookup_entity(self, entity: str) -> Optional[List[Edge]]:
        ent = entity.lower()
        edges = self._hot.get(ent) or self._graph.get(ent)
        if not edges and self._encoder:
            # Lookup geometrico: alias/varianti ("Google LLC", typo)
            # cadono nella stessa palla di Hamming dell'entità nota
            near = self._nearest_entity(ent)
            if near:
                edges = self._hot.get(near) or self._graph.get(near)
        if not edges:
            return None
        edges.sort(key=lambda e: (-e.confidence, -e.support))
        return edges

    def get_bridge(self, entity: str) -> Optional[str]:
        edges = self.lookup_entity(entity)
        if not edges:
            return None
        return edges[0].target

    def get_answer(self, bridge_entity: str) -> Optional[Any]:
        for src, edges in self._graph.items():
            for e in edges:
                if e.target.lower() == bridge_entity.lower():
                    return e.answer_payload
        for src, edges in self._hot.items():
            for e in edges:
                if e.target.lower() == bridge_entity.lower():
                    return e.answer_payload
        return None

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(self, query: str, entity: str, bridge_entity: str,
              bridge_chunk: str, answer_payload: Any, score: float):
        ent = entity.lower()
        bridge = bridge_entity.lower()
        now = time.time()

        self._source_queries[query.lower().strip()] = ent

        existing = self._find_edge(ent, bridge)
        if existing:
            # Re-finding the same answer is evidence the path is reliable
            existing.support += 1
            existing.success += 1
            existing.last_seen = now
            existing.answer_payload = answer_payload
            existing.bridge_chunk = bridge_chunk
            existing._rebuild_confidence()
            self._maybe_promote(existing)
        else:
            edge = Edge(
                source=ent,
                target=bridge,
                answer_payload=answer_payload,
                bridge_chunk=bridge_chunk,
                confidence=min(score, 0.99),
                support=1,
                success=1 if score > 0.5 else 0,
                failure=0 if score > 0.5 else 1,
                first_seen=now,
                last_seen=now,
                provenance="retrieval",
                version=1,
            )
            if ent not in self._graph:
                self._graph[ent] = []
            self._graph[ent].append(edge)
            self._maybe_promote(edge)

        # Bridge → answer mapping
        bridge_key = f"_bridge_{bridge}"
        b_payload_text = _payload_text(answer_payload)
        b_edge = self._find_edge(bridge_key, b_payload_text)
        if b_edge:
            b_edge.support += 1
            b_edge.success += 1
            b_edge.last_seen = now
            b_edge._rebuild_confidence()
        else:
            b_edge = Edge(
                source=bridge_key,
                target=b_payload_text,
                answer_payload=answer_payload,
                bridge_chunk="",
                confidence=min(score, 0.99),
                support=1,
                success=1 if score > 0.5 else 0,
                failure=0 if score > 0.5 else 1,
                first_seen=now,
                last_seen=now,
                provenance="retrieval",
                version=1,
            )
            if bridge_key not in self._graph:
                self._graph[bridge_key] = []
            self._graph[bridge_key].append(b_edge)

    def record_cache_hit(self, entity: str, bridge_entity: str,
                         verified: bool = False):
        ent = entity.lower()
        bridge = bridge_entity.lower()
        edge = self._find_edge(ent, bridge)
        if edge:
            edge.record_hit(verified=verified)
            self._maybe_promote(edge)

    def _find_edge(self, source: str, target: str) -> Optional[Edge]:
        for edges in (self._hot.get(source), self._graph.get(source)):
            if not edges:
                continue
            for e in edges:
                if e.target.lower() == target.lower():
                    return e
        return None

    def _maybe_promote(self, edge: Edge):
        if not edge.is_hot():
            return
        src = edge.source.lower()
        if src in self._hot:
            for e in self._hot[src]:
                if e.key() == edge.key():
                    return
            self._hot[src].append(edge)
        else:
            self._hot[src] = [edge]
        self._metrics["promoted"] += 1

    # ------------------------------------------------------------------
    # Sleep – manutenzione della memoria
    # ------------------------------------------------------------------

    def sleep(self, decay: bool = True, merge: bool = True,
              forget: bool = True, now: Optional[float] = None):
        """Manutenzione completa della memoria.

        Args:
            decay: applica decadimento temporale alla confidenza.
            merge: fonde archi con target simili.
            forget: rimuove archi con confidenza o supporto bassi.

        Returns:
            dict con le metriche dell'operazione.
        """
        t = now or time.time()
        result = {
            "decayed": 0,
            "merged": 0,
            "forgotten": 0,
            "promoted": 0,
            "hot_edges": 0,
            "total_edges": 0,
        }

        all_keys = list(self._graph.keys())
        for key in all_keys:
            if key.startswith("_"):
                continue
            edges = self._graph.get(key, [])
            if not edges:
                continue

            if decay:
                self._apply_decay(edges, t, result)

            if merge and len(edges) > 1:
                edges = self._apply_merge(edges, t, result)

            if forget:
                edges = self._apply_forget(edges, t, result)

            if edges:
                self._graph[key] = edges
                for e in edges:
                    if e.is_hot():
                        self._maybe_promote(e)
            else:
                del self._graph[key]

        # Rebuild the hot cache strictly from surviving graph edges, so
        # edges dropped by merge/forget can't linger in _hot
        new_hot = {}
        for src, edges in self._graph.items():
            if src.startswith("_"):
                continue
            fresh = [e for e in edges if e.is_hot()]
            if fresh:
                new_hot[src] = fresh
        self._hot = new_hot

        result["hot_edges"] = sum(len(v) for v in self._hot.values())
        result["total_edges"] = sum(len(v) for v in self._graph.values()
                                     if not v[0].source.startswith("_")) if self._graph else 0

        self._metrics["forgotten"] += result["forgotten"]
        self._metrics["merged"] += result["merged"]
        self._metrics["promoted"] += result["promoted"]
        return result

    def _apply_decay(self, edges: list, now: float, result: dict):
        for e in edges:
            delta_days = (now - e.last_seen) / 86400
            if delta_days > 0:
                decay = math.exp(-DECAY_LAMBDA * delta_days)
                old_conf = e.confidence
                e.confidence = max(0.01, e.confidence * decay)
                if abs(e.confidence - old_conf) > 0.001:
                    result["decayed"] += 1

    def _apply_merge(self, edges: list, now: float, result: dict) -> list:
        """Merge edges with similar targets (by entity overlap)."""
        merged = []
        used = set()
        for i, a in enumerate(edges):
            if i in used:
                continue
            a_target = self._target_entities(a.target)
            group = [a]
            used.add(i)
            for j, b in enumerate(edges[i + 1:], i + 1):
                if j in used:
                    continue
                if self._encoder:
                    # Jaccard stimato dalla distanza di Hamming tra gli
                    # sketch MinHash dei target: J ≈ 1 - 2*dist/D.
                    # Lo stimatore ha std ≈ 1/sqrt(D): la si aggiunge
                    # come tolleranza per non mancare i merge al limite.
                    d = self._hamming(a.target, b.target)
                    D = self._encoder.state_dim
                    overlap = max(0.0, 1.0 - 2.0 * d / D) + 2.0 / math.sqrt(D)
                else:
                    b_target = self._target_entities(b.target)
                    overlap = self._jaccard(a_target, b_target)
                if overlap >= MERGE_MIN_OVERLAP:
                    group.append(b)
                    used.add(j)
            if len(group) > 1:
                group.sort(key=lambda e: -e.confidence)
                best = group[0]
                best.support = sum(e.support for e in group)
                best.success = sum(e.success for e in group)
                best.failure = sum(e.failure for e in group)
                best.last_seen = max(e.last_seen for e in group)
                best._rebuild_confidence()
                best.version += 1
                merged.append(best)
                result["merged"] += len(group) - 1
            else:
                merged.append(a)
        return merged

    def _apply_forget(self, edges: list, now: float, result: dict) -> list:
        surviving = []
        for e in edges:
            if e.confidence < FORGET_CONFIDENCE:
                result["forgotten"] += 1
                continue
            if e.support < 3 and (now - e.last_seen) / 86400 > FORGET_MAX_AGE_DAYS:
                result["forgotten"] += 1
                continue
            surviving.append(e)
        return surviving

    @staticmethod
    def _target_entities(target: str) -> set:
        words = target.replace(".", " ").split()
        return {w.lower().strip(".,;:!?()'\"") for w in words if len(w) > 1}

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def metrics(self) -> dict:
        edges = []
        for key, lst in self._graph.items():
            if key.startswith("_"):
                continue
            edges.extend(lst)

        if not edges:
            return {
                "entities": 0,
                "edges": 0,
                "hot_edges": 0,
                "avg_confidence": 0.0,
                "avg_support": 0,
                "forgotten_total": self._metrics["forgotten"],
                "merged_total": self._metrics["merged"],
                "promoted_total": self._metrics["promoted"],
            }

        return {
            "entities": self.size(),
            "edges": len(edges),
            "hot_edges": sum(len(v) for v in self._hot.values()),
            "avg_confidence": round(sum(e.confidence for e in edges) / len(edges), 4),
            "avg_support": round(sum(e.support for e in edges) / len(edges), 1),
            "forgotten_total": self._metrics["forgotten"],
            "merged_total": self._metrics["merged"],
            "promoted_total": self._metrics["promoted"],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None):
        p = Path(path) if path else self._path
        if not p:
            return
        data = {
            "graph": {
                k: [e.to_dict() for e in v]
                for k, v in self._graph.items()
            },
            "source_queries": self._source_queries,
            "metrics": self._metrics,
        }
        p.write_text(json.dumps(data, indent=2, default=str))

    def _load(self):
        data = json.loads(self._path.read_text())
        self._graph = {}
        for k, v in data.get("graph", {}).items():
            self._graph[k] = [Edge.from_dict(item) for item in v]
        self._source_queries = data.get("source_queries", {})
        self._metrics = data.get("metrics", {"forgotten": 0, "merged": 0, "promoted": 0})
        # Rebuild hot edges
        for key, edges in self._graph.items():
            if key.startswith("_"):
                continue
            for e in edges:
                if e.is_hot():
                    self._maybe_promote(e)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def size(self) -> int:
        return len([k for k in self._graph if not k.startswith("_")])

    def total_paths(self) -> int:
        return sum(len(v) for v in self._graph.values())

    def clear(self):
        self._graph.clear()
        self._hot.clear()
        self._source_queries.clear()
        self._metrics = {"forgotten": 0, "merged": 0, "promoted": 0}

    def __repr__(self):
        m = self.metrics()
        hot = m["hot_edges"]
        return (f"GraphCache(entities={m['entities']}, "
                f"edges={m['edges']}, avg_c={m['avg_confidence']:.2f}, "
                f"hot={hot})")
