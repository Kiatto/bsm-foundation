"""
reasoning_engine.py — Phase III: reasoning over geometric memory.

Multi-hop reasoning via query decomposition: extract the target entity,
retrieve the bridge chunk, extract the bridge entity, retrieve the answer.

Usage:
    engine = ReasoningEngine(bsm)
    result = engine.reason(query, max_hops=6)
    print(result.answer, result.confidence)
"""

import time
import math
import numpy as np
from typing import List, Optional, Any
from dataclasses import dataclass, field

from bsm.memory.ensemble import EnsembleRetriever
from bsm.memory.graph_cache import GraphCache
from bsm.memory.experience import Experience, ExperienceBuffer


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReasoningHop:
    """A single step in the reasoning chain."""
    hop: int
    query_state: np.ndarray       # state used for retrieval
    retrieved: List[tuple]        # (payload, dist, meta) from BSM
    centroid_state: np.ndarray    # fused centroid (or None if not fused)
    confidence: float             # 0..1
    entropy: float                # estimated uncertainty
    score: float                  # overall score for this hop

    def __repr__(self):
        return (f"Hop({self.hop}, conf={self.confidence:.3f}, "
                f"entropy={self.entropy:.3f}, "
                f"retrieved={len(self.retrieved)})")


@dataclass
class ReasoningResult:
    """The output of a complete reasoning run."""
    answer: Any                   # final answer (payload text)
    confidence: float             # 0..1
    hops: int                     # number of hops executed
    max_hops: int                 # configured max
    convergence_reason: str       # why we stopped
    graph: List[ReasoningHop]     # full reasoning chain
    visited_nodes: int            # unique nodes visited
    elapsed_ms: float             # total runtime
    experience: Optional['Experience'] = None  # episodic memory record

    def summary(self) -> dict:
        return {
            "answer": str(self.answer)[:80],
            "confidence": self.confidence,
            "hops": self.hops,
            "convergence": self.convergence_reason,
            "visited": self.visited_nodes,
            "ms": self.elapsed_ms,
        }

    def __repr__(self):
        return (f"ReasoningResult(answer={str(self.answer)[:50]}, "
                f"conf={self.confidence:.2f}, hops={self.hops}, "
                f"reason={self.convergence_reason})")


# ---------------------------------------------------------------------------
# Reasoning Engine
# ---------------------------------------------------------------------------

def _payload_text(payload: Any) -> str:
    """Testo di un payload BSM arbitrario (dict con 'text', dict, o altro)."""
    if isinstance(payload, dict):
        return str(payload.get("text", payload))
    return str(payload)


def calibrated_confidence(dist: float, state_dim: int,
                          temperature: float = 8.0) -> float:
    """Confidence calibrata sulla distribuzione nulla di Hamming.

    Due stati binari indipendenti a D bit distano Binomial(D, 0.5),
    cioè ~N(D/2, sqrt(D)/2).  Lo z-score misura quante deviazioni
    standard il match è sotto il caso; la logistica lo mappa in (0,1)
    senza saturare (a differenza della CDF normale, che schiaccerebbe
    tutti i match forti a 1.0 distruggendo il ranking).

        dist = D/2  → 0.5   (indistinguibile dal rumore)
        dist = D/4  → z=8   → ~0.73
        dist = 0    → z=16  → ~0.88

    A differenza di `1 - dist/D` (che vive quasi tutta in [0.4, 0.6]),
    questa confidence è interpretabile: >0.5 significa "meglio del caso".

    temperature=8 è scelta perché a livello di chance la pendenza
    locale (1/(4·τ) per z, cioè 1/(2·τ·sqrt(D)) per bit) coincide con
    quella della scala legacy 1-dist/D a D=256: stesso potere
    discriminante nel ranking, valore assoluto interpretabile.
    """
    mean = state_dim / 2.0
    std = math.sqrt(state_dim) / 2.0
    z = (mean - dist) / std
    return 1.0 / (1.0 + math.exp(-z / temperature))


class ReasoningEngine:
    """Multi-hop reasoning over BSM geometric memory.

    Query decomposition in 2 hop:
        1. Extract the query's target entity
        2. Retrieve the bridge chunk with the entity as sub-query
        3. Extract the bridge entity from the bridge chunk
        4. Retrieve the answer chunk via the bridge entity

    Bridge entity extraction enables true multi-hop: hop 0 finds the
    connecting fact, hop 1 finds the target fact through the entity.

    Returns a ReasoningResult with the full hop graph.
    """

    def __init__(self,
                 bsm: "BSM",
                 ensemble: Optional[EnsembleRetriever] = None,
                 graph_cache: Optional[GraphCache] = None,
                 experience_buffer: Optional[ExperienceBuffer] = None,
                 beam_width: int = 4,
                 algebraic: Optional[Any] = None):
        """algebraic: AlgebraicReasoner opzionale — se presente è il
        PRIMO percorso di risoluzione (multi-hop come XOR); le
        euristiche testuali diventano il fallback."""
        if bsm is None:
            raise ValueError("ReasoningEngine requires a BSM instance")
        self.bsm = bsm
        self.ensemble = ensemble
        self.graph_cache = graph_cache
        self.experience_buffer = experience_buffer or ExperienceBuffer()
        self.beam_width = beam_width
        self.algebraic = algebraic

    # ------------------------------------------------------------------
    # Retrieval (ensemble-aware)
    # ------------------------------------------------------------------

    def _recall(self, text: str, k: int) -> List[tuple]:
        """Recupera da BSM singolo (ProjectionEncoder)."""
        state = self.bsm.encode(text)
        return self.bsm.recall(state, k=k)

    def _recall_ensemble(self, text: str, k: int) -> List[tuple]:
        """Recupera dall'ensemble (fallback)."""
        if self.ensemble:
            return self.ensemble.recall(text, k=k)
        return self._recall(text, k=k)

    # ------------------------------------------------------------------
    # Keyword extraction
    # ------------------------------------------------------------------

    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "with", "by", "and", "or", "not", "it", "its",
        "as", "from", "this", "that", "be", "has", "have", "had", "do",
        "does", "did", "will", "would", "could", "should", "may", "can",
        "about", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "over", "out", "off", "than", "then",
        "also", "very", "just", "because", "but", "which", "what", "who",
        "where", "when", "why", "how", "all", "each", "every", "both",
        "few", "more", "most", "some", "any", "no", "up", "down",
    }

    def _query_entity(self, query: str) -> str:
        """Extract the target entity from the query.

        Priority:
        1. Last uppercase-starting word (after removing question words)
           or any word with internal capitals (camelCase like iPhone)
        2. Last 2 non-stop, non-question words

        This is the object being asked about (e.g. 'iPhone' from
        'Who founded the company that makes the iPhone?').
        """
        words = query.split()
        cleaned = [w.strip(".,;:!?()'") for w in words]

        # Uppercase-starting or camelCase words, excluding question words
        q_words = {"who", "what", "where", "when", "why", "how", "which"}
        capitals = []
        for i, wc in enumerate(cleaned):
            if not wc:
                continue
            has_upper = (wc[0].isupper() or any(c.isupper() for c in wc[1:]))
            if (has_upper and len(wc) > 1
                    and wc.lower() not in self.STOP_WORDS
                    and wc.lower() not in q_words
                    and wc.isalpha()):
                capitals.append((i, wc))
        if capitals:
            last_idx = capitals[-1][0]
            entity = capitals[-1][1]
            for i, wc in reversed(capitals[:-1]):
                if i == last_idx - 1:
                    entity = f"{wc} {entity}"
                    break
            return entity

        # No capitals: take last 2 non-stop words (using !isalpha for punctuated words)
        content = [(i, wc) for i, wc in enumerate(cleaned)
                    if (wc.lower() not in self.STOP_WORDS
                        and wc.lower() not in q_words
                        and len(wc) > 2
                        and wc.isalpha())]
        if len(content) >= 2:
            return f"{content[-2][1]} {content[-1][1]}"
        if content:
            return content[-1][1]
        return ""

    def _word_overlap(self, text: str, query: str) -> float:
        """Fraction of query content words that appear in *text*."""
        q_words = {w.lower().strip(".,;:!?()")
                   for w in query.split()
                   if w.lower() not in self.STOP_WORDS and len(w) > 2}
        if not q_words:
            return 0.0
        t_words = {w.lower().strip(".,;:!?()")
                   for w in text.split()}
        matches = q_words & t_words
        return len(matches) / len(q_words)

    def _bridge_entity(self, chunk_text: str, query: str) -> str:
        """Extract the bridge entity from *chunk_text*.

        Uses the FIRST proper noun in the chunk (typically the company/
        organization name) that is not in the query or stop words.
        Falls back to the longest new word if no proper noun found.
        """
        query_words = set(w.lower() for w in query.split())
        words = chunk_text.split()
        candidates = []
        for i, w in enumerate(words):
            wc = w.strip(".,;:!?()[]'\";:")
            if (len(wc) > 1
                    and wc.lower() not in query_words
                    and wc.lower() not in self.STOP_WORDS
                    and all(c.isalpha() or c == '.' for c in wc)):
                is_proper = wc[0].isupper()
                candidates.append((wc, is_proper, i, len(wc)))
        if not candidates:
            return ""
        # Prefer proper nouns (sorted by position), then longest word
        proper = [c for c in candidates if c[1]]
        if proper:
            proper.sort(key=lambda x: x[2])  # sort by position
            return proper[0][0]
        # No proper nouns: longest word
        candidates.sort(key=lambda x: -x[3])
        return candidates[0][0]

    def reason(self,
               query: str,
               max_hops: int = 8) -> ReasoningResult:
        """Multi-hop reasoning via query decomposition.

        Architecture:
        1. Extract the query's target entity (last capitalized noun phrase)
        2. Retrieve with the entity as a sub-query → find the bridge chunk
        3. Extract the bridge entity (first proper noun) from the bridge chunk
        4. Build an answer query: '<question_word> <bridge_entity>'
        5. Retrieve with the answer query → find the answer chunk

        This avoids the accidental-alignment problem: the sub-query is
        focused on the target entity, not the full question text.
        """
        t0 = time.perf_counter()

        graph: List[ReasoningHop] = []

        # ── Step 0: percorso algebrico (VSA) ──
        # Il multi-hop come XOR: se il planner riconosce la query e il
        # cleanup è sopra soglia, la risposta arriva dall'algebra senza
        # toccare le euristiche testuali.
        if self.algebraic is not None:
            alg = self.algebraic.answer(query)
            if alg is not None:
                payload, conf, chain_desc = alg
                elapsed = (time.perf_counter() - t0) * 1000
                conv = f"algebraic:{chain_desc}"
                hops = max(1, chain_desc.count("→"))
                exp = Experience(
                    query=query, query_entity=chain_desc.split("→")[0],
                    bridge_entity=(chain_desc.split("→")[1]
                                   if chain_desc.count("→") > 1 else ""),
                    answer_payload=payload, confidence=float(conf),
                    hops=hops, latency_ms=elapsed, convergence=conv,
                )
                self.experience_buffer.add(exp)
                return ReasoningResult(
                    answer=payload, confidence=float(conf),
                    hops=hops, max_hops=max_hops,
                    convergence_reason=conv,
                    graph=[], visited_nodes=hops + 1,
                    elapsed_ms=elapsed, experience=exp,
                )

        # ── Step 1: extract target entity from query ──
        entity = self._query_entity(query)
        if not entity:
            # Fallback: direct retrieval with full query
            results = self._recall(query, k=1)
            payload = results[0][0] if results else None
            conf = (0.0 if not results
                    else calibrated_confidence(results[0][1],
                                               self.bsm.state_dim))
            elapsed = (time.perf_counter() - t0) * 1000
            exp = Experience(
                query=query, query_entity=entity,
                answer_payload=payload, confidence=conf,
                hops=1, latency_ms=elapsed, convergence="no_entity",
            )
            self.experience_buffer.add(exp)
            return ReasoningResult(
                answer=payload, confidence=conf,
                hops=1, max_hops=max_hops,
                convergence_reason="no_entity",
                graph=[], visited_nodes=0,
                elapsed_ms=elapsed,
                experience=exp,
            )

        # ── Step 1b: check GraphCache for cached path ──
        if self.graph_cache:
            cached = self.graph_cache.lookup_entity(entity)
            if cached:
                best = cached[0]
                elapsed = (time.perf_counter() - t0) * 1000
                conv = f"graph_cache:{entity}→{best.target}"
                exp = Experience(
                    query=query, query_entity=entity,
                    bridge_entity=best.target, bridge_chunk=best.bridge_chunk,
                    answer_payload=best.answer_payload,
                    confidence=float(best.confidence),
                    hops=0, latency_ms=elapsed, convergence=conv,
                    edge_keys=[best.key()],
                )
                self.experience_buffer.add(exp)
                return ReasoningResult(
                    answer=best.answer_payload, confidence=float(best.confidence),
                    hops=0, max_hops=max_hops,
                    convergence_reason=conv,
                    graph=[], visited_nodes=0,
                    elapsed_ms=elapsed,
                    experience=exp,
                )

        # ── Step 2: retrieve bridge candidates ──
        # Primary: entity sub-query with strict entity-word overlap filter.
        # Uses the single ProjectionEncoder (most precise for entity queries).
        # Fallback: full query via ensemble when entity sub-query yields
        # zero valid bridges (e.g. "Seattle" → "Amazon founded" lacks "seattle").
        # The ensemble (if available) can help recover from projection blind spots.
        bridge_results = self._recall(entity, k=self.beam_width * 3)

        using_fallback = False
        if bridge_results:
            valid = [c for c in bridge_results
                     if self._word_overlap(_payload_text(c[0]), entity) > 0.0]
            if len(valid) == 0:
                using_fallback = True
                bridge_results = self._recall_ensemble(query,
                                                        k=self.beam_width * 3)

        if not bridge_results:
            elapsed = (time.perf_counter() - t0) * 1000
            exp = Experience(
                query=query, query_entity=entity,
                answer_payload=None, confidence=0.0,
                hops=1, latency_ms=elapsed, convergence="no_bridge",
            )
            self.experience_buffer.add(exp)
            return ReasoningResult(
                answer=None, confidence=0.0, hops=1, max_hops=max_hops,
                convergence_reason="no_bridge",
                graph=[], visited_nodes=0,
                elapsed_ms=elapsed,
                experience=exp,
            )

        hop0_state = (self.bsm.encode(entity)
                      if not using_fallback
                      else self.bsm.encode(query))
        hop0 = ReasoningHop(
            hop=0, query_state=hop0_state,
            retrieved=bridge_results[:self.beam_width],
            centroid_state=hop0_state,
            confidence=0.0, entropy=0.0, score=0.0,
        )
        graph.append(hop0)

        # ── Step 3: beam search over bridge entities ──
        q_word = query.split()[0].lower()

        all_paths = []  # [(answer_payload, score, bridge_entity, bridge_chunk)]
        seen = set()

        for b_payload, b_dist, _ in bridge_results:
            b_text = _payload_text(b_payload)
            bridge_ent = self._bridge_entity(b_text, query)
            if not bridge_ent or bridge_ent.lower() in seen:
                continue
            # Skip bridges with zero word overlap to entity
            # (in fallback mode, the entity word may not appear in the chunk)
            if not using_fallback and self._word_overlap(b_text, entity) == 0.0:
                continue
            seen.add(bridge_ent.lower())

            # Answer queries: entity alone + question+entity
            ans_queries = [bridge_ent, f"{q_word} {bridge_ent}"]

            best_dist = {}
            for aq in ans_queries:
                a_results = self._recall(aq, k=self.beam_width)
                for a_payload, a_dist, _ in a_results:
                    a_text = _payload_text(a_payload)
                    # Skip the bridge chunk in normal mode (the bridge is a
                    # connector, not the answer). In fallback mode, don't skip
                    # — the bridge may itself contain the answer.
                    if a_text == b_text and not using_fallback:
                        continue
                    key = a_text
                    if key not in best_dist or a_dist < best_dist[key][0]:
                        best_dist[key] = (a_dist, a_payload)

            for a_text, (a_dist, a_payload) in best_dist.items():
                # Confidence Propagation: P(bridge|query) × P(answer|bridge),
                # ogni fattore calibrato sulla distribuzione nulla di Hamming
                state_dim = self.bsm.state_dim
                p_bridge = calibrated_confidence(b_dist, state_dim)
                p_answer = calibrated_confidence(a_dist, state_dim)
                score = min(p_bridge * p_answer, 1.0)  # compound probability
                # Boost answer chunks matching question intent
                a_lower = a_text.lower()
                if q_word == "who" and "founded" in a_lower:
                    score += 0.07
                elif q_word == "where" and ("headquartered" in a_lower
                                            or "located" in a_lower):
                    score += 0.05
                elif q_word == "what" and any(w in a_lower for w in
                         ("manufactures", "develops", "created",
                          "provides", "owns")):
                    score += 0.05
                score = min(score, 1.0)
                all_paths.append((a_payload, score, bridge_ent, b_text))

        if not all_paths:
            # No valid path: fallback to best bridge chunk
            s = calibrated_confidence(bridge_results[0][1], self.bsm.state_dim)
            best = (bridge_results[0][0], s, "", "")
        else:
            all_paths.sort(key=lambda x: -x[1])
            best = all_paths[0]

        answer_payload, answer_score, used_bridge, used_chunk = best

        hop1_state = self.bsm.encode(used_bridge or entity)
        hop1 = ReasoningHop(
            hop=1, query_state=hop1_state,
            retrieved=[(answer_payload, 0.0, {})],
            centroid_state=hop1_state,
            confidence=answer_score, entropy=0.0, score=0.0,
        )
        graph.append(hop1)

        # ── Graph Cache: store the discovered path ──
        if self.graph_cache and used_bridge and answer_payload:
            self.graph_cache.store(
                query=query,
                entity=entity,
                bridge_entity=used_bridge,
                bridge_chunk=used_chunk,
                answer_payload=answer_payload,
                score=answer_score,
            )

        elapsed = (time.perf_counter() - t0) * 1000
        conv = f"q_decomp:{entity}→{used_bridge}" if used_bridge else "bridge_fallback"
        edge_keys = []
        if used_bridge and self.graph_cache:
            edge_keys.append(f"{entity.lower()}→{used_bridge.lower()}")
        exp = Experience(
            query=query, query_entity=entity,
            bridge_entity=used_bridge, bridge_chunk=used_chunk,
            answer_payload=answer_payload,
            confidence=float(answer_score),
            hops=2, latency_ms=elapsed, convergence=conv,
            edge_keys=edge_keys,
        )
        self.experience_buffer.add(exp)
        return ReasoningResult(
            answer=answer_payload,
            confidence=float(answer_score),
            hops=2, max_hops=max_hops,
            convergence_reason=conv,
            graph=graph,
            visited_nodes=len(seen) + 1,
            elapsed_ms=elapsed,
            experience=exp,
        )

    # ------------------------------------------------------------------
    # Feedback: aggiorna la confidenza degli archi dall'esperienza
    # ------------------------------------------------------------------

    def feedback(self, exp_id: str, correct: bool) -> bool:
        """Fornisce feedback su un'esperienza passata.

        Aggiorna la confidenza dell'arco corrispondente nel GraphCache
        e registra il reward nell'ExperienceBuffer.

        Args:
            exp_id: ID dell'esperienza da aggiornare.
            correct: True se la risposta era corretta.

        Returns:
            True se il feedback è stato applicato.
        """
        exp = self.experience_buffer.get(exp_id)
        if not exp:
            return False
        exp.reward = 1.0 if correct else 0.0
        exp.rewarded_at = time.time()
        if self.graph_cache and exp.edge_keys:
            for key in exp.edge_keys:
                parts = key.split("→", 1)
                if len(parts) == 2:
                    self.graph_cache.record_cache_hit(
                        parts[0], parts[1], verified=correct,
                    )
            exp.consolidated = True
        # Chiude il loop anche verso il layer geometrico: adatta i pesi
        # RRF dell'ensemble in base a chi aveva rankato bene la risposta
        if self.ensemble and exp.answer_payload is not None:
            self.ensemble.reward(
                exp.query, _payload_text(exp.answer_payload), correct)
        return True

    # ------------------------------------------------------------------
    # Sleep: consolidamento della memoria
    # ------------------------------------------------------------------

    def sleep(self, decay: bool = True, merge: bool = False,
              forget: bool = True):
        """Manutenzione del GraphCache: decay, merge, forget, promotion.

        Applica anche le esperienze premiate non ancora consolidate
        (quelle a cui feedback() non ha già applicato il reward),
        rinforzando gli edge corrispondenti.

        I path piu' affidabili vengono promossi a hot edges (cache
        veloce), quelli deboli decadono o vengono dimenticati.
        """
        results = {}
        if self.graph_cache:
            results = self.graph_cache.sleep(
                decay=decay, merge=merge, forget=forget,
            )

        # Rinforza edge da esperienze positive non ancora consolidate
        # (feedback() marca consolidated=True quando applica il reward,
        # così ogni esperienza conta una volta sola)
        if self.graph_cache:
            for exp in self.experience_buffer.with_reward(0.5):
                if exp.consolidated or not exp.edge_keys:
                    continue
                for key in exp.edge_keys:
                    parts = key.split("→", 1)
                    if len(parts) == 2:
                        self.graph_cache.record_cache_hit(
                            parts[0], parts[1], verified=exp.is_positive(),
                        )
                exp.consolidated = True

        results["experiences"] = self.experience_buffer.size()
        results["labeled"] = len(self.experience_buffer.with_reward(0.0))
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self):
        return (f"ReasoningEngine(beam={self.beam_width}, "
                f"ensemble={self.ensemble is not None}, "
                f"graph_cache={self.graph_cache is not None})")
