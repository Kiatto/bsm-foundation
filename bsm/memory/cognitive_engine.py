"""
cognitive_engine.py — Ciclo cognitivo completo.

    observe → recall → reason → act → feedback → sleep

Il CognitiveEngine wrappa ReasoningEngine e GraphCache in un ciclo
esplicito.  Ogni passo è opzionale e tracciabile.

Usage:
    ce = CognitiveEngine(bsm, ensemble, graph_cache)
    result = ce.run("Who founded the company that makes the iPhone?")
    ce.feedback(result.experience.id, correct=True)
    ce.sleep()
"""

import time
from typing import Optional, Any

from bsm.memory.reasoning_engine import ReasoningEngine, ReasoningResult
from bsm.memory.graph_cache import GraphCache
from bsm.memory.ensemble import EnsembleRetriever
from bsm.memory.experience import Experience, ExperienceBuffer


class CognitiveEngine:
    """Ciclo cognitivo completo: observe → recall → reason → act → feedback → sleep.

    Ogni metodo del ciclo restituisce metriche e può essere usato
    indipendentemente.  Il metodo `run()` esegue il ciclo completo.
    """

    def __init__(self,
                 bsm: "BSM",
                 ensemble: Optional[EnsembleRetriever] = None,
                 graph_cache: Optional[GraphCache] = None,
                 experience_buffer: Optional[ExperienceBuffer] = None,
                 beam_width: int = 6):
        self.bsm = bsm
        self.ensemble = ensemble
        self.graph_cache = graph_cache
        self.experience_buffer = experience_buffer or ExperienceBuffer()

        self._engine = ReasoningEngine(
            bsm=bsm,
            ensemble=ensemble,
            graph_cache=graph_cache,
            experience_buffer=self.experience_buffer,
            beam_width=beam_width,
        )

        self._metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_confidence": 0.0,
            "avg_latency_ms": 0.0,
        }

    # ------------------------------------------------------------------
    # Ciclo completo
    # ------------------------------------------------------------------

    def run(self, query: str, max_hops: int = 6) -> ReasoningResult:
        """Esegue il ciclo completo: observe → recall → reason → act.

        observe  → already in BSM (data was fed during init)
        recall   → automatic within reason()
        reason   → ReasoningEngine.reason()
        act      → returns result + stores Experience

        Call .feedback() separately to close the loop.
        """
        result = self._engine.reason(query, max_hops=max_hops)

        self._metrics["total_queries"] += 1
        if result.hops == 0:
            self._metrics["cache_hits"] += 1
        else:
            self._metrics["cache_misses"] += 1

        n = self._metrics["total_queries"]
        self._metrics["avg_confidence"] += (
            (result.confidence - self._metrics["avg_confidence"]) / n
        )
        self._metrics["avg_latency_ms"] += (
            (result.elapsed_ms - self._metrics["avg_latency_ms"]) / n
        )

        return result

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def feedback(self, exp_id: str, correct: bool) -> bool:
        """Chiude il ciclo: aggiorna la confidenza degli edge."""
        return self._engine.feedback(exp_id, correct)

    # ------------------------------------------------------------------
    # Sleep
    # ------------------------------------------------------------------

    def sleep(self, decay: bool = True, merge: bool = True,
              forget: bool = True) -> dict:
        """Manutenzione: consolida edge e applica esperienze."""
        return self._engine.sleep(decay=decay, merge=merge, forget=forget)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def metrics(self) -> dict:
        m = dict(self._metrics)
        if self.graph_cache:
            m["graph_cache"] = self.graph_cache.metrics()
        m["experience_buffer"] = self.experience_buffer.size()
        m["labeled"] = len(self.experience_buffer.with_reward(0.0))
        return m

    def __repr__(self):
        m = self.metrics()
        return (f"CognitiveEngine(queries={m['total_queries']}, "
                f"cache_hits={m['cache_hits']}, "
                f"avg_conf={m['avg_confidence']:.2f})")
