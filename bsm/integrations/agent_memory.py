"""
agent_memory.py — High-Level Agent Memory & Semantic Router for LLMs.

Integrates Bitpacked Algebraic Binary Memory (ABM), Memory Contracts, and
GraphCache probabilistic lifecycle management into a unified agent memory module.

Features:
- Deterministic zero-hallucination fact storage & retrieval.
- Pre-LLM Semantic Routing: Intercepts queries answerable from memory with high confidence.
- Multi-hop reasoning chain execution.
- Sleep-time memory consolidation (decay, hot path promotion, entity merging).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from bsm.memory.bitpack import BitpackedMemory
from bsm.memory.graph_cache import GraphCache, Edge


@dataclass
class SemanticRouteResult:
    handled: bool
    answer: Optional[str] = None
    confidence: float = 0.0
    source: str = "abm_memory"
    forward_to_llm: bool = True
    explanation: str = ""


class AgentMemory:
    """Unified Long-Term Agent Memory and Semantic Router for LLMs."""

    def __init__(self, dim: int = 2048, confidence_threshold: Optional[float] = None,
                 alpha: float = 0.01):
        """Create an agent memory.

        alpha: target false-accept rate for recall/routing. Acceptance uses
            bitpack.acceptance_z(candidates, alpha), which adapts to codebook size.

        confidence_threshold: optional *extra* floor on the reported confidence.
            Leave None (default). It is a legacy knob and it does not scale: with
            temperature=8 a threshold of 0.75 demands z >= 8.79, which measured
            0/30 accepted answers on a D=2048 memory holding 30 facts whose
            predicted accuracy was 1.00 — the router degenerated into always
            forwarding to the LLM. The calibrated test accepted 30/30 there, with
            a measured false-accept rate of ~1% = alpha.
        """
        self.dim = dim
        self.confidence_threshold = confidence_threshold
        self.alpha = alpha
        self.abm = BitpackedMemory(dim=dim)
        self.graph_cache = GraphCache()

    def remember(self, subject: str, relation: str, obj: str, weight: int = 1) -> str:
        """Store a factual triple (subject, relation, object) in memory.
        
        Updates both the holographic trace and the GraphCache edge lifecycle.
        """
        self.abm.store(subject, relation, obj, weight=weight)
        
        # Track edge in GraphCache
        query_key = f"{subject} {relation}"
        self.graph_cache.store(
            query=query_key,
            entity=subject,
            bridge_entity=relation,
            bridge_chunk=f"{subject} {relation} {obj}",
            answer_payload=obj,
            score=0.95
        )
        return f"Stored: ({subject}, {relation}, {obj})"

    def recall(self, subject: str, relation: str) -> Tuple[Optional[str], float]:
        """Query memory for a fact (subject, relation).

        Returns (object_name, confidence), or (None, confidence) when the hit is
        rejected. Acceptance is the candidate-count-aware test of
        bitpack.acceptance_z, which caps the false-accept rate at self.alpha
        whatever the codebook size; self.confidence_threshold, if set, applies as
        an additional floor.
        """
        ans, conf, _z = self.abm.query_calibrated(subject, relation, alpha=self.alpha)
        if ans is None:
            return None, conf
        if self.confidence_threshold is not None and conf < self.confidence_threshold:
            return None, conf
        return ans, conf


    def member_check(self, subject: str, relation: str, obj: str) -> bool:
        """Algebraic truth oracle: verifies if exact triple exists in memory."""
        return self.abm.member(subject, relation, obj)

    def chain_reasoning(self, start_entity: str, relations: List[str]) -> Tuple[Optional[str], float]:
        """Execute a multi-hop reasoning path, rejecting the chain if any hop fails.

        Every hop must clear the calibrated acceptance test: a chain is only as
        sound as its weakest hop, and an unaccepted hop makes every later hop a
        query about an entity the memory never asserted. The returned confidence
        is the product over hops.
        """
        node: Optional[str] = start_entity
        conf = 1.0
        for relation in relations:
            node, hop_conf = self.recall(node, relation)
            conf *= hop_conf
            if node is None:
                return None, conf
        return node, conf

    def semantic_route(self, prompt: str, entity: str, relation: str) -> SemanticRouteResult:
        """Front-end semantic router: check if prompt can be answered directly from ABM memory.

        Eliminates redundant, expensive LLM API calls when memory answers with a
        false-accept rate bounded by self.alpha.
        """
        answer, conf = self.recall(entity, relation)
        if answer is not None:
            return SemanticRouteResult(
                handled=True,
                answer=answer,
                confidence=conf,
                source="abm_memory",
                forward_to_llm=False,
                explanation=f"Answered directly from ABM holographic memory with calibrated confidence {conf:.1%}."
            )
        
        return SemanticRouteResult(
            handled=False,
            answer=None,
            confidence=conf,
            source="llm_fallback",
            forward_to_llm=True,
            explanation=(f"Memory hit not distinguishable from noise at alpha={self.alpha:.0%} "
                         f"(confidence {conf:.1%}). Forwarding query to LLM.")
        )

    def remember_text(self, text: str, extractor_fn=None) -> List[str]:
        """Extract relational triples from natural text and store them in memory.
        
        Uses extractor_fn if provided, otherwise uses a built-in heuristic pattern extractor.
        """
        stored_results = []
        if extractor_fn is not None:
            triples = extractor_fn(text)
        else:
            import re
            triples = []
            patterns = [
                (r"(\w+)\s+is\s+(?:a\s+|an\s+)?(\w+)", "is_a"),
                (r"(\w+)\s+has\s+(\w+)", "has"),
                (r"(\w+)\s+works\s+at\s+(\w+)", "works_at"),
                (r"(\w+)\s+located\s+in\s+(\w+)", "located_in"),
                (r"(\w+)\s+depends\s+on\s+(\w+)", "depends_on"),
                (r"(\w+)\s+uses\s+(\w+)", "uses"),
            ]
            for pat, rel in patterns:
                matches = re.findall(pat, text, re.IGNORECASE)
                for s, o in matches:
                    triples.append((s.lower().strip(), rel, o.lower().strip()))

        for s, r, o in triples:
            res = self.remember(s, r, o)
            stored_results.append(res)

        return stored_results

    def sleep_consolidation(self) -> Dict[str, int]:
        """Trigger sleep-time maintenance (decay, promotion, forgetting)."""
        metrics = self.graph_cache.sleep()
        return metrics

    def stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics including Memory Contract and GraphCache stats."""
        gc_metrics = self.graph_cache.metrics()
        return {
            "dim": self.dim,
            "facts_count": len(self.abm._facts),
            "items_count": len(self.abm.items),
            "expected_accuracy": self.abm.expected_accuracy(),
            "graph_entities": gc_metrics.get("entities", 0),
            "graph_edges": gc_metrics.get("edges", 0),
            "hot_edges": gc_metrics.get("hot_edges", 0),
            "avg_edge_confidence": gc_metrics.get("avg_confidence", 0.0),
        }

    def save(self, dir_path: Any):
        """Save AgentMemory trace to disk."""
        from pathlib import Path
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        self.abm.save(path / "abm")

    @classmethod
    def load(cls, dir_path: Any, mmap_mode: Optional[str] = None) -> "AgentMemory":
        """Load AgentMemory trace from disk."""
        from pathlib import Path
        path = Path(dir_path)
        abm = BitpackedMemory.load(path / "abm", mmap_mode=mmap_mode)
        instance = cls(dim=abm.dim)
        instance.abm = abm
        return instance

