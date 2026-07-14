"""
experience.py — Episodic Memory per il BSM Cognitive Engine.

Ogni ragionamento produce un Experience:
    query → bridge → answer  (con confidenza, latenza, edge_keys)

Le esperienze sono il ponte tra il GraphCache (archi) e l'apprendimento:
    feedback(correct) → edge confidence evolve
    sleep()           → esperienze consolidate in archi stabili
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Any


@dataclass
class Experience:
    """Episodio di ragionamento completo.

    Contiene tutto ciò che serve per:
    - Tracciare l'evoluzione della confidenza nel tempo
    - Fornire feedback esplicito (reward)
    - Ricostruire il percorso di ragionamento
    """
    id: str = ""
    query: str = ""
    query_entity: str = ""
    bridge_entity: str = ""
    bridge_chunk: str = ""
    answer_payload: Any = None
    confidence: float = 0.0
    hops: int = 0
    latency_ms: float = 0.0
    convergence: str = ""
    edge_keys: List[str] = field(default_factory=list)
    created_at: float = 0.0
    reward: Optional[float] = None
    rewarded_at: Optional[float] = None
    consolidated: bool = False   # reward già applicato al GraphCache

    def __post_init__(self):
        if not self.id:
            self.id = f"exp_{uuid.uuid4().hex[:12]}"
        if self.created_at == 0.0:
            self.created_at = time.time()

    def is_rewarded(self) -> bool:
        return self.reward is not None

    def is_positive(self) -> bool:
        return self.reward is not None and self.reward > 0.5

    def __repr__(self):
        r = f"r={self.reward:.2f}" if self.reward is not None else "no_reward"
        return (f"Experience({self.id[:8]} {self.query_entity}→"
                f"{self.bridge_entity}, c={self.confidence:.2f}, {r})")


class ExperienceBuffer:
    """Buffer circolare di esperienze.

    Le esperienze recenti influenzano la confidenza degli archi
    durante sleep: esperienze positive rinforzano, negative indeboliscono.
    """

    def __init__(self, max_size: int = 10000):
        self._experiences: List[Experience] = []
        self._by_id: dict = {}
        self._max_size = max_size

    def add(self, exp: Experience):
        if len(self._experiences) >= self._max_size:
            old = self._experiences.pop(0)
            self._by_id.pop(old.id, None)
        self._experiences.append(exp)
        self._by_id[exp.id] = exp

    def get(self, exp_id: str) -> Optional[Experience]:
        return self._by_id.get(exp_id)

    def feedback(self, exp_id: str, correct: bool) -> Optional[Experience]:
        exp = self._by_id.get(exp_id)
        if not exp:
            return None
        exp.reward = 1.0 if correct else 0.0
        exp.rewarded_at = time.time()
        return exp

    def recent(self, n: int = 10) -> List[Experience]:
        return self._experiences[-n:]

    def for_entity(self, entity: str, n: int = 10) -> List[Experience]:
        results = [e for e in self._experiences
                   if e.query_entity.lower() == entity.lower() or
                   e.bridge_entity.lower() == entity.lower()]
        return results[-n:]

    def with_reward(self, min_reward: float = 0.5) -> List[Experience]:
        return [e for e in self._experiences
                if e.reward is not None and e.reward >= min_reward]

    def unlabeled(self) -> List[Experience]:
        return [e for e in self._experiences if e.reward is None]

    def size(self) -> int:
        return len(self._experiences)

    def clear(self):
        self._experiences.clear()
        self._by_id.clear()
