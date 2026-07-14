"""BSM Phase II — Cognitive Memory Layer."""

from .encoder.bsm_encoder import HashEncoder, ProjectionEncoder, LearnedEncoder
from .encoder.entity_encoder import EntityEncoder
from .store.memory_store import MemoryStore
from .router.bsm_router import BSMRouter
from .context_compiler import ContextCompiler
from .ensemble import EnsembleRetriever
from .graph_cache import GraphCache
from .reasoning_engine import ReasoningEngine
from .prototypes import PrototypeIndex
from .vsa import (WorkingMemory, SemanticMemory, ItemMemory,
                  RoleProjection, bind_xor, bundle, permute, random_hv)
from .algebraic import AlgebraicReasoner, TripleExtractor, QueryPlanner

__all__ = [
    "HashEncoder", "ProjectionEncoder", "LearnedEncoder",
    "EntityEncoder",
    "MemoryStore", "BSMRouter", "ContextCompiler",
    "EnsembleRetriever", "GraphCache", "ReasoningEngine",
    "PrototypeIndex",
    "WorkingMemory", "SemanticMemory", "ItemMemory", "RoleProjection",
    "bind_xor", "bundle", "permute", "random_hv",
    "AlgebraicReasoner", "TripleExtractor", "QueryPlanner",
]
