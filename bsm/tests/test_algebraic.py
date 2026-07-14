"""
test_algebraic.py — Tests per il percorso di ragionamento algebrico.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bsm import BSM
from bsm.memory.encoder.bsm_encoder import ProjectionEncoder
from bsm.memory.algebraic import (AlgebraicReasoner, TripleExtractor,
                                  QueryPlanner)
from bsm.memory.reasoning_engine import ReasoningEngine


KB = [
    "Apple Inc. manufactures the iPhone smartphone.",
    "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
    "Apple Inc. is headquartered in Cupertino, California.",
    "Microsoft Corporation develops the Windows operating system.",
    "Microsoft Corporation was founded by Bill Gates in 1975.",
    "Microsoft Corporation is headquartered in Redmond, Washington.",
    "Water freezes at 0 degrees Celsius.",
]


class TestTripleExtractor:
    def test_abbreviation_not_a_sentence_boundary(self):
        # "Inc. manufactures" non deve spezzare la frase
        t = TripleExtractor().extract(KB[0])
        assert ("apple inc", "makes", "iphone smartphone", KB[0]) in t

    def test_founded_strips_location_and_year(self):
        t = TripleExtractor().extract(KB[1])
        assert ("apple inc", "founded_by", "steve jobs", KB[1]) in t
        t2 = TripleExtractor().extract(KB[4])
        assert ("microsoft corporation", "founded_by", "bill gates",
                KB[4]) in t2

    def test_inverse_triples_emitted(self):
        t = TripleExtractor().extract(KB[2])
        rels = {(s, r, o) for s, r, o, _ in t}
        assert ("apple inc", "hq", "cupertino, california") in rels
        assert ("cupertino, california", "hq_of", "apple inc") in rels

    def test_non_factual_text_ignored(self):
        assert TripleExtractor().extract(KB[6]) == []


class TestQueryPlanner:
    def test_two_hop_chain(self):
        plan = QueryPlanner().plan(
            "Who founded the company that makes the iPhone?")
        assert plan == ("iphone", ["made_by", "founded_by"])

    def test_hq_of_chain(self):
        plan = QueryPlanner().plan(
            "What does the company based in Cupertino make?")
        assert plan == ("cupertino", ["hq_of", "makes"])

    def test_unknown_pattern_returns_none(self):
        assert QueryPlanner().plan("What is the meaning of life?") is None


class TestAlgebraicReasoner:
    def _reasoner(self):
        r = AlgebraicReasoner(state_dim=2048)
        r.learn(KB)
        return r

    def test_two_hop_answer_with_provenance(self):
        r = self._reasoner()
        out = r.answer("Who founded the company that makes Windows?")
        assert out is not None
        payload, conf, chain = out
        assert "Bill Gates" in payload["text"]
        assert payload["source"] == "algebraic"
        assert conf > 0.3
        assert "microsoft corporation" in chain

    def test_unplannable_query_returns_none(self):
        assert self._reasoner().answer("What freezes at 0 degrees?") is None

    def test_ungroundable_anchor_returns_none(self):
        assert self._reasoner().answer(
            "Who founded the company that makes spaceships?") is None


class TestEngineIntegration:
    def _engine(self, algebraic):
        enc = ProjectionEncoder(state_dim=256)
        enc.fit(KB)
        bsm = BSM(encoder=enc, state_dim=256)
        for doc in KB:
            bsm.observe(bsm.encode(doc), {"text": doc})
        return ReasoningEngine(bsm=bsm, beam_width=3, algebraic=algebraic)

    def test_algebraic_path_first(self):
        r = AlgebraicReasoner(state_dim=2048)
        r.learn(KB)
        engine = self._engine(r)
        result = engine.reason("Who founded the company that makes Windows?")
        assert result.convergence_reason.startswith("algebraic:")
        assert "Bill Gates" in result.answer["text"]
        assert result.experience is not None

    def test_fallback_to_heuristics(self):
        r = AlgebraicReasoner(state_dim=2048)
        r.learn(KB)
        engine = self._engine(r)
        # query fuori pattern → deve passare alle euristiche, non fallire
        result = engine.reason("Who founded Apple?")
        assert result.answer is not None

    def test_without_algebraic_unchanged(self):
        result = self._engine(None).reason(
            "Who founded the company that makes Windows?")
        assert not result.convergence_reason.startswith("algebraic")
