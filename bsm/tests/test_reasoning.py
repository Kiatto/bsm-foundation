"""
test_reasoning.py — Tests for EntityEncoder, EnsembleRetriever, ReasoningEngine.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time
import numpy as np
from bsm.memory.encoder.entity_encoder import EntityEncoder
from bsm.memory.ensemble import EnsembleRetriever
from bsm.memory.reasoning_engine import ReasoningEngine
from bsm.memory.graph_cache import GraphCache
from bsm.memory.cognitive_engine import CognitiveEngine
from bsm.memory.experience import Experience, ExperienceBuffer
from bsm import BSM
from bsm.memory.encoder.bsm_encoder import ProjectionEncoder


KNOWLEDGE_BASE = [
    "Apple Inc. manufactures the iPhone smartphone.",
    "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
    "Apple Inc. is headquartered in Cupertino, California.",
    "Microsoft Corporation develops the Windows operating system.",
    "Microsoft Corporation was founded by Bill Gates in 1975.",
    "Microsoft Corporation is headquartered in Redmond, Washington.",
    "Tesla Inc. manufactures electric cars and SUVs.",
    "Tesla Inc. was founded by Elon Musk in 2003.",
    "Tesla Inc. is headquartered in Austin, Texas.",
    "Google LLC developed the Android operating system.",
    "Google LLC was founded by Larry Page and Sergey Brin.",
    "Google LLC is headquartered in Mountain View, California.",
    "Amazon.com Inc. created the Alexa voice assistant.",
    "Amazon.com Inc. was founded by Jeff Bezos in 1994.",
    "Amazon.com Inc. is headquartered in Seattle, Washington.",
    "Netflix Inc. provides streaming video on demand.",
    "Netflix Inc. was founded by Reed Hastings in 1997.",
    "Netflix Inc. is headquartered in Los Gatos, California.",
    "The speed of light in vacuum is 299,792 kilometers per second.",
    "Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius.",
]


def check(answer, keyword):
    kw = keyword.lower()
    if isinstance(answer, dict):
        return kw in answer.get("text", "").lower()
    return kw in str(answer).lower()


# ---------------------------------------------------------------------------
# EntityEncoder
# ---------------------------------------------------------------------------

class TestEntityEncoder:
    def test_camelcase_extraction(self):
        enc = EntityEncoder(state_dim=256)
        ents = enc._extract_entities("The company that makes iPhone")
        assert "iPhone" in ents, f"camelCase 'iPhone' not extracted: {ents}"

    def test_proper_noun_extraction(self):
        enc = EntityEncoder(state_dim=256)
        ents = enc._extract_entities("Apple Inc. was founded by Steve Jobs")
        assert "Apple" in ents, f"'Apple' not found: {ents}"
        assert "Steve" in ents or "Jobs" in ents, f"'Steve Jobs' not found: {ents}"
        assert "Inc" in ents, f"'Inc' not found: {ents}"

    def test_jaccard_distance(self):
        enc = EntityEncoder(state_dim=256)
        enc.fit(KNOWLEDGE_BASE)
        for doc in KNOWLEDGE_BASE:
            enc.observe(doc, {"text": doc, "source": "kb"})

        results = enc.recall("iPhone", k=3)
        texts = [r[0]["text"] for r in results]
        assert any("iPhone" in t for t in texts), \
            f"No iPhone result found: {texts}"
        # "iPhone" + "Apple Inc manufactures iPhone" → Jaccard=1/3
        assert 0 < results[0][1] < 256, \
            f"Distance should be in (0, 256), got {results[0][1]}"

    def test_no_entity_chunks_dont_outrank(self):
        enc = EntityEncoder(state_dim=256)
        enc.fit(KNOWLEDGE_BASE)
        for doc in KNOWLEDGE_BASE:
            enc.observe(doc, {"text": doc, "source": "kb"})

        results = enc.recall("Seattle", k=5)
        texts = [r[0]["text"] for r in results]
        # "Amazon HQ in Seattle" should be ranked above random trivia
        amazon_hq = any("Seattle" in t for t in texts)
        assert amazon_hq, f"Amazon HQ chunk not in top 5: {texts}"


# ---------------------------------------------------------------------------
# EnsembleRetriever
# ---------------------------------------------------------------------------

class TestEnsembleRetriever:
    def test_ensemble_recall(self):
        ensemble = EnsembleRetriever(state_dim=256)
        ensemble.fit(KNOWLEDGE_BASE)
        for doc in KNOWLEDGE_BASE:
            ensemble.observe(doc, {"text": doc, "source": "kb"})

        results = ensemble.recall("iPhone", k=3)
        assert len(results) > 0
        texts = [r[0]["text"] for r in results]
        assert any("iPhone" in t for t in texts), \
            f"No iPhone result: {texts}"

    def test_raw_returns_per_encoder(self):
        ensemble = EnsembleRetriever(state_dim=256)
        ensemble.fit(KNOWLEDGE_BASE)
        for doc in KNOWLEDGE_BASE:
            ensemble.observe(doc, {"text": doc, "source": "kb"})

        raw = ensemble.recall_raw("iPhone", k=3)
        assert "projection" in raw
        assert "hash" in raw
        assert "entity" in raw
        for name, results in raw.items():
            assert len(results) > 0, f"{name} returned no results"


# ---------------------------------------------------------------------------
# ReasoningEngine (smoke tests)
# ---------------------------------------------------------------------------

def _build_engine():
    enc = ProjectionEncoder(state_dim=256)
    enc.fit(KNOWLEDGE_BASE)
    bsm = BSM(encoder=enc, state_dim=256)
    for doc in KNOWLEDGE_BASE:
        bsm.observe(bsm.encode(doc), {"text": doc, "source": "kb"})

    ensemble = EnsembleRetriever(state_dim=256)
    ensemble.fit(KNOWLEDGE_BASE)
    for doc in KNOWLEDGE_BASE:
        ensemble.observe(doc, {"text": doc, "source": "kb"})

    gc = GraphCache()
    engine = ReasoningEngine(
        bsm=bsm, ensemble=ensemble, graph_cache=gc, beam_width=6,
    )
    return engine, gc


class TestReasoningEngine:
    def test_multihop_iphone(self):
        engine, _ = _build_engine()
        result = engine.reason("Who founded the company that makes the iPhone?")
        assert result.answer is not None
        assert check(result.answer, "Steve Jobs"), \
            f"Expected Steve Jobs, got {result.answer}"

    def test_multihop_seattle(self):
        engine, _ = _build_engine()
        result = engine.reason("Who founded the company based in Seattle?")
        assert result.answer is not None
        assert check(result.answer, "Jeff Bezos"), \
            f"Expected Jeff Bezos, got {result.answer}"

    def test_multihop_cupertino(self):
        engine, _ = _build_engine()
        result = engine.reason("What does the company based in Cupertino make?")
        assert result.answer is not None
        assert check(result.answer, "iPhone"), \
            f"Expected iPhone, got {result.answer}"

    def test_graph_cache_populates(self):
        engine, gc = _build_engine()
        engine.reason("Who founded the company that makes Windows?")
        assert gc.size() == 1, f"Expected 1 entity in cache, got {gc.size()}"

    def test_graph_cache_hit(self):
        engine, gc = _build_engine()
        # Populate cache
        engine.reason("Who founded the company that makes Windows?")
        assert gc.size() == 1

        # Second call should hit cache
        result = engine.reason("Who founded the company that makes Windows?")
        assert "graph_cache" in result.convergence_reason, \
            f"Expected cache hit, got {result.convergence_reason}"
        assert result.hops == 0, \
            f"Expected 0 hops (cached), got {result.hops}"

    def test_confidence_propagation(self):
        engine, _ = _build_engine()
        result = engine.reason("Who founded the company that makes Windows?")
        assert 0.0 < result.confidence <= 1.0, \
            f"Invalid confidence: {result.confidence}"


# ---------------------------------------------------------------------------
# GraphCache Memory Confidence
# ---------------------------------------------------------------------------

class TestGraphCacheMemory:
    def test_edge_initial_confidence(self):
        from bsm.memory.graph_cache import Edge
        e = Edge(source="android", target="google",
                 answer_payload={"text": "Google LLC"})
        assert e.confidence == 0.50
        assert e.support == 0

    def test_edge_confidence_from_score(self):
        gc = GraphCache()
        gc.store(
            query="Who makes Android?",
            entity="Android",
            bridge_entity="Google",
            bridge_chunk="Google LLC developed Android",
            answer_payload={"text": "Google LLC"},
            score=0.92,
        )
        edges = gc.lookup_entity("Android")
        assert edges is not None and len(edges) > 0
        e = edges[0]
        assert e.confidence > 0.5
        assert e.support == 1
        assert e.success == 1

    def test_edge_record_hit(self):
        gc = GraphCache()
        for i in range(5):
            gc.store(
                query="Who makes Android?",
                entity="Android",
                bridge_entity="Google",
                bridge_chunk="Google LLC developed Android",
                answer_payload={"text": "Google LLC"},
                score=0.5,
            )
        edges = gc.lookup_entity("Android")
        e = edges[0]
        assert e.support == 5, f"Expected support=5, got {e.support}"
        assert e.confidence > 0.5

    def test_hot_edge_promotion(self):
        gc = GraphCache()
        for i in range(15):
            gc.store(
                query="Who makes Android?",
                entity="Android",
                bridge_entity="Google",
                bridge_chunk="Google LLC developed Android",
                answer_payload={"text": "Google LLC"},
                score=0.95,
            )
        m = gc.metrics()
        assert m["hot_edges"] > 0, f"No hot edges: {m}"

    def test_sleep_decay(self):
        gc = GraphCache()
        gc.store(
            query="Who makes Android?",
            entity="Android", bridge_entity="Google",
            bridge_chunk="Google LLC developed Android",
            answer_payload={"text": "Google LLC"}, score=0.90,
        )
        e = gc.lookup_entity("Android")[0]
        original = e.confidence

        # Sleep with far-future timestamp → heavy decay
        future = time.time() + 365 * 86400 * 10  # 10 years later
        result = gc.sleep(decay=True, merge=False, forget=False, now=future)
        assert result["decayed"] > 0

        e2 = gc.lookup_entity("Android")[0]
        assert e2.confidence < original, \
            f"Confidence should decay: {original} → {e2.confidence}"

    def test_sleep_forget(self):
        gc = GraphCache()
        gc.store(
            query="Who makes Android?",
            entity="Android", bridge_entity="Google",
            bridge_chunk="Google LLC developed Android",
            answer_payload={"text": "Google LLC"}, score=0.10,
        )
        m_before = gc.metrics()
        gc.sleep(decay=False, merge=False, forget=True)
        m_after = gc.metrics()
        assert m_after["forgotten_total"] > m_before["forgotten_total"], \
            "Edge with low confidence should be forgotten"

    def test_sleep_merge(self):
        gc = GraphCache()
        gc.store(
            query="Who makes Android?",
            entity="Android", bridge_entity="Google LLC",
            bridge_chunk="Google LLC developed Android",
            answer_payload={"text": "Google LLC"}, score=0.92,
        )
        gc.store(
            query="Who makes Android?",
            entity="Android", bridge_entity="Google",
            bridge_chunk="Google LLC developed Android",
            answer_payload={"text": "Google LLC"}, score=0.85,
        )
        gc.sleep(decay=False, merge=True, forget=False)
        edges = gc.lookup_entity("Android")
        assert len(edges) == 1, \
            f"Merged edges should collapse to 1, got {len(edges)}"

    def test_metrics_after_sleep(self):
        gc = GraphCache()
        for name in ("Android", "iPhone", "Windows"):
            gc.store(
                query=f"Who makes {name}?",
                entity=name, bridge_entity="TestCorp",
                bridge_chunk="TestCorp makes " + name,
                answer_payload={"text": "TestCorp"}, score=0.80,
            )
        m = gc.metrics()
        assert m["entities"] == 3
        assert m["edges"] == 3
        assert 0 < m["avg_confidence"] < 1.0

    def test_clear(self):
        gc = GraphCache()
        gc.store(
            query="Who makes Android?",
            entity="Android", bridge_entity="Google",
            bridge_chunk="Google LLC developed Android",
            answer_payload={"text": "Google LLC"}, score=0.90,
        )
        assert gc.size() > 0
        gc.clear()
        assert gc.size() == 0
        assert len(gc._hot) == 0


# ---------------------------------------------------------------------------
# Experience / Episodic Memory
# ---------------------------------------------------------------------------

class TestExperience:
    def test_create_experience(self):
        exp = Experience(
            query="Who makes Android?",
            query_entity="Android",
            bridge_entity="Google",
            answer_payload={"text": "Google LLC"},
            confidence=0.85,
            hops=2, latency_ms=12.0,
            convergence="q_decomp:Android→Google",
            edge_keys=["android→google"],
        )
        assert exp.id.startswith("exp_")
        assert exp.query_entity == "Android"
        assert not exp.is_rewarded()

    def test_experience_feedback(self):
        exp = Experience(
            query="Who makes Android?",
            query_entity="Android", bridge_entity="Google",
            answer_payload={"text": "Google LLC"},
            confidence=0.85, hops=2, latency_ms=10.0,
            convergence="q_decomp:Android→Google",
        )
        assert not exp.is_rewarded()
        exp.reward = 1.0
        exp.rewarded_at = time.time()
        assert exp.is_rewarded()
        assert exp.is_positive()

    def test_experience_buffer(self):
        buf = ExperienceBuffer(max_size=5)
        for i in range(10):
            exp = Experience(
                query=f"Q{i}", query_entity=f"E{i}",
                bridge_entity="Bridge", answer_payload={"text": "Ans"},
                confidence=0.5, hops=2, latency_ms=1.0,
                convergence="test",
            )
            buf.add(exp)
        assert buf.size() == 5, f"Buffer should cap at 5, got {buf.size()}"
        assert buf.recent(1)[0].query == "Q9"

    def test_experience_buffer_feedback(self):
        buf = ExperienceBuffer()
        exp = Experience(
            query="Q", query_entity="E", bridge_entity="B",
            answer_payload={"text": "A"}, confidence=0.5,
            hops=2, latency_ms=1.0, convergence="test",
        )
        buf.add(exp)
        # Apply feedback
        updated = buf.feedback(exp.id, correct=True)
        assert updated is not None
        assert updated.reward == 1.0
        # Verify via buffer
        stored = buf.get(exp.id)
        assert stored.reward == 1.0

    def test_reason_produces_experience(self):
        enc = ProjectionEncoder(state_dim=256)
        enc.fit(KNOWLEDGE_BASE)
        bsm = BSM(encoder=enc, state_dim=256)
        for doc in KNOWLEDGE_BASE:
            bsm.observe(bsm.encode(doc), {"text": doc, "source": "kb"})
        engine = ReasoningEngine(bsm=bsm, ensemble=None, graph_cache=None,
                                  beam_width=6)
        result = engine.reason("Who founded the company that makes Windows?")
        assert result.experience is not None
        assert result.experience.query_entity == "Windows"
        assert result.experience.confidence > 0

    def test_feedback_updates_edge(self):
        enc = ProjectionEncoder(state_dim=256)
        enc.fit(KNOWLEDGE_BASE)
        bsm = BSM(encoder=enc, state_dim=256)
        for doc in KNOWLEDGE_BASE:
            bsm.observe(bsm.encode(doc), {"text": doc, "source": "kb"})
        gc = GraphCache()
        engine = ReasoningEngine(bsm=bsm, ensemble=None, graph_cache=gc,
                                  beam_width=6)
        result = engine.reason("Who founded the company that makes Windows?")
        exp_id = result.experience.id
        # Cache should be populated
        assert gc.size() == 1
        edge = gc.lookup_entity("Windows")[0]
        before = edge.confidence
        # Apply positive feedback → confidence should rise
        engine.feedback(exp_id, correct=True)
        edge_after = gc.lookup_entity("Windows")[0]
        assert edge_after.confidence >= before, \
            f"Confidence should not decrease after positive feedback"


# ---------------------------------------------------------------------------
# CognitiveEngine — ciclo completo
# ---------------------------------------------------------------------------

class TestCognitiveEngine:
    def test_run_returns_result(self):
        enc = ProjectionEncoder(state_dim=256)
        enc.fit(KNOWLEDGE_BASE)
        bsm = BSM(encoder=enc, state_dim=256)
        for doc in KNOWLEDGE_BASE:
            bsm.observe(bsm.encode(doc), {"text": doc, "source": "kb"})
        ce = CognitiveEngine(bsm=bsm, graph_cache=GraphCache(), beam_width=6)
        result = ce.run("Who founded the company that makes Windows?")
        assert result.answer is not None
        assert check(result.answer, "Bill Gates")
        assert result.experience is not None

    def test_cognitive_metrics(self):
        enc = ProjectionEncoder(state_dim=256)
        enc.fit(KNOWLEDGE_BASE)
        bsm = BSM(encoder=enc, state_dim=256)
        for doc in KNOWLEDGE_BASE:
            bsm.observe(bsm.encode(doc), {"text": doc, "source": "kb"})
        ce = CognitiveEngine(bsm=bsm, graph_cache=GraphCache(), beam_width=6)
        ce.run("Who founded the company that makes Windows?")
        m = ce.metrics()
        assert m["total_queries"] == 1
        assert m["cache_misses"] == 1
        assert m["experience_buffer"] == 1

    def test_full_loop(self):
        """Observe → recall → reason → act → feedback → sleep."""
        enc = ProjectionEncoder(state_dim=256)
        enc.fit(KNOWLEDGE_BASE)
        bsm = BSM(encoder=enc, state_dim=256)
        for doc in KNOWLEDGE_BASE:
            bsm.observe(bsm.encode(doc), {"text": doc, "source": "kb"})
        gc = GraphCache()
        ce = CognitiveEngine(bsm=bsm, graph_cache=gc, beam_width=6)

        # reason + act
        result = ce.run("Who founded the company that makes Windows?")
        assert result.answer is not None
        exp_id = result.experience.id

        # feedback
        ok = ce.feedback(exp_id, correct=True)
        assert ok
        exp = ce.experience_buffer.get(exp_id)
        assert exp.is_positive()

        # sleep
        sleep_result = ce.sleep()
        assert sleep_result["experiences"] >= 1
        assert ce.metrics()["total_queries"] == 1
