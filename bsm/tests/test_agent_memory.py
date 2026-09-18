"""
test_agent_memory.py — Tests for AgentMemory integration and Semantic Router.
"""

import pytest
from bsm.integrations.agent_memory import AgentMemory


def test_agent_memory_remember_and_recall():
    agent_mem = AgentMemory(dim=2048, confidence_threshold=0.70)
    agent_mem.remember("server_alpha", "runs_service", "payment_gateway")
    agent_mem.remember("payment_gateway", "uses_database", "postgres_cluster")

    # Recall stored facts
    ans, conf = agent_mem.recall("server_alpha", "runs_service")
    assert ans == "payment_gateway"
    assert conf >= 0.70

    # Recall non-existent fact
    ans_unknown, conf_unknown = agent_mem.recall("server_alpha", "owner")
    assert ans_unknown is None or conf_unknown < 0.70


def test_semantic_router_decision():
    agent_mem = AgentMemory(dim=2048, confidence_threshold=0.75)
    agent_mem.remember("user_101", "has_permission", "admin_dashboard")

    # High-confidence query -> intercepted by semantic router
    res = agent_mem.semantic_route("Does user_101 have access to admin_dashboard?", "user_101", "has_permission")
    assert res.handled is True
    assert res.answer == "admin_dashboard"
    assert res.forward_to_llm is False
    assert res.confidence >= 0.75

    # Unknown query -> forwarded to LLM
    res_unknown = agent_mem.semantic_route("What is user_101's phone number?", "user_101", "phone_number")
    assert res_unknown.handled is False
    assert res_unknown.forward_to_llm is True


def test_agent_memory_multi_hop_and_sleep():
    agent_mem = AgentMemory(dim=2048)
    agent_mem.remember("auth_service", "connects_to", "redis_cache")
    agent_mem.remember("redis_cache", "hosted_on", "node_42")

    # Multi-hop chain
    dest, conf = agent_mem.chain_reasoning("auth_service", ["connects_to", "hosted_on"])
    assert dest == "node_42"

    # Sleep consolidation
    metrics = agent_mem.sleep_consolidation()
    assert isinstance(metrics, dict)

    stats = agent_mem.stats()
    assert stats["facts_count"] == 2
    assert stats["dim"] == 2048


def test_router_still_answers_when_memory_is_loaded():
    """The router must keep intercepting queries as facts accumulate.

    With the legacy fixed 0.75 confidence floor this degenerated into forwarding
    every query to the LLM once the trace held a few dozen facts.
    """
    agent_mem = AgentMemory(dim=2048)
    for i in range(30):
        agent_mem.remember(f"user_{i}", "has_role", f"role_{i}")

    handled = sum(1 for i in range(30)
                  if agent_mem.semantic_route(f"role of user_{i}?",
                                              f"user_{i}", "has_role").handled)
    assert handled == 30

    unknown = agent_mem.semantic_route("home address of user_3?", "user_3", "home_address")
    assert unknown.handled is False
    assert unknown.forward_to_llm is True


def test_chain_reasoning_rejects_when_a_hop_is_unsupported():
    """A hop the memory never asserted must abort the chain, not propagate a guess."""
    agent_mem = AgentMemory(dim=2048)
    agent_mem.remember("svc_a", "connects_to", "svc_b")

    dest, _conf = agent_mem.chain_reasoning("svc_a", ["connects_to", "hosted_on"])
    assert dest is None
