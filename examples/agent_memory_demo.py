"""
agent_memory_demo.py — Demo of LLM Agent Memory & Semantic Router using Bitpacked ABM.

Demonstrates:
1. Fact acquisition (remembering facts during conversation).
2. Semantic routing (intercepting known queries with 0ms LLM cost).
3. Zero-hallucination verification (member truth oracle).
4. Multi-hop reasoning chain execution.
5. Memory consolidation via sleep maintenance.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bsm.integrations.agent_memory import AgentMemory


def main():
    print("================================================================")
    print("      DEMO: LLM AGENT MEMORY & SEMANTIC ROUTER (Bitpacked ABM)  ")
    print("================================================================\n")

    # D=2048, acceptance calibrated to a 1% false-accept rate (no fixed
    # confidence cutoff: that one rejects valid hits as facts accumulate)
    memory = AgentMemory(dim=2048, alpha=0.01)

    print("--- 1. Fact Acquisition ---")
    facts = [
        ("user_101", "has_role", "lead_engineer"),
        ("lead_engineer", "has_access_to", "production_k8s_cluster"),
        ("production_k8s_cluster", "located_in", "region_eu_central"),
        ("lead_engineer", "assigned_project", "quantum_core"),
    ]
    for s, r, o in facts:
        res = memory.remember(s, r, o)
        print(f"  [STORE] {res}")

    print("\n--- 2. Pre-LLM Semantic Router ---")
    prompts = [
        ("What role does user_101 have?", "user_101", "has_role"),
        ("What project is lead_engineer assigned to?", "lead_engineer", "assigned_project"),
        ("What is user_101's home address?", "user_101", "home_address"),
    ]

    for prompt_text, entity, relation in prompts:
        route = memory.semantic_route(prompt_text, entity, relation)
        print(f"\nUser Query: '{prompt_text}'")
        if route.handled:
            print(f"  [ROUTER ACCELERATED] Answer: '{route.answer}' (Confidence: {route.confidence:.1%})")
            print(f"  Explanation: {route.explanation}")
        else:
            print(f"  [FORWARDED TO LLM] Confidence: {route.confidence:.1%}")
            print(f"  Explanation: {route.explanation}")

    print("\n--- 3. Multi-hop Reasoning Chain ---")
    start = "user_101"
    chain = ["has_role", "has_access_to", "located_in"]
    dest, chain_conf = memory.chain_reasoning(start, chain)
    print(f"  Chain: {' -> '.join([start] + chain)}")
    print(f"  Derived Destination: '{dest}' (Combined Confidence: {chain_conf:.1%})")

    print("\n--- 4. Zero-Hallucination Algebraic Truth Oracle ---")
    is_true1 = memory.member_check("user_101", "has_role", "lead_engineer")
    is_true2 = memory.member_check("user_101", "has_role", "intern")
    print(f"  Is ('user_101', 'has_role', 'lead_engineer') in memory? -> {is_true1}")
    print(f"  Is ('user_101', 'has_role', 'intern') in memory?        -> {is_true2}")

    print("\n--- 5. Sleep Consolidation & Memory Diagnostics ---")
    sleep_stats = memory.sleep_consolidation()
    print(f"  Sleep Maintenance Stats: {sleep_stats}")

    diag = memory.stats()
    print(f"\n  Memory Contract & Diagnostics:")
    print(f"    Vector Dimension    : {diag['dim']}")
    print(f"    Facts Stored        : {diag['facts_count']}")
    print(f"    Expected Accuracy   : {diag['expected_accuracy']:.1%}")
    print(f"    Graph Cache Edges   : {diag['graph_edges']}")
    print(f"================================================================\n")


if __name__ == "__main__":
    main()
