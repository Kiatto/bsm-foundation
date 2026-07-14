"""
reasoning_demo.py — ReasoningEngine vs single-hop retrieval.

Confronto multi-hop reasoning vs single-hop su 30 domande,
con 50 documenti nel knowledge base.
"""

import sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bsm import BSM
from bsm.memory.reasoning_engine import ReasoningEngine


# 30 domande factuali
QA_PAIRS = [
    ("Where is the Eiffel Tower?", "Paris"),
    ("When was the Eiffel Tower built?", "1889"),
    ("What is the Great Wall of China?", "fortification"),
    ("How long is the Great Wall of China?", "13,000"),
    ("What does photosynthesis produce?", "glucose"),
    ("What pigment absorbs light in plants?", "chlorophyll"),
    ("What temperature does water freeze at?", "0"),
    ("What temperature does water boil at?", "100"),
    ("Who created Python?", "van Rossum"),
    ("When was Python created?", "1991"),
    ("How many neurons are in the human brain?", "86 billion"),
    ("What percentage of energy does the brain use?", "20%"),
    ("What does DNA stand for?", "deoxyribonucleic acid"),
    ("What are the four DNA bases?", "adenine"),
    ("What is the Louvre?", "museum"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("What is the speed of light?", "299,792"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What element has symbol Au?", "gold"),
    ("What is the largest ocean?", "Pacific"),
    ("How many bones are in the human body?", "206"),
    ("What year did World War II end?", "1945"),
    ("What is the chemical formula for water?", "H2O"),
    ("What is the smallest unit of life?", "cell"),
    ("What galaxy is Earth in?", "Milky Way"),
    ("What is the hardest natural substance?", "diamond"),
    ("What is the main gas in Earth's atmosphere?", "nitrogen"),
    ("What organ pumps blood in the body?", "heart"),
    ("What is the square root of 144?", "12"),
]

KNOWLEDGE_BASE = [
    "The Eiffel Tower is located in Paris, France. It was built in 1889.",
    "The Great Wall of China is an ancient fortification stretching 13,000 miles.",
    "The Louvre Museum in Paris is the world's largest art museum.",
    "Photosynthesis produces glucose and oxygen from sunlight.",
    "Chlorophyll is the green pigment that absorbs light for photosynthesis.",
    "Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius.",
    "Python was created by Guido van Rossum in 1991.",
    "Variables in Python are dynamically typed.",
    "The human brain contains 86 billion neurons and uses 20% of body energy.",
    "DNA stands for deoxyribonucleic acid with bases adenine, guanine, cytosine, thymine.",
    "Mars is known as the Red Planet due to its iron oxide surface.",
    "The speed of light in vacuum is 299,792 kilometers per second.",
    "William Shakespeare wrote Romeo and Juliet in the 1590s.",
    "Tokyo is the capital city of Japan.",
    "The chemical symbol Au represents gold on the periodic table.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "The human adult body contains 206 bones.",
    "World War II ended in 1945 after the surrender of Japan.",
    "The chemical formula for water is H2O, two hydrogen and one oxygen.",
    "The cell is the smallest unit of life in biology.",
    "Earth is located in the Milky Way galaxy.",
    "Diamond is the hardest known natural material.",
    "Nitrogen makes up about 78% of Earth's atmosphere.",
    "The heart pumps blood through the circulatory system.",
    "The square root of 144 equals 12.",
    "Mount Everest is the tallest mountain on Earth at 8,848 meters.",
    "The Amazon River is the largest river by water volume.",
    "Albert Einstein developed the theory of relativity.",
    "The human skeleton provides structure and protects organs.",
    "Oxygen is essential for cellular respiration in living organisms.",
    "The Earth orbits the Sun at about 150 million kilometers distance.",
    "Antarctica is the coldest continent on Earth.",
    "The United Nations was founded in 1945 after World War II.",
    "Electricity is the flow of electrons through a conductor.",
    "The Roman Empire fell in 476 AD.",
    "Gravity is the force that attracts objects with mass.",
    "The Internet was developed from ARPANET in the 1960s.",
    "Bacteria are single-celled microorganisms found everywhere.",
    "The Moon is Earth's only natural satellite.",
    "The Great Barrier Reef is the largest coral reef system.",
    "Leonardo da Vinci painted the Mona Lisa in the early 1500s.",
    "The periodic table organizes chemical elements by atomic number.",
    "Photosynthesis occurs in the chloroplasts of plant cells.",
    "The nervous system transmits signals through neurons.",
    "Mitosis is the process of cell division producing two identical cells.",
    "The Sahara is the largest hot desert in the world.",
    "Vincent van Gogh painted The Starry Night in 1889.",
    "The Industrial Revolution began in Britain in the 18th century.",
    "Microscopes use lenses to magnify small objects.",
    "Volcanoes form when magma rises through cracks in Earth's crust.",
]


def check(answer, keyword):
    kw = keyword.lower()
    if isinstance(answer, dict):
        return kw in answer.get("text", "").lower()
    return kw in str(answer).lower()


def main():
    print("=" * 65)
    print("  ReasoningEngine vs Single-Hop (50 documenti)")
    print("=" * 65)

    from bsm.memory.encoder.bsm_encoder import ProjectionEncoder
    enc = ProjectionEncoder(state_dim=256)
    enc.fit(KNOWLEDGE_BASE)
    bsm = BSM(encoder=enc, state_dim=256)

    for doc in KNOWLEDGE_BASE:
        state = bsm.encode(doc)
        bsm.observe(state, {"text": doc, "source": "bench"})

    print(f"\n  BSM store: {bsm._store.size()} entries, "
          f"encoder={bsm._encoder._name}")

    # ---- Single-hop ----
    print("\n  [Single-hop baseline]")
    single_correct = 0
    for q, kw in QA_PAIRS:
        state = bsm.encode(q)
        results = bsm.recall(state, k=1)
        ans = results[0][0] if results else None
        ok = check(ans, kw)
        if ok: single_correct += 1
        print(f"    {'✓' if ok else '✗'} {q[:44]:44s} → {str(ans)[:45] if ans else '—'}")
    single_acc = single_correct / len(QA_PAIRS)
    print(f"  Accuracy: {single_acc:.0%} ({single_correct}/{len(QA_PAIRS)})")

    # ---- Reasoning Engine ----
    print("\n  [ReasoningEngine]")
    engine = ReasoningEngine(bsm, beam_width=4)
    reason_correct = 0
    reason_results = []

    for q, kw in QA_PAIRS:
        result = engine.reason(q, max_hops=6)
        ans = result.answer
        ok = check(ans, kw)
        if ok: reason_correct += 1
        reason_results.append(result)
        print(f"    {'✓' if ok else '✗'} {q[:44]:44s} → "
              f"{str(ans)[:44] if ans else '—':44s} "
              f"[h={result.hops}, c={result.confidence:.2f}, "
              f"{result.convergence_reason}]")

    reason_acc = reason_correct / len(QA_PAIRS)
    avg_hops = np.mean([r.hops for r in reason_results])
    avg_conf = np.mean([r.confidence for r in reason_results])
    avg_ms = np.mean([r.elapsed_ms for r in reason_results])

    print(f"\n  Accuracy: {reason_acc:.0%} ({reason_correct}/{len(QA_PAIRS)})")
    print(f"  Avg hops: {avg_hops:.1f}, avg conf: {avg_conf:.2f}, "
          f"avg latency: {avg_ms:.0f}ms")

    reasons = {}
    for r in reason_results:
        reasons[r.convergence_reason] = reasons.get(r.convergence_reason, 0) + 1
    print(f"  Convergence: {reasons}")

    # ---- Summary ----
    print(f"\n{'=' * 65}")
    print(f"  CONFRONTO")
    print(f"{'=' * 65}")
    print(f"  {'Metodo':<35} {'Accuracy':<10} {'Hop':<8} {'Lat(ms)':<8}")
    print(f"  {'─' * 35} {'─' * 10} {'─' * 8} {'─' * 8}")
    print(f"  {'Single-hop':<35} {single_acc:<10.0%} {'1':<8} {'—':<8}")
    print(f"  {'ReasoningEngine':<35} {reason_acc:<10.0%} {avg_hops:<8.1f} {avg_ms:<8.0f}")
    print(f"  {'Δ':<35} {reason_acc - single_acc:<+10.0%}")

    # Detailed trace
    r = reason_results[0]
    print(f"\n  [Trace: {QA_PAIRS[0][0]}]")
    for i, hop in enumerate(r.graph):
        t = str(hop.retrieved[0][0])[:60] if hop.retrieved else "—"
        print(f"    Hop {i}: conf={hop.confidence:.3f}, entr={hop.entropy:.3f}, "
              f"top_chunk={t}")


if __name__ == "__main__":
    main()
