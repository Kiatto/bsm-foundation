"""
multihop_demo.py — ReasoningEngine su domande multi-hop con KB connesso.

Dimostra il multi-hop: domande la cui risposta non è nel chunk più simile,
ma richiede di concatenare 2 fatti attraverso un'entità ponte.
"""

import sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bsm import BSM
from bsm.memory.reasoning_engine import ReasoningEngine


# ---------------------------------------------------------------------------
# KB: 20 entità collegate + 30 trivia di contorno = 50 entry
# ---------------------------------------------------------------------------

ENTITIES = [
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
    "Meta Platforms Inc. owns the Facebook social network.",
    "Meta Platforms Inc. was founded by Mark Zuckerberg.",
    "Meta Platforms Inc. is headquartered in Menlo Park, California.",
    "Netflix Inc. provides streaming video on demand.",
    "Netflix Inc. was founded by Reed Hastings in 1997.",
    "Netflix Inc. is headquartered in Los Gatos, California.",
    "OpenAI created the ChatGPT language model.",
    "OpenAI was founded by Sam Altman and Elon Musk.",
    "OpenAI is headquartered in San Francisco, California.",
]

TRIVIA = [
    "The Eiffel Tower is located in Paris, France. It was built in 1889.",
    "The Great Wall of China is an ancient fortification stretching 13,000 miles.",
    "The Louvre Museum in Paris is the world's largest art museum.",
    "Photosynthesis produces glucose and oxygen from sunlight.",
    "Chlorophyll is the green pigment that absorbs light for photosynthesis.",
    "Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius.",
    "Python was created by Guido van Rossum in 1991.",
    "The human brain contains 86 billion neurons and uses 20% of body energy.",
    "DNA is deoxyribonucleic acid with bases adenine, guanine, cytosine, thymine.",
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
    "Antarctica is the coldest continent on Earth.",
    "The United Nations was founded in 1945 after World War II.",
    "The Roman Empire fell in 476 AD.",
]

KNOWLEDGE_BASE = ENTITIES + TRIVIA  # 54 total


# ---------------------------------------------------------------------------
# Domande multi-hop (2 hop) e single-hop
# ---------------------------------------------------------------------------

# Ogni domanda multi-hop: la risposta NON è nel chunk più simile alla domanda
# Serve connettere 2 fatti attraverso un'entità ponte
MULTIHOP_QA = [
    # (domanda, parola-chiave, entità ponte, hop_1_chunk, hop_2_chunk)
    ("Who founded the company that makes the iPhone?",
     "Steve Jobs", "Apple",
     "Apple Inc. manufactures the iPhone smartphone.",
     "Apple Inc. was founded by Steve Jobs in Cupertino, California."),
    ("Where is the company based that was founded by Bill Gates?",
     "Redmond", "Microsoft",
     "Microsoft Corporation was founded by Bill Gates in 1975.",
     "Microsoft Corporation is headquartered in Redmond, Washington."),
    ("What does the company founded by Jeff Bezos make?",
     "Alexa", "Amazon",
     "Amazon.com Inc. was founded by Jeff Bezos in 1994.",
     "Amazon.com Inc. created the Alexa voice assistant."),
    ("Who founded the company that makes electric cars?",
     "Elon Musk", "Tesla",
     "Tesla Inc. manufactures electric cars and SUVs.",
     "Tesla Inc. was founded by Elon Musk in 2003."),
    ("What does the company based in Cupertino make?",
     "iPhone", "Apple",
     "Apple Inc. is headquartered in Cupertino, California.",
     "Apple Inc. manufactures the iPhone smartphone."),
    ("Where is the company based that makes Android?",
     "Mountain View", "Google",
     "Google LLC developed the Android operating system.",
     "Google LLC is headquartered in Mountain View, California."),
    ("Who founded the company that makes Windows?",
     "Bill Gates", "Microsoft",
     "Microsoft Corporation develops the Windows operating system.",
     "Microsoft Corporation was founded by Bill Gates in 1975."),
    ("What does the company based in Menlo Park make?",
     "Facebook", "Meta",
     "Meta Platforms Inc. is headquartered in Menlo Park, California.",
     "Meta Platforms Inc. owns the Facebook social network."),
    ("Who founded the company based in Seattle?",
     "Jeff Bezos", "Amazon",
     "Amazon.com Inc. is headquartered in Seattle, Washington.",
     "Amazon.com Inc. was founded by Jeff Bezos in 1994."),
    ("Where is the company based that makes streaming video?",
     "Los Gatos", "Netflix",
     "Netflix Inc. provides streaming video on demand.",
     "Netflix Inc. is headquartered in Los Gatos, California."),
]

# Domande single-hop per controllo (la risposta è nel chunk più simile)
SINGLEHOP_QA = [
    ("Where is the Eiffel Tower?", "Paris"),
    ("What is the speed of light?", "299,792"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is the chemical formula for water?", "H2O"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check(answer, keyword):
    kw = keyword.lower()
    if isinstance(answer, dict):
        return kw in answer.get("text", "").lower()
    return kw in str(answer).lower()


def print_result(ok, prefix, text, extra=""):
    icon = "✓" if ok else "✗"
    print(f"    {icon} {prefix[:44]:44s} → {str(text)[:44] if text else '—':44s} {extra}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  ReasoningEngine — Multi-Hop Demo (KB connesso)")
    print("=" * 70)

    from bsm.memory.encoder.bsm_encoder import ProjectionEncoder
    enc = ProjectionEncoder(state_dim=256)
    enc.fit(KNOWLEDGE_BASE)
    bsm = BSM(encoder=enc, state_dim=256)

    for doc in KNOWLEDGE_BASE:
        state = bsm.encode(doc)
        bsm.observe(state, {"text": doc, "source": "kb"})

    print(f"\n  BSM store: {bsm._store.size()} entries, "
          f"encoder={bsm._encoder._name}")

    # ===== Single-hop su domande multi-hop =====
    print("\n  [Single-hop — fallisce perché la risposta è in un altro chunk]")
    single_correct = 0
    for q, kw, *_ in MULTIHOP_QA:
        state = bsm.encode(q)
        results = bsm.recall(state, k=1)
        ans = results[0][0] if results else None
        ok = check(ans, kw)
        if ok: single_correct += 1
        print_result(ok, q, ans)
    single_acc = single_correct / len(MULTIHOP_QA)
    print(f"\n  ➤ Single-hop accuracy: {single_acc:.0%} "
          f"({single_correct}/{len(MULTIHOP_QA)})")

    # ===== ReasoningEngine su domande multi-hop =====
    print("\n  [ReasoningEngine — multi-hop con keyword expansion]")
    # Test diverse configurazioni
    configs = [
        ("beam=3", dict(beam_width=3)),
        ("beam=4", dict(beam_width=4)),
        ("beam=5", dict(beam_width=5)),
    ]

    best_acc = 0
    best_name = ""
    best_results = None
    best_config = None

    for name, cfg in configs:
        engine = ReasoningEngine(bsm, **cfg)
        correct = 0
        results = []
        for q, kw, bridge, _, _ in MULTIHOP_QA:
            result = engine.reason(q, max_hops=6)
            ans = result.answer
            ok = check(ans, kw)
            if ok: correct += 1
            results.append((q, kw, bridge, result, ok))
        acc = correct / len(MULTIHOP_QA)
        print(f"\n    Config: {name}  →  {acc:.0%} ({correct}/{len(MULTIHOP_QA)})")
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_results = results
            best_config = cfg

    # Dettaglio della migliore config
    print(f"\n  ── Migliore: {best_name} ({best_acc:.0%}) ──")
    for q, kw, bridge, result, ok in best_results:
        ans = result.answer
        extra = (f"[h={result.hops}, c={result.confidence:.2f}, "
                 f"{result.convergence_reason}]")
        print_result(ok, q, ans, extra)

    avg_hops = np.mean([r.hops for _, _, _, r, _ in best_results])
    avg_conf = np.mean([r.confidence for _, _, _, r, _ in best_results])
    avg_ms = np.mean([r.elapsed_ms for _, _, _, r, _ in best_results])
    print(f"\n  Avg hops: {avg_hops:.1f}, avg conf: {avg_conf:.2f}, "
          f"avg latency: {avg_ms:.0f}ms")

    # ===== Single-hop su domande semplici (controllo) =====
    print("\n  [Controllo — single-hop su domande semplici]")
    sh_correct = 0
    for q, kw in SINGLEHOP_QA:
        state = bsm.encode(q)
        results = bsm.recall(state, k=1)
        ans = results[0][0] if results else None
        ok = check(ans, kw)
        if ok: sh_correct += 1
        print_result(ok, q, ans)
    sh_acc = sh_correct / len(SINGLEHOP_QA)
    print(f"\n  ➤ Single-hop accuracy: {sh_acc:.0%} "
          f"({sh_correct}/{len(SINGLEHOP_QA)})")

    # ===== Summary =====
    print(f"\n{'=' * 70}")
    print(f"  RIEPILOGO")
    print(f"{'=' * 70}")
    print(f"  {'Metodo':<25} {'Accuracy':<12} {'Hop':<8}")
    print(f"  {'─' * 25} {'─' * 12} {'─' * 8}")
    print(f"  {'Single-hop (multi-hop q)':<25} {single_acc:<12.0%} {'1':<8}")
    print(f"  {'ReasoningEngine':<25} {best_acc:<12.0%} {avg_hops:<8.1f}")
    print(f"  {'Δ':<25} {best_acc - single_acc:<+12.0%}")

    if best_results:
        q, kw, bridge, result, ok = best_results[0]
        print(f"\n  [Trace: {q[:40]}]")
        print(f"  Expected bridge: «{bridge}» → answer: «{kw}»")
        for i, hop in enumerate(result.graph):
            t = str(hop.retrieved[0][0])[:60] if hop.retrieved else "—"
            print(f"    Hop {i}: conf={hop.confidence:.3f}, entr={hop.entropy:.3f}, "
                  f"top={t}")


# ---------------------------------------------------------------------------
# main_ensemble — full system con EnsembleRetriever + GraphCache
# ---------------------------------------------------------------------------

def main_ensemble():
    """Versione completa: BSM + EnsembleRetriever + GraphCache + confidence propagation."""
    print("=" * 70)
    print("  ReasoningEngine v2 — Full Ensemble + GraphCache")
    print("=" * 70)

    from bsm.memory.encoder.bsm_encoder import ProjectionEncoder
    from bsm.memory.ensemble import EnsembleRetriever
    from bsm.memory.graph_cache import GraphCache
    from bsm import BSM

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

    print(f"\n  BSM store: {bsm._store.size()} entries")
    print(f"  Ensemble encoders: {list(ensemble.encoders.keys())}")
    print(f"  Entity encoder: Jaccard distance")

    test_cases = [
        ("beam=3", dict(beam_width=3)),
        ("beam=6", dict(beam_width=6)),
    ]

    best_acc = 0
    best_name = ""
    best_results = None

    for name, cfg in test_cases:
        engine = ReasoningEngine(bsm, ensemble=ensemble, graph_cache=gc, **cfg)
        correct = 0
        results = []
        for q, kw, bridge, _, _ in MULTIHOP_QA:
            result = engine.reason(q, max_hops=6)
            ans = result.answer
            ok = check(ans, kw)
            if ok: correct += 1
            results.append((q, kw, bridge, result, ok))
        acc = correct / len(MULTIHOP_QA)
        print(f"\n    Config: {name}  →  {acc:.0%} ({correct}/{len(MULTIHOP_QA)})")
        if acc >= best_acc:
            best_acc = acc
            best_name = name
            best_results = results

    # Dettaglio
    print(f"\n  ── Migliore: {best_name} ({best_acc:.0%}) ──")
    hop_counts = []
    for q, kw, bridge, result, ok in best_results:
        icon = "✓" if ok else "✗"
        ans = result.answer
        extra = (f"[hops={result.hops}, c={result.confidence:.2f}, "
                 f"{result.convergence_reason}]")
        hp = result.convergence_reason.split(":")[0]
        hop_counts.append(0 if hp == "graph_cache" else result.hops)
        print(f"    {icon} {q[:44]:44s} → {str(ans)[:44] if ans else '—':44s} {extra}")

    avg_hops = np.mean(hop_counts)
    avg_conf = np.mean([r.confidence for _, _, _, r, _ in best_results])
    avg_ms = np.mean([r.elapsed_ms for _, _, _, r, _ in best_results])
    print(f"\n  Avg hops: {avg_hops:.1f}, avg conf: {avg_conf:.2f}, "
          f"avg latency: {avg_ms:.0f}ms")
    print(f"  GraphCache: {gc}")

    # Seconda passata — deve usare la cache
    print(f"\n  ── Seconda passata (GraphCache) ──")
    engine = ReasoningEngine(bsm, ensemble=ensemble, graph_cache=gc, beam_width=6)
    correct = 0
    for q, kw, bridge, _, _ in MULTIHOP_QA:
        result = engine.reason(q, max_hops=6)
        ans = result.answer
        ok = check(ans, kw)
        if ok: correct += 1
        is_cached = "graph_cache" in result.convergence_reason
        icon = "✓" if ok else "✗"
        tag = " 🟢 cached" if is_cached else ""
        print(f"    {icon} {q[:44]:44s} → {str(ans)[:44] if ans else '—':44s}"
              f" [{result.elapsed_ms:.0f}ms]{tag}")
    print(f"\n  Accuracy: {correct}/{len(MULTIHOP_QA)} = {correct/len(MULTIHOP_QA):.0%}")


if __name__ == "__main__":
    import sys
    if "--ensemble" in sys.argv:
        main_ensemble()
    else:
        main()
