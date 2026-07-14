"""
algebraic_demo.py — Il paradigma adottato: percorso algebrico (VSA)
integrato nel ReasoningEngine, euristiche come fallback.

Confronto sullo stesso benchmark di multihop_demo:
  1. motore euristico (baseline attuale)
  2. motore integrato (algebrico + fallback)
  con quota di query risolte per via algebrica e latenze.
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from bsm import BSM
from bsm.memory.encoder.bsm_encoder import ProjectionEncoder
from bsm.memory.reasoning_engine import ReasoningEngine
from bsm.memory.algebraic import AlgebraicReasoner
from multihop_demo import KNOWLEDGE_BASE, MULTIHOP_QA, check


def build_engine(algebraic=None):
    enc = ProjectionEncoder(state_dim=256)
    enc.fit(KNOWLEDGE_BASE)
    bsm = BSM(encoder=enc, state_dim=256)
    for doc in KNOWLEDGE_BASE:
        bsm.observe(bsm.encode(doc), {"text": doc, "source": "kb"})
    return ReasoningEngine(bsm=bsm, beam_width=3, algebraic=algebraic)


def run(engine, label):
    correct = alg_used = 0
    lat = []
    rows = []
    for q, kw, *_ in MULTIHOP_QA:
        r = engine.reason(q)
        ok = check(r.answer, kw)
        correct += ok
        is_alg = r.convergence_reason.startswith("algebraic")
        alg_used += is_alg
        lat.append(r.elapsed_ms)
        rows.append((ok, is_alg, q, r))
    print(f"\n  [{label}]")
    for ok, is_alg, q, r in rows:
        via = "XOR " if is_alg else "heur"
        print(f"    {'✓' if ok else '✗'} [{via}] {q[:46]:46s} "
              f"c={r.confidence:.2f} {r.elapsed_ms:5.1f}ms")
    print(f"  → accuracy {correct}/{len(MULTIHOP_QA)}, "
          f"via algebrica {alg_used}/{len(MULTIHOP_QA)}, "
          f"latenza mediana {np.median(lat):.1f}ms")
    return correct, alg_used, float(np.median(lat))


def main():
    print("=" * 70)
    print("  Percorso algebrico (VSA) nel ReasoningEngine — stesso benchmark")
    print("=" * 70)

    reasoner = AlgebraicReasoner(state_dim=2048)
    n = reasoner.learn(KNOWLEDGE_BASE)
    print(f"\n  {reasoner}  ({n} triple da {len(KNOWLEDGE_BASE)} frasi)")

    base = run(build_engine(algebraic=None), "solo euristiche (baseline)")
    integ = run(build_engine(algebraic=reasoner),
                "integrato: algebra + fallback")

    print(f"\n  {'':24s}{'baseline':>10} {'integrato':>10}")
    print(f"  {'accuracy':24s}{base[0]:>9}/{len(MULTIHOP_QA)} "
          f"{integ[0]:>9}/{len(MULTIHOP_QA)}")
    print(f"  {'quota algebrica':24s}{base[1]:>10} {integ[1]:>10}")
    print(f"  {'latenza mediana (ms)':24s}{base[2]:>10.1f} {integ[2]:>10.1f}")


if __name__ == "__main__":
    main()
