"""
hotpotqa_eval.py — Prima validazione esterna: HotpotQA (distractor).

Per ogni domanda: 10 paragrafi Wikipedia (2 gold + 8 distrattori) in un
BSM fresco + AlgebraicReasoner; risposta dal motore integrato.

Metrica: containment — la risposta gold (stringa) è contenuta nel testo
restituito?  (Il sistema restituisce frasi, non span: EM/F1 classici non
si applicano direttamente; il containment sul top-1 è l'equivalente
onesto, e l'oracle dà il tetto raggiungibile.)

Riferimenti:
  - baseline: answer nel chunk top-1 del recall single-hop
  - integrato: answer nel testo restituito da reason()
  - oracle:   answer in almeno una frase del contesto (tetto del metric)

Metriche di stadio (dove si rompe la pipeline):
  - triple estratte per domanda, % planner match, % via algebrica

Uso:  python hotpotqa_eval.py [n_domande] [bridge|comparison|all]
"""

import sys, time, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow.parquet as pq

from bsm import BSM
from bsm.memory.encoder.bsm_encoder import ProjectionEncoder
from bsm.memory.reasoning_engine import ReasoningEngine
from bsm.memory.algebraic import AlgebraicReasoner

PARQUET = ("/tmp/claude-1000/-var-www-html-BitKore/"
           "c904bff8-7b97-4d4b-9e76-49f65ca6a95e/scratchpad/"
           "hotpot_val.parquet")


def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def contains(text: str, answer: str) -> bool:
    return norm(answer) in norm(text)


def payload_text(ans) -> str:
    if isinstance(ans, dict):
        return str(ans.get("text", "")) + " " + str(ans.get("entity", ""))
    return str(ans)


def eval_question(row):
    sentences = []
    for title, sents in zip(row["context"]["title"],
                            row["context"]["sentences"]):
        for s in sents:
            s = s.strip()
            if s:
                sentences.append(s)

    q, gold = row["question"], row["answer"]

    # oracle: il metric è raggiungibile?
    oracle = any(contains(s, gold) for s in sentences)

    enc = ProjectionEncoder(state_dim=256)
    enc.fit(sentences)
    bsm = BSM(encoder=enc, state_dim=256)
    for s in sentences:
        bsm.observe(bsm.encode(s), {"text": s})

    reasoner = AlgebraicReasoner(state_dim=2048)
    n_triples = reasoner.learn(sentences)
    planned = reasoner.planner.plan(q) is not None

    # baseline single-hop (+ recall@k: separa retrieval da reasoning)
    topk = bsm.recall(bsm.encode(q), k=10)
    base_hit = bool(topk) and contains(topk[0][0]["text"], gold)
    rec5 = any(contains(p["text"], gold) for p, _, _ in topk[:5])
    rec10 = any(contains(p["text"], gold) for p, _, _ in topk)

    # motore integrato
    engine = ReasoningEngine(bsm=bsm, beam_width=4, algebraic=reasoner)
    t0 = time.perf_counter()
    r = engine.reason(q)
    ms = (time.perf_counter() - t0) * 1000
    algebraic = r.convergence_reason.startswith("algebraic")
    hit = r.answer is not None and contains(payload_text(r.answer), gold)

    return dict(oracle=oracle, base=base_hit, rec5=rec5, rec10=rec10,
                hit=hit, alg=algebraic,
                planned=planned, triples=n_triples, ms=ms,
                level=row["level"], qtype=row["type"])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    qtype = sys.argv[2] if len(sys.argv) > 2 else "bridge"

    table = pq.read_table(PARQUET).to_pylist()
    rows = [r for r in table if qtype == "all" or r["type"] == qtype][:n]
    print(f"HotpotQA distractor validation — {len(rows)} domande "
          f"(type={qtype})")

    res = []
    for i, row in enumerate(rows):
        try:
            res.append(eval_question(row))
        except Exception as e:
            print(f"  ! errore su {row['id']}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  … {i + 1}/{len(rows)}")

    N = len(res)
    oracle = sum(r["oracle"] for r in res)
    base = sum(r["base"] for r in res)
    hit = sum(r["hit"] for r in res)
    alg = sum(r["alg"] for r in res)
    planned = sum(r["planned"] for r in res)
    triples = np.mean([r["triples"] for r in res])
    ms = np.median([r["ms"] for r in res])

    print(f"\n  {'metrica':38s} {'valore':>12}")
    print(f"  {'oracle (answer nel contesto)':38s} {oracle}/{N} ({oracle/N:.0%})")
    print(f"  {'baseline single-hop top-1':38s} {base}/{N} ({base/N:.0%})")
    r5 = sum(r['rec5'] for r in res); r10 = sum(r['rec10'] for r in res)
    print(f"  {'recall@5 / recall@10 (containment)':38s} "
          f"{r5/N:.0%} / {r10/N:.0%}")
    print(f"  {'motore integrato (containment top-1)':38s} {hit}/{N} ({hit/N:.0%})")
    print(f"\n  {'-- stadi della pipeline --':38s}")
    print(f"  {'triple estratte / domanda (media)':38s} {triples:>12.1f}")
    print(f"  {'query pianificate dal planner':38s} {planned}/{N} ({planned/N:.0%})")
    print(f"  {'risolte per via algebrica':38s} {alg}/{N} ({alg/N:.0%})")
    print(f"  {'latenza mediana reason()':38s} {ms:>10.1f}ms")

    # breakdown per hit tra i sottoinsiemi
    for key, label in (("base", "baseline"), ("hit", "integrato")):
        easy = [r for r in res if r["level"] == "easy"]
        hard = [r for r in res if r["level"] == "hard"]
        if easy and hard:
            print(f"  {label+' easy/hard':38s} "
                  f"{sum(r[key] for r in easy)}/{len(easy)}  "
                  f"{sum(r[key] for r in hard)}/{len(hard)}")

    out = {"n": N, "type": qtype, "oracle": oracle, "baseline": base,
           "integrated": hit, "algebraic": alg, "planned": planned,
           "avg_triples": float(triples), "median_ms": float(ms)}
    Path("hotpotqa_results.json").write_text(json.dumps(out, indent=2))
    print("\n  → hotpotqa_results.json")


if __name__ == "__main__":
    main()
