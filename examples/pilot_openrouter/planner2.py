"""Fase 1B: planner v2 (piani ≤2 hop, vocabolario relazioni allineato)
+ esecutore/audit estesi. Estrazioni CONGELATE (compiler A).

    OR_KEY=... python planner2.py plan     # 3 chiamate LLM batch
    python planner2.py eval                # offline: audit→contratto→misura
"""

import json, os, sys, time
from pathlib import Path
from collections import Counter
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate import (build_memory, match, tokens, Grounder,
                      predicted_accuracy, D, AUDIT_N, call_llm)
from extract import sample

PLANNER2_PROMPT = """For each question below, output a JSON retrieval plan.
You only see the question and the relation vocabulary — never the documents.

Relation vocabulary actually available in the knowledge base (use ONLY
these names): {rels}

Plan format (JSON array, one object per question, nothing else, no
markdown fences):
{{"q": <index>, "anchor": "<entity named in the question>",
 "chain": ["<rel1>"] or ["<rel1>", "<rel2>"],
 "constraint": {{"relation": "<rel>", "value": "<value>"}} or null}}

Semantics: start from anchor; candidates = entities linked by rel1
(either direction); with rel2, extend one more hop from those
candidates; if constraint present, keep candidates linked to value by
the constraint relation. Use 2 hops whenever the question mentions an
intermediate entity you must pass through (e.g. "the director of the
film X won what award?" → anchor=X, chain=[directed_by, won_award]).

Questions:
{questions}"""


def rel_vocab(ext, top=60):
    c = Counter(str(t[1]) for ts in ext.values() for t in ts)
    return [r for r, _ in c.most_common(top)]


def execute2(mem, entities, plan):
    g = Grounder(entities)
    anchor, aj = g(plan.get("anchor", ""))
    if aj <= 0.05:
        return None
    chain = plan.get("chain") or [plan.get("relation", "")]
    nodes = {anchor}
    from abm import phi
    for r in chain[:2]:
        nxt = set()
        for node in nodes:
            for e in entities:
                if e != node and (mem.member(node, r, e)
                                  or mem.member(node, r + "_inv", e)
                                  or mem.member(e, r, node)):
                    nxt.add(e)
        if not nxt:
            return None
        nodes = nxt
    cand = sorted(nodes - {anchor}) or sorted(nodes)
    con = plan.get("constraint")
    if con and cand:
        rc = con.get("relation", "")
        val, vj = g(con.get("value", ""))
        if vj > 0.05:
            kept = [c for c in cand if mem.member(c, rc, val)
                    or mem.member(val, rc, c)
                    or mem.member(c, rc + "_inv", val)]
            if kept:
                cand = kept
    if len(cand) > 1:
        r_last = chain[min(len(chain), 2) - 1]
        cand.sort(key=lambda e: -max(
            phi(mem.fact_hv(n, r_last, e), mem._trace)
            for n in ([anchor] if len(chain) == 1 else cand[:1] + [anchor])))
    return cand[0] if cand else None


def audit2(triples, plan, gold):
    ents = sorted({str(t[0]) for t in triples} | {str(t[2]) for t in triples})
    if not ents:
        return False
    g = Grounder(ents)
    anchor, aj = g(plan.get("anchor", ""))
    if aj <= 0.05:
        return False
    tset = {(str(a), str(b), str(c)) for a, b, c in triples}

    def linked(x, rr, y):
        return (x, rr, y) in tset or (y, rr, x) in tset

    chain = plan.get("chain") or [plan.get("relation", "")]
    nodes = {anchor}
    for r in chain[:2]:
        nodes = {e for n in nodes for e in ents if e != n and linked(n, r, e)}
        if not nodes:
            return False
    cand = nodes - {anchor} or nodes
    con = plan.get("constraint")
    if con and cand:
        val, vj = g(con.get("value", ""))
        if vj > 0.05:
            kept = {c for c in cand
                    if linked(c, con.get("relation", ""), val)}
            if kept:
                cand = kept
    return any(match(c, gold) for c in cand)


def load_rows():
    qids = json.loads((HERE / "phase1_question_ids.json").read_text())
    ext = {json.loads(l)["id"]: json.loads(l)["triples"]
           for l in (HERE / "extractions_A.jsonl").open()
           if json.loads(l)["triples"]}
    allq = {r["id"]: r for r in sample()}
    return [(allq[q], ext[q]) for q in qids if q in ext], ext


if __name__ == "__main__":
    mode = sys.argv[1]
    rows, ext = load_rows()

    if mode == "plan":
        import re as _re

        def parse_objects(txt):
            """Parser tollerante: estrae ogni {...} bilanciato con 'anchor'."""
            out, depth, start_i = [], 0, None
            for i, ch in enumerate(txt):
                if ch == "{":
                    if depth == 0:
                        start_i = i
                    depth += 1
                elif ch == "}" and depth:
                    depth -= 1
                    if depth == 0:
                        blob = txt[start_i:i + 1]
                        if '"anchor"' in blob:
                            try:
                                out.append(json.loads(blob))
                            except Exception:
                                pass
            return out

        vocab = rel_vocab(ext)
        qs = [m["question"] for m, _ in rows]
        plans = []
        models = ["tencent/hy3:free",
                  "nvidia/nemotron-3-nano-30b-a3b:free",
                  "openai/gpt-oss-20b:free"]
        for start in range(0, len(qs), 11):
            qtext = "\n".join(f"{start+i}. {q}"
                              for i, q in enumerate(qs[start:start + 11]))
            got = []
            for attempt in range(5):
                try:
                    txt = call_llm(models[attempt % len(models)],
                                   PLANNER2_PROMPT.format(
                                       rels=", ".join(vocab),
                                       questions=qtext))
                    got = parse_objects(txt)
                    if len(got) >= 8:              # batch accettato
                        break
                except Exception as e:
                    print(f"batch {start} retry: {str(e)[:80]}", flush=True)
                time.sleep(20)
            print(f"batch {start}: {len(got)} piani", flush=True)
            plans += got
        byid = {rows[p["q"]][0]["id"]: p for p in plans
                if isinstance(p.get("q"), int) and p["q"] < len(rows)}
        json.dump(byid, open(HERE / "plans2_by_id.json", "w"), indent=1)
        print("piani v2:", len(byid))

    elif mode == "eval":
        plans = json.loads((HERE / "plans2_by_id.json").read_text())
        audit = [audit2(t, plans.get(m["id"], {}), m["answer"])
                 for m, t in rows[:AUDIT_N]]
        pg = float(np.mean(audit))
        se = float(np.sqrt(max(pg * (1 - pg), 1e-9) / len(audit)))
        loads = [2 * len(t) for _, t in rows]
        ms = [len({str(x[0]) for x in t} | {str(x[2]) for x in t})
              for _, t in rows]
        # Pr per piani a k hop: prodotto per-hop (qui l'esecutore usa
        # membership, 1 test per candidato: manteniamo Pr per-hop medio)
        hops = [len(plans.get(m["id"], {}).get("chain", [1]) or [1])
                for m, _ in rows]
        pr = float(np.mean([predicted_accuracy(n, D, max(mm, 2)) ** h
                            for n, mm, h in zip(loads, ms, hops)]))
        contract = pg * pr
        half = 1.96 * se * pr + 0.02
        print(f"CONTRATTO v2 (pre-query): Pg={pg:.2f}±{1.96*se:.2f} "
              f"(n={len(audit)}) × Pr={pr:.3f} = {contract:.0%} ± {half:.0%}")
        ok, hits = 0, []
        for m, t in rows:
            mem = build_memory(t)
            ents = sorted({str(x[0]) for x in t} | {str(x[2]) for x in t})
            ans = execute2(mem, ents, plans.get(m["id"], {}))
            hit = bool(ans and match(ans, m["answer"]))
            ok += hit
            if hit:
                hits.append(m["question"][:55])
        acc = ok / len(rows)
        err = abs(acc - contract)
        print(f"MISURA: {ok}/{len(rows)} = {acc:.0%}")
        print(f"ERRORE: {err:.1%} → "
              f"{'RISPETTATO' if err <= half else 'VIOLATO'} (CI ±{half:.0%})")
        for h in hits:
            print("  hit:", h)
        json.dump({"pg": pg, "se": se, "pr": pr, "contract": contract,
                   "ci": half, "measured": acc, "n": len(rows),
                   "respected": bool(err <= half)},
                  open(HERE / "results_v2.json", "w"), indent=1)
