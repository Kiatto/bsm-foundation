"""Checklist automatica di attribuzione dei fallimenti (niente analisi
a mano). Per ogni miss, categorie MUTUAMENTE ESCLUSIVE valutate in
ordine, con criteri dichiarati:

  A_gold_absent      la gold non è in NESSUNA tripla (match token-based,
                     stesso criterio dello scoring: tokens(gold) ⊆
                     tokens(campo) o viceversa, su soggetti e oggetti)
  B_no_path          gold presente ma NON esiste percorso simbolico
                     ≤ 2 hop (a meno di inverse) da un'entità del piano
                     alla gold → estrazione incompleta sul percorso
  C_plan             percorso ≤ 2 hop ESISTE ma il piano a 1 hop +
                     vincolo non lo esprime o usa relazioni non
                     estratte → planner/schema (livello C)
  D_algebra          il piano era eseguibile sul grafo simbolico con
                     esito gold, ma l'esecuzione algebrica ha risposto
                     altro → livello B (algebra/carico)

Output: failure_attribution.json
"""

import json, sys
from pathlib import Path
from collections import deque

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate import tokens, match, audit_symbolic


def gold_present(triples, gold):
    return any(match(f, gold) for t in triples for f in (t[0], t[2]))


def path_within_2hops(triples, plan, gold):
    """BFS ≤2 hop, archi non orientati, da anchor (grounded per token)."""
    ents = sorted({str(t[0]) for t in triples} | {str(t[2]) for t in triples})
    adj = {}
    for s, r, o in triples:
        adj.setdefault(str(s), set()).add(str(o))
        adj.setdefault(str(o), set()).add(str(s))
    anchor_txt = plan.get("anchor", "")
    starts = [e for e in ents if match(e, anchor_txt)] or \
             [e for e in ents if tokens(anchor_txt) & tokens(e)]
    seen = set(starts)
    frontier = deque((s, 0) for s in starts)
    while frontier:
        node, d = frontier.popleft()
        if match(node, gold):
            return True
        if d < 2:
            for nxt in adj.get(node, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append((nxt, d + 1))
    return False


if __name__ == "__main__":
    res = json.load(open(HERE / "results.json"))
    plans = json.load(open(HERE / "plans.json"))
    meta = json.load(open(HERE / "sample_meta.json"))[1:]
    ext = {json.loads(l)["id"]: json.loads(l)
           for l in (HERE / "extractions.jsonl").open()}
    rows = [(m, ext[m["id"]]) for m in meta if m["id"] in ext]

    cats = {"A_gold_absent": 0, "B_no_path": 0, "C_plan": 0,
            "D_algebra": 0}
    per_q = []
    for x in res["details"]:
        if x["hit"]:
            continue
        i = x["i"]
        m, e = rows[i]
        plan = plans[i] if i < len(plans) else {}
        if not gold_present(e["triples"], m["answer"]):
            cat = "A_gold_absent"
        elif not path_within_2hops(e["triples"], plan, m["answer"]):
            cat = "B_no_path"
        elif not audit_symbolic(e["triples"], plan, m["answer"]):
            cat = "C_plan"
        else:
            cat = "D_algebra"
        cats[cat] += 1
        per_q.append({"i": i, "cat": cat, "gold": m["answer"],
                      "ans": x["ans"]})

    n_miss = sum(cats.values())
    print(f"miss totali: {n_miss}")
    for k, v in cats.items():
        print(f"  {k:15s} {v:3d}  ({v / n_miss:.0%})")
    json.dump({"counts": cats, "per_question": per_q},
              open(HERE / "failure_attribution.json", "w"), indent=1)
    print("→ failure_attribution.json")
