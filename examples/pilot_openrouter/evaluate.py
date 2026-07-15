"""Valutazione del pilot OpenRouter (PROTOCOL.md).

Fasi: planner (1 chiamata batch, vede SOLO le domande) → audit sulle
prime 15 (Pg simbolico, a meno di inverse) → CONTRATTO pre-query →
esecuzione algebrica sulle 40 → verifica.

Esecutore: oracolo di membership della reference (gestisce relazioni
multi-oggetto), grounding di anchor/valori via sketch MinHash token.
"""

import json, os, re, sys, time, urllib.request
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT))
from abm import Memory, predicted_accuracy
from bsm.memory.encoder.entity_encoder import _minhash_sketch

D, SK, AUDIT_N = 8192, 256, 15


def norm(s):
    s = re.sub(r"\b(a|an|the)\b", " ", str(s).lower())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def tokens(s):
    return {w.rstrip("s") for w in norm(s).split()
            if w and w not in {"of", "in", "and", "on", "to", "by",
                               "for", "with"}}


def match(a, b):
    """Risposta corretta se i token della gold sono contenuti (o
    viceversa) — stesso criterio del pilot precedente."""
    ta, tb = tokens(a), tokens(b)
    return bool(ta and tb and (ta <= tb or tb <= ta))


class Grounder:
    def __init__(self, names):
        self.names = list(names)
        self.sk = [_minhash_sketch(tokens(n) or {"_"}, SK)
                   for n in self.names]

    def __call__(self, text):
        probe = _minhash_sketch(tokens(text) or {"_"}, SK)
        d = [int(np.count_nonzero(probe != s)) for s in self.sk]
        i = int(np.argmin(d))
        return self.names[i], 1 - 2 * d[i] / SK


PLANNER_PROMPT = """For each question below, output a JSON retrieval plan.
You only see the question — never the documents. Available relations:
{rels} (plus their inverses, suffix _inv).

Plan format (one JSON object per question, output a JSON array of them,
nothing else, no markdown fences):
{{"q": <index>, "anchor": "<entity named in the question to start from>",
 "relation": "<relation from anchor to the candidate answers>",
 "constraint": {{"relation": "<relation>", "value": "<value>"}} or null}}

Semantics: candidates = entities linked to anchor by relation (either
direction); if constraint is present, keep candidates linked to value
by the constraint relation. Example: "which star of FILM was born on
DATE?" → anchor=FILM, relation=starred, constraint={{"relation":
"born_on", "value": DATE}}.

Questions:
{questions}"""


def call_llm(model, prompt, max_tok=8000):
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps({"model": model, "max_tokens": max_tok,
                         "messages": [{"role": "user",
                                       "content": prompt}]}).encode(),
        headers={"Authorization": "Bearer " + os.environ["OR_KEY"],
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        msg = json.load(resp)["choices"][0]["message"]
        txt = msg.get("content") or ""
        if "[" not in txt:
            txt = msg.get("reasoning") or txt
        return txt


def get_plans(questions):
    cache = HERE / "plans.json"
    if cache.exists():
        return json.loads(cache.read_text())
    rels = ("directed_by, written_by, produced_by, starred, born_in, "
            "born_on, died_in, nationality, occupation, member_of, "
            "part_of, located_in, capital_of, founded_by, founded_in, "
            "created_by, published_in, released_in, genre, instance_of, "
            "known_for, spouse_of, child_of, works_for, plays_for, "
            "album_of, song_by, author_of, developed_by, based_on")
    qtext = "\n".join(f"{i}. {q}" for i, q in enumerate(questions))
    txt = call_llm("tencent/hy3:free",
                   PLANNER_PROMPT.format(rels=rels, questions=qtext))
    plans = json.loads(txt[txt.find("["):txt.rfind("]") + 1])
    cache.write_text(json.dumps(plans, indent=1))
    return plans


def build_memory(triples):
    mem = Memory(D)
    for s, r, o in triples:
        mem._facts.append(mem.fact_hv(str(s), str(r), str(o)))
        mem._facts.append(mem.fact_hv(str(o), str(r) + "_inv", str(s)))
    from abm import bundle
    mem._trace = bundle(mem._facts)
    return mem


def execute(mem, entities, plan):
    """Membership algebrico: candidati via relazione (entrambe le
    direzioni), poi filtro sul vincolo."""
    g = Grounder(entities)
    anchor, aj = g(plan.get("anchor", ""))
    if aj <= 0.05:
        return None
    r = plan.get("relation", "")
    cand = [e for e in entities if e != anchor and
            (mem.member(anchor, r, e) or mem.member(anchor, r + "_inv", e)
             or mem.member(e, r, anchor))]
    con = plan.get("constraint")
    if con and cand:
        rc = con.get("relation", "")
        val, vj = g(con.get("value", ""))
        if vj > 0.05:
            kept = [c for c in cand if
                    mem.member(c, rc, val) or mem.member(val, rc, c)
                    or mem.member(c, rc + "_inv", val)]
            if kept:
                cand = kept
    if not cand:
        return None
    if len(cand) > 1:
        from abm import phi
        cand.sort(key=lambda e: -phi(mem.fact_hv(anchor, r, e), mem._trace))
    return cand[0]


def audit_symbolic(triples, plan, gold):
    """Pg: piano eseguibile sul grafo simbolico (a meno di inverse) e
    la gold raggiungibile. Stessa semantica dell'esecutore, su insiemi."""
    ents = sorted({str(t[0]) for t in triples} | {str(t[2]) for t in triples})
    if not ents:
        return False
    g = Grounder(ents)
    anchor, aj = g(plan.get("anchor", ""))
    if aj <= 0.05:
        return False
    r = plan.get("relation", "")
    tset = {(str(a), str(b), str(c)) for a, b, c in triples}
    def linked(x, rr, y):
        return (x, rr, y) in tset or (y, rr, x) in tset
    cand = [e for e in ents if e != anchor and linked(anchor, r, e)]
    con = plan.get("constraint")
    if con and cand:
        val, vj = g(con.get("value", ""))
        if vj > 0.05:
            kept = [c for c in cand if linked(c, con.get("relation", ""), val)]
            if kept:
                cand = kept
    return any(match(c, gold) for c in cand)


if __name__ == "__main__":
    meta = json.loads((HERE / "sample_meta.json").read_text())[1:]
    ext = {json.loads(l)["id"]: json.loads(l)
           for l in (HERE / "extractions.jsonl").open() if l.strip()}
    rows = [(m, ext.get(m["id"])) for m in meta]
    rows = [(m, e) for m, e in rows if e and e["triples"]]
    print(f"domande con estrazione riuscita: {len(rows)}/40")

    plans = {p["q"]: p for p in get_plans([m["question"] for m, _ in rows])}
    print(f"piani: {len(plans)}")

    # --- AUDIT (prime AUDIT_N in ordine di campionamento) ---
    audit = [audit_symbolic(e["triples"], plans.get(i, {}), m["answer"])
             for i, (m, e) in enumerate(rows[:AUDIT_N])]
    pg = float(np.mean(audit))
    se = float(np.sqrt(max(pg * (1 - pg), 1e-9) / len(audit)))

    # --- CONTRATTO pre-query ---
    loads = [2 * len(e["triples"]) for _, e in rows]
    ms = [len({str(t[0]) for t in e["triples"]}
              | {str(t[2]) for t in e["triples"]}) for _, e in rows]
    pr = float(np.mean([predicted_accuracy(n, D, max(m_, 2)) ** 1
                        for n, m_ in zip(loads, ms)]))
    contract = pg * pr
    half = 1.96 * se * pr + 0.02
    print(f"\nCONTRATTO (pre-query): Pg={pg:.2f}±{1.96*se:.2f} (n={len(audit)}) "
          f"× Pr={pr:.3f} = {contract:.0%} ± {half:.0%}")

    # --- ESECUZIONE ---
    ok, details = 0, []
    for i, (m, e) in enumerate(rows):
        mem = build_memory(e["triples"])
        ents = sorted({str(t[0]) for t in e["triples"]}
                      | {str(t[2]) for t in e["triples"]})
        ans = execute(mem, ents, plans.get(i, {}))
        hit = bool(ans and match(ans, m["answer"]))
        ok += hit
        details.append({"i": i, "q": m["question"], "gold": m["answer"],
                        "ans": ans, "hit": hit})
    acc = ok / len(rows)
    err = abs(acc - contract)
    print(f"MISURA: {ok}/{len(rows)} = {acc:.0%}")
    print(f"ERRORE DEL CONTRATTO: {err:.1%} → "
          f"{'RISPETTATO' if err <= half else 'VIOLATO'} "
          f"(CI dichiarato ±{half:.0%})")

    json.dump({"pg_audit": pg, "se": se, "pr_theory": pr,
               "contract": contract, "ci": half, "measured": acc,
               "n": len(rows), "respected": bool(err <= half),
               "details": details},
              open(HERE / "results.json", "w"), indent=1)
    print("→ examples/pilot_openrouter/results.json")
