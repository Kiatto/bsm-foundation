"""Fase 1 (PROTOCOL.md): un compilatore, riga completa della tabella.

    OR_KEY=... python run_compiler.py <model_id> <label>

Congelati: le 33 domande (phase1_question_ids.json), planner v1
(plans_by_id.json), esecutore/audit di evaluate.py, D=8192.
Estrae con <model_id> (question-blind), poi: audit → contratto →
misura → appende la riga a phase1_rows.json.
"""

import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract import sample, call, PROMPT, RELS
from evaluate import (build_memory, execute, audit_symbolic, match,
                      predicted_accuracy, D, AUDIT_N)
from inspector import aliasing as inspector_aliasing  # noqa: E402
sys.path.insert(0, str(HERE.parent.parent / "reference"))


def extract_all(model, label, questions):
    out = HERE / f"extractions_{label}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.open()
                if json.loads(l)["triples"]}
    for k, r in enumerate(questions):
        if r["id"] in done:
            continue
        ctx = r["context"]
        paras = "\n\n".join(f"[{ti}] " + " ".join(s)
                            for ti, s in zip(ctx["title"], ctx["sentences"]))
        prompt = PROMPT.format(rels=RELS, paras=paras[:12000])
        triples, err = None, None
        for attempt in range(3):
            try:
                txt = call(model, prompt)
                s = txt[txt.find("["):txt.rfind("]") + 1]
                triples = [t for t in json.loads(s)
                           if isinstance(t, list) and len(t) == 3]
                break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"[:100]
                time.sleep(25 * (attempt + 1))
        with out.open("a") as f:
            f.write(json.dumps({"id": r["id"], "triples": triples,
                                "error": None if triples else err}) + "\n")
        print(f"  [{k+1}/{len(questions)}] "
              f"{len(triples) if triples else 'FAIL:'+str(err)}", flush=True)
        time.sleep(5)
    return {json.loads(l)["id"]: json.loads(l)["triples"]
            for l in out.open() if json.loads(l)["triples"]}


if __name__ == "__main__":
    model, label = sys.argv[1], sys.argv[2]
    qids = json.loads((HERE / "phase1_question_ids.json").read_text())
    plans = json.loads((HERE / "plans_by_id.json").read_text())
    allq = {r["id"]: r for r in sample()}
    questions = [allq[q] for q in qids]

    print(f"Compilatore {label} = {model}", flush=True)
    ext = extract_all(model, label, questions)
    rows = [(allq[q], ext[q]) for q in qids if q in ext]
    print(f"estrazioni riuscite: {len(rows)}/{len(qids)}")

    # audit sulle prime AUDIT_N (ordine congelato)
    audit = [audit_symbolic(t, plans.get(m["id"], {}), m["answer"])
             for m, t in rows[:AUDIT_N]]
    pg = float(np.mean(audit))
    se = float(np.sqrt(max(pg * (1 - pg), 1e-9) / len(audit)))

    # metriche di risorsa (per la tabella estesa)
    loads = [2 * len(t) for _, t in rows]
    ms = [len({str(x[0]) for x in t} | {str(x[2]) for x in t})
          for _, t in rows]
    from abm import capacity
    pr = float(np.mean([predicted_accuracy(n, D, max(mm, 2))
                        for n, mm in zip(loads, ms)]))
    pressure = float(np.mean([n / capacity(D, max(mm, 2))
                              for n, mm in zip(loads, ms)]))
    alias = float(np.mean(
        [inspector_aliasing([tuple(map(str, x)) for x in t],
                            [(plans.get(m["id"], {}).get("anchor", ""),
                              [plans.get(m["id"], {}).get("relation", "")])]
                            )["aliasing_factor"] for m, t in rows]))
    contract = pg * pr * alias
    half = 1.96 * se * pr * alias + 0.02

    ok = 0
    for m, t in rows:
        mem = build_memory(t)
        ents = sorted({str(x[0]) for x in t} | {str(x[2]) for x in t})
        ans = execute(mem, ents, plans.get(m["id"], {}))
        ok += bool(ans and match(ans, m["answer"]))
    acc = ok / len(rows)

    row = {"label": label, "model": model, "n": len(rows),
           "pg": round(pg, 3), "pg_ci": round(1.96 * se, 3),
           "n_eff_mean": round(float(np.mean(loads)), 1),
           "m_mean": round(float(np.mean(ms)), 1),
           "pressure": round(pressure, 3), "aliasing": round(alias, 3),
           "pr": round(pr, 3), "contract": round(contract, 3),
           "ci": round(half, 3), "measured": round(acc, 3),
           "error": round(abs(acc - contract), 3),
           "respected": bool(abs(acc - contract) <= half)}
    tab = HERE / "phase1_rows.json"
    data = json.loads(tab.read_text()) if tab.exists() else []
    data = [r for r in data if r["label"] != label] + [row]
    tab.write_text(json.dumps(data, indent=1))
    print(json.dumps(row, indent=1))
