"""
industrial_pilot.py — Il benchmark decisivo, versione pilota:

    Documenti → Compiler → Triple → ABM → 1000 query → Inspector
                                                → contratto rispettato?

Il compiler è PLUGGABLE: di default un estrattore deterministico a
template (con lacune di copertura REALI, non tassi iniettati: alcune
frasi usano fraseggi che l'estrattore non copre o mis-interpreta).
Per la replica con LLM reali: passare un JSON di triple estratte con
--extraction file.json (stesso formato: [[s, r, o], ...]).

PROTOCOLLO (l'ordine è il punto):
  1. Si genera/legge il corpus (500 frasi da 250 catene gold a 2 hop,
     4 fraseggi di cui 1 non coperto e 1 trappola).
  2. Il compiler estrae le triple. NESSUN accesso al gold.
  3. AUDIT su un campione di 40 catene (come farebbe un cliente):
     stima di E_q[Pg] a livello di catena — la forma per-query.
  4. L'Inspector emette il CONTRATTO: accuratezza proiettata, pressione,
     aliasing, collo di bottiglia dominante. PRIMA di qualsiasi query.
  5. Si eseguono 1000 query a 2 hop e si verifica il contratto.

La metrica finale è UNA: |contratto − misurato|.

Output: industrial_pilot_results.json
"""

import sys, json, re, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference"))

import numpy as np
from abm import Memory, capacity
from inspector import stats

N_CHAINS, N_QUERIES, AUDIT = 250, 1000, 40
SEED = 20260715


# --- 1. il corpus: manualistica sintetica ma testuale ----------------------

TEMPLATES = [
    "The {x} module requires the {y} service to start.",        # coperto
    "{x} depends on {y} for its operation.",                    # coperto
    "Before enabling {x}, operators must configure {y}.",       # NON coperto
    "It is {y} that the {x} subsystem calls first.",            # trappola
]
T2 = [
    "The {y} service writes its state to {z}.",                 # coperto
    "{y} stores all runtime data in {z}.",                      # coperto
    "Data from {y} eventually reaches {z} storage.",            # NON coperto
    "It is {z} which receives the output of {y}.",              # trappola
]


def build_corpus(rng):
    docs, gold = [], []
    for c in range(N_CHAINS):
        x, y, z = f"mod_{c}", f"svc_{c}", f"store_{c}"
        gold.append((x, y, z))
        docs.append(TEMPLATES[rng.randint(4)].format(x=x, y=y))
        docs.append(T2[rng.randint(4)].format(y=y, z=z))
    return docs, gold


# --- 2. il compiler di default (deterministico, imperfetto per davvero) ----

PATTERNS = [
    (re.compile(r"The (\S+) module requires the (\S+) service"),
     "requires", (1, 2)),
    (re.compile(r"(\S+) depends on (\S+) for"), "requires", (1, 2)),
    (re.compile(r"The (\S+) service writes its state to (\S+)\."),
     "writes_to", (1, 2)),
    (re.compile(r"(\S+) stores all runtime data in (\S+)\."),
     "writes_to", (1, 2)),
    # trappole: il regex inverte gli argomenti (bug di parsing realistico)
    (re.compile(r"It is (\S+) that the (\S+) subsystem"),
     "requires", (1, 2)),                     # gold: (2)->x requires (1)->y
    (re.compile(r"It is (\S+) which receives the output of (\S+)\."),
     "writes_to", (1, 2)),                    # gold: (2) writes_to (1)
]


def default_compiler(docs):
    triples = []
    for sent in docs:
        for pat, rel, (a, b) in PATTERNS:
            m = pat.search(sent)
            if m:
                triples.append((m.group(a), rel, m.group(b)))
                break
    return triples


# --- il resto della pipeline ------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extraction", help="JSON [[s,r,o],...] da LLM reale")
    args = ap.parse_args()

    rng = np.random.RandomState(SEED)
    docs, gold = build_corpus(rng)
    print("=" * 72)
    print("  Industrial pilot — Documenti → Compiler → ABM → Inspector → 1000 query")
    print("=" * 72)
    print(f"\n  Corpus: {len(docs)} frasi, {N_CHAINS} catene gold")

    if args.extraction:
        triples = [tuple(t) for t in
                   json.loads(Path(args.extraction).read_text())]
        compiler_name = args.extraction
    else:
        triples = default_compiler(docs)
        compiler_name = "template-extractor (default)"
    print(f"  Compiler: {compiler_name} → {len(triples)} triple estratte")

    # --- 3. audit per-query su un campione (senza usare tutto il gold) ---
    # AUDIT ALGEBRICO: membership a meno della simmetria degli archi.
    # (Il primo run, conservato nel report, usava l'audit simbolico
    # direction-blind: contratto 10% vs 57% misurato — VIOLATO. Il
    # runtime è direction-insensitive: (o, r, s) risponde quanto
    # (s, r, o). L'audit deve misurare ciò che la memoria vede.)
    tset = set(triples)

    def member(s_, r_, o_):
        return (s_, r_, o_) in tset or (o_, r_, s_) in tset

    audit_idx = rng.choice(N_CHAINS, AUDIT, replace=False)
    ok_chain = sum(member(gold[i][0], "requires", gold[i][1]) and
                   member(gold[i][1], "writes_to", gold[i][2])
                   for i in audit_idx)
    pg_query = ok_chain / AUDIT
    print(f"  Audit algebrico ({AUDIT} catene): E_q[Pg] stimato = "
          f"{pg_query:.2f}")

    # --- dimensionamento dalla teoria e ingestione ---
    m_est = len({t[0] for t in triples} | {t[2] for t in triples}) + 2
    dim = 1024
    while len(triples) > 0.5 * capacity(dim, m_est) and dim < 2 ** 20:
        dim *= 2
    mem = Memory(dim)
    t0 = time.perf_counter()
    for t in triples:
        mem._facts.append(mem.fact_hv(*t))
    from abm import bundle
    mem._trace = bundle(mem._facts)
    t_ingest = time.perf_counter() - t0

    # --- 4. il CONTRATTO, prima di qualsiasi query ---
    queries_plan = [(g[0], ["requires", "writes_to"]) for g in gold]
    s = stats(mem, extractor_precision=np.sqrt(pg_query),
              triples=triples, queries=queries_plan)
    contract_acc = round(pg_query * s["expected_accuracy"]
                         * s["aliasing_factor"], 3)
    # il contratto dichiara la PROPRIA incertezza: l'unico termine
    # stimato è Pg (audit binomiale su n catene) — tutto il resto è teoria
    se_pg = np.sqrt(max(pg_query * (1 - pg_query), 1e-9) / AUDIT)
    half = 1.96 * se_pg * s["expected_accuracy"] * s["aliasing_factor"]
    print(f"\n  CONTRATTO (pre-query): D={dim}, pressione={s['pressure']}, "
          f"aliasing={s['aliasing_factor']}")
    print(f"    accuratezza proiettata = Pg×Pr×alias = {pg_query:.2f} × "
          f"{s['expected_accuracy']} × {s['aliasing_factor']} "
          f"= {contract_acc:.0%} ± {half:.0%} (95%, da audit n={AUDIT})")
    print(f"    collo di bottiglia dominante: "
          f"{'grounding' if pg_query < s['expected_accuracy'] else 'capacity'}")

    # --- 5. 1000 query e verifica ---
    qi = rng.randint(0, N_CHAINS, N_QUERIES)
    t0 = time.perf_counter()
    ok = sum(mem.chain(gold[i][0], ["requires", "writes_to"])[0]
             == gold[i][2] for i in qi)
    t_query = (time.perf_counter() - t0) / N_QUERIES
    measured = ok / N_QUERIES
    err = abs(measured - contract_acc)
    inside = abs(measured - contract_acc) <= half + 0.02
    print(f"\n  MISURA: {N_QUERIES} query → {measured:.1%}  "
          f"(latenza {1000 * t_query:.2f} ms/query, "
          f"ingestione {t_ingest:.2f}s)")
    print(f"  ERRORE DEL CONTRATTO: {err:.1%}  → "
          f"{'RISPETTATO' if inside else 'VIOLATO'} "
          f"(entro il CI dichiarato ± {half:.0%} + 2% teoria)")

    out = {"compiler": compiler_name, "n_triples": len(triples),
           "pg_query_audit": pg_query, "dimension": dim,
           "trace_bytes": dim // 8,
           "inspector": s, "contract_accuracy": contract_acc,
           "measured_accuracy": measured, "contract_error": round(err, 3), "contract_ci95": round(half, 3),
           "query_ms": round(1000 * t_query, 2),
           "ingest_s": round(t_ingest, 2)}
    Path("industrial_pilot_results.json").write_text(
        json.dumps(out, indent=2))
    print("\n  → industrial_pilot_results.json")
