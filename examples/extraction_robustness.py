"""
extraction_robustness.py — Law VIII quantitativa: robustezza all'errore
di estrazione (esperimento (b)).

PREDIZIONI (derivate PRIMA della misura, dalla teoria):
  Tipi 1-3 (missing / wrong relation / wrong entity): il fatto corrotto
      è irraggiungibile o avvelenato per la query → legge MOLTIPLICATIVA
          Acc(ε) = (1-ε)^k · Pr        (k = fatti necessari; qui k=2)
      con Pr = accuracy di reasoning a estrazione pulita.
  Tipo 4 (spurious facts): agisce solo sul CARICO → legge di capacità
          Acc(ε) = p(N·(1+ε))^2        (Law IV, degrado dolce)
  Distinguibilità: tipo 3 produce risposte SBAGLIATE MA CONFIDENTI
      (confidence alta sugli errori); tipi 1-2 falliscono a bassa
      confidence → l'errore di grounding tipo-3 è il più pericoloso
      perché il meno rilevabile.

Setup: reference implementation congelata (reference/abm.py), catene a
2 hop, D=2048, N=120 fatti, 3 seed, ε ∈ {0..0.5}.
Output: extraction_robustness_results.json
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference"))

import numpy as np
from abm import Memory, predicted_accuracy

D, SEEDS, N_CHAINS = 2048, 3, 60
EPS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def build_triples(seed):
    """60 catene a 2 hop = 120 triple 'estratte' (ground truth)."""
    triples, queries = [], []
    for c in range(N_CHAINS):
        x, y, z = f"x{seed}_{c}", f"y{seed}_{c}", f"z{seed}_{c}"
        triples.append((x, "r1", y))
        triples.append((y, "r2", z))
        queries.append((x, z))
    return triples, queries


def corrupt(triples, eps, kind, rng, seed):
    out = []
    for i, (s, r, o) in enumerate(triples):
        if rng.rand() >= eps:
            out.append((s, r, o))
            continue
        if kind == "missing":
            continue                                    # fatto perso
        if kind == "wrong_relation":
            out.append((s, "r_bogus", o))               # relazione errata
        elif kind == "wrong_entity":
            out.append((s, r, f"noise{seed}_{i}"))      # oggetto errato
    if kind == "spurious":
        out = list(triples)
        n_extra = int(eps * len(triples))
        for j in range(n_extra):                        # fatti inventati
            out.append((f"sp{seed}_{j}", f"spr{j % 7}", f"spo{seed}_{j}"))
    return out


def run(kind):
    rows = {}
    for eps in EPS:
        accs, conf_correct, conf_wrong = [], [], []
        for seed in range(SEEDS):
            rng = np.random.RandomState(1000 * seed + int(eps * 100))
            triples, queries = build_triples(seed)
            mem = Memory(D)
            for t in corrupt(triples, eps, kind, rng, seed):
                mem.store(*t)
            ok = 0
            for x, z in queries:
                ans, conf = mem.chain(x, ["r1", "r2"])
                if ans == z:
                    ok += 1
                    conf_correct.append(conf)
                else:
                    conf_wrong.append(conf)
            accs.append(ok / len(queries))
        rows[eps] = {
            "acc": float(np.mean(accs)),
            "conf_ok": float(np.mean(conf_correct)) if conf_correct else 0,
            "conf_err": float(np.mean(conf_wrong)) if conf_wrong else 0,
        }
    return rows


def predictions(pr_clean):
    """Le curve teoriche (nessun parametro fittato oltre Pr misurato a ε=0)."""
    pred = {}
    for kind in ("missing", "wrong_relation", "wrong_entity"):
        pred[kind] = {eps: (1 - eps) ** 2 * pr_clean for eps in EPS}
    n0, m0 = 2 * N_CHAINS, 3 * N_CHAINS + 2
    scale = pr_clean / predicted_accuracy(n0, D, m0) ** 2
    pred["spurious"] = {
        eps: min(1.0, scale) * predicted_accuracy(int(n0 * (1 + eps)), D,
                                                  m0 + int(eps * n0)) ** 2
        for eps in EPS}
    return pred


if __name__ == "__main__":
    print("=" * 70)
    print("  Law VIII quantitativa — Acc(ε) per 4 tipi di errore di estrazione")
    print("=" * 70)
    measured = {k: run(k) for k in
                ("missing", "wrong_relation", "wrong_entity", "spurious")}
    pr_clean = measured["missing"][0.0]["acc"]
    pred = predictions(pr_clean)

    print(f"\n  Pr (reasoning pulito, ε=0) = {pr_clean:.0%}\n")
    print(f"  {'ε':>5} | " + " | ".join(f"{k:>22}" for k in measured))
    print(f"  {'':>5} | " + " | ".join(f"{'mis':>10} {'pred':>10}"
                                       for _ in measured))
    devs = {k: [] for k in measured}
    for eps in EPS:
        cells = []
        for k in measured:
            m, p = measured[k][eps]["acc"], pred[k][eps]
            devs[k].append(abs(m - p))
            cells.append(f"{m:>9.0%} {p:>10.0%}")
        print(f"  {eps:>5.0%} | " + " | ".join(cells))
    print("\n  |dev| media predizione:")
    for k in measured:
        print(f"    {k:15s} {np.mean(devs[k]):.3f}")
    # distinguibilità: confidence degli errori a ε=0.2
    print("\n  Rilevabilità (confidence media delle risposte SBAGLIATE, ε=20%):")
    for k in measured:
        r = measured[k][0.20]
        print(f"    {k:15s} conf_err={r['conf_err']:.2f}  "
              f"(conf_ok={r['conf_ok']:.2f})")
    out = {"measured": {k: {str(e): v for e, v in rows.items()}
                        for k, rows in measured.items()},
           "predicted": {k: {str(e): v for e, v in rows.items()}
                         for k, rows in pred.items()},
           "pr_clean": pr_clean}
    Path("extraction_robustness_results.json").write_text(
        json.dumps(out, indent=2))
    print("\n  → extraction_robustness_results.json")
