"""
multicompiler_bench.py — Esperimento multi-compilatore (dry-run).

DOMANDA: la Law VIII (Resource Composition Law, forma per-query)
cattura il contributo del Livello A al punto da permettere all'Inspector
di PREVEDERE IL RANKING di compilatori diversi prima di qualsiasi query?

Tre "compilatori" simulati con profili d'errore realistici e distinti,
stesso corpus ground-truth (60 catene a 2 hop, 3 seed, D=2048):

  alpha  "uniforme"     wrong_entity i.i.d. ε=8%   (precisione media 92%)
  beta   "a cluster"    intere catene corrotte, 20% (precisione media 80%)
  gamma  "recall-tuned" missing 10% + spurious 30%  (precisione media 90%,
                                                     ma carica la memoria)

PREDIZIONI (scritte PRIMA della misura, dai soli parametri di profilo +
Pr_clean calibrato a ε=0 — nessun parametro fittato per compilatore):

  Acc(c) = E_q[Pg(q)] × Pr(N_eff, M_eff)          (forma per-query)

  con E_q[Pg]:  alpha (1-0.08)² = 0.846
                beta  1-0.20    = 0.800   (danno concentrato: query o
                                           intatta o persa)
                gamma (1-0.10)² = 0.810
  e il fattore di carico p(N_eff, M_eff)/p(N0, M0) per hop (solo gamma
  cambia il carico in modo sostanziale).

  Il RANKING PREVISTO è quello calcolato dai contratti (stampato prima
  della misura). Nota per-query vs media: per beta la forma media
  prevede 0.80²=0.64 di grounding, la per-query 0.80 — se beta risale
  sopra la forma media, solo la per-query spiega i dati.

FALSIFICATORE: se il ranking misurato non coincide con quello previsto
per-query SU COPPIE RISOLVIBILI (contratti separati da più dell'errore
statistico della misura), la Law VIII non cattura il Livello A.

Output: multicompiler_results.json
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference"))

import numpy as np
from abm import Memory, predicted_accuracy
from inspector import stats

D, SEEDS, N_CHAINS = 2048, 10, 60
N0 = 2 * N_CHAINS                       # triple ground-truth
M0 = 3 * N_CHAINS + 2                   # entità + 2 relazioni


def build_corpus(seed):
    triples, queries = [], []
    for c in range(N_CHAINS):
        x, y, z = f"x{seed}_{c}", f"y{seed}_{c}", f"z{seed}_{c}"
        triples += [(x, "r1", y), (y, "r2", z)]
        queries.append((x, z))
    return triples, queries


# --- i tre compilatori simulati -------------------------------------------

def compile_alpha(triples, rng, seed):
    """Uniforme: wrong_entity i.i.d. all'8% dei fatti."""
    out = []
    for i, (s, r, o) in enumerate(triples):
        if rng.rand() < 0.08:
            out.append((s, r, f"nA{seed}_{i}"))
        else:
            out.append((s, r, o))
    return out


def compile_beta(triples, rng, seed):
    """A cluster: il 20% dei DOCUMENTI (catene) esce interamente corrotto."""
    bad_chains = set(np.where(rng.rand(N_CHAINS) < 0.20)[0])
    out = []
    for i, (s, r, o) in enumerate(triples):
        if i // 2 in bad_chains:
            out.append((s, r, f"nB{seed}_{i}"))
        else:
            out.append((s, r, o))
    return out


def compile_gamma(triples, rng, seed):
    """Recall-tuned: perde il 10% dei fatti ma ne inventa il 30% in più."""
    out = [t for t in triples if rng.rand() >= 0.10]
    for j in range(int(0.30 * len(triples))):
        out.append((f"gs{seed}_{j}", f"gr{j % 7}", f"go{seed}_{j}"))
    return out


COMPILERS = {"alpha": compile_alpha, "beta": compile_beta,
             "gamma": compile_gamma}

# profili dichiarati (in un caso reale: audit su un campione di triple)
PROFILE = {
    "alpha": dict(pg_query=(1 - 0.08) ** 2, mean_prec=0.92,
                  n_eff=N0 + 0,            m_eff=M0 + int(0.08 * N0)),
    "beta":  dict(pg_query=1 - 0.20,       mean_prec=0.80,
                  n_eff=N0,                m_eff=M0 + int(0.20 * N0)),
    "gamma": dict(pg_query=(1 - 0.10) ** 2, mean_prec=0.90,
                  n_eff=int(0.90 * N0) + int(0.30 * N0),
                  m_eff=M0 + 2 * int(0.30 * N0) + 7),
}


def predict(pr_clean):
    """Contratti pre-query. Due forme a confronto: per-query vs media."""
    p0 = predicted_accuracy(N0, D, M0)
    pred = {}
    for name, pf in PROFILE.items():
        load = (predicted_accuracy(pf["n_eff"], D, pf["m_eff"]) / p0) ** 2
        pred[name] = {
            "per_query": pf["pg_query"] * pr_clean * load,
            "mean_form": pf["mean_prec"] ** 2 * pr_clean * load,
        }
    return pred


def measure(name):
    accs = []
    for seed in range(SEEDS):
        rng = np.random.RandomState(7000 + 13 * seed)
        triples, queries = build_corpus(seed)
        mem = Memory(D)
        for t in COMPILERS[name](triples, rng, seed):
            mem.store(*t)
        ok = sum(mem.chain(x, ["r1", "r2"])[0] == z for x, z in queries)
        accs.append(ok / len(queries))
    sem = float(np.std(accs, ddof=1) / np.sqrt(SEEDS))
    return float(np.mean(accs)), sem, mem   # ultima memoria per l'Inspector


if __name__ == "__main__":
    print("=" * 72)
    print("  Multi-compilatore: l'Inspector prevede il ranking prima delle query?")
    print("=" * 72)

    # calibrazione: Pr a estrazione pulita (una sola misura, condivisa)
    clean_accs = []
    for seed in range(SEEDS):
        triples, queries = build_corpus(seed)
        mem = Memory(D)
        for t in triples:
            mem.store(*t)
        ok = sum(mem.chain(x, ["r1", "r2"])[0] == z for x, z in queries)
        clean_accs.append(ok / len(queries))
    pr_clean = float(np.mean(clean_accs))
    print(f"\n  Pr_clean (calibrazione, ε=0) = {pr_clean:.0%}")

    pred = predict(pr_clean)
    rank_pq = sorted(pred, key=lambda c: -pred[c]["per_query"])
    rank_mf = sorted(pred, key=lambda c: -pred[c]["mean_form"])
    print("\n  CONTRATTI PRE-QUERY (predizioni scritte prima):")
    for c in pred:
        print(f"    {c:6s} per-query={pred[c]['per_query']:.0%}   "
              f"forma-media={pred[c]['mean_form']:.0%}")
    print(f"  Ranking previsto (per-query):   {' > '.join(rank_pq)}")
    print(f"  Ranking previsto (forma media): {' > '.join(rank_mf)}")

    print("\n  MISURA:")
    measured, sems, inspections = {}, {}, {}
    for c in COMPILERS:
        acc, sem, mem = measure(c)
        measured[c], sems[c] = acc, sem
        inspections[c] = stats(mem, extractor_precision=PROFILE[c]["mean_prec"])
        print(f"    {c:6s} acc={acc:.0%} ±{1.96 * sem:.0%} (95% CI)   "
              f"|dev| per-query={abs(acc - pred[c]['per_query']):.3f}   "
              f"|dev| media={abs(acc - pred[c]['mean_form']):.3f}")
    rank_obs = sorted(measured, key=lambda c: -measured[c])
    print(f"\n  Ranking osservato:              {' > '.join(rank_obs)}")
    print(f"  per-query indovina il ranking:  {rank_obs == rank_pq}")
    print(f"  forma media indovina il ranking: {rank_obs == rank_mf}")
    # coppie risolvibili: differenza misurata oltre l'errore combinato
    print("\n  Coppie risolvibili (|Δ misurato| > 1.96·SE combinato):")
    names = list(measured)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            delta = measured[a] - measured[b]
            se = 1.96 * np.sqrt(sems[a] ** 2 + sems[b] ** 2)
            resolvable = abs(delta) > se
            pq_ok = (np.sign(pred[a]["per_query"] - pred[b]["per_query"])
                     == np.sign(delta)) if resolvable else None
            print(f"    {a} vs {b}: Δ={delta:+.0%} (soglia {se:.0%}) "
                  f"risolvibile={resolvable}"
                  + (f"  per-query concorde={pq_ok}" if resolvable else ""))

    out = {"pr_clean": float(pr_clean), "predicted": pred,
           "measured": measured, "sem": sems,
           "ranking": {"per_query": rank_pq, "mean_form": rank_mf,
                       "observed": rank_obs},
           "inspector": inspections}
    Path("multicompiler_results.json").write_text(json.dumps(out, indent=2))
    print("\n  → multicompiler_results.json")
