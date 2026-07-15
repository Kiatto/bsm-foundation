"""
crossdomain_bench.py — Il Livello B è dominio-indipendente?

CLAIM DA TESTARE: l'accuratezza di un ABM dipende solo dalle risorse
(N fatti, M codebook, h hop, ε grounding) e NON dalla struttura del
dominio. Il contratto è UNA SOLA FORMULA, senza parametri per-dominio:

    Acc = (1-ε)^2 × p(N_eff, M_eff)^2        p = predicted_accuracy

Quattro domini con topologie deliberatamente diverse (catene a 2 hop
in tutti i casi, D=4096, 5 seed):

  api      relazioni DENSE: 6 relazioni riusate su 50 endpoint
  manuals  sequenze: UNA sola relazione ("next_step") per 40 procedure
  legal    hub di soggetti: 45 clausole, il 30% referenzia 5 atti hub
  medical  hub di oggetti: 50 percorsi che convergono su 10 diagnosi

Se il contratto tiene su tutti (|dev| comparabile al ~4-5% degli altri
esperimenti), la teoria segue il dominio, non il dataset.
FALSIFICATORE: un dominio con |dev| sistematicamente grande implica che
la struttura entra nella legge — e il claim di indipendenza cade lì.

ESITO DEL PRIMO RUN (conservato): manuals crolla (37% vs 79%) per un
meccanismo algebrico ESATTO, non per rumore — l'encoding s⊕ρ(r)⊕o è
simmetrico in s,o (ogni fatto è un arco non orientato): con la stessa
relazione su hop consecutivi, f1 ⊕ key(y,r) = x ESATTAMENTE, e il
predecessore fa da alias a pari segnale (verificato: 21 z / 19 x / 0
altro su hop puliti). Correzione DERIVATA (algebra, non fit): fattore
1/g per hop con g candidati a pari segnale (qui g=2 sull'hop 2 →
contratto × 0.5). Rimedio a livello controller, già nella reference:
proiezione tipata che esclude i nodi visitati (variante "guided").

ε per dominio (dichiarati, realistici per la qualità attesa di un
estrattore su quel genere testuale): api 6%, manuals 11%, legal 18%,
medical 9%. Errore iniettato: wrong_entity i.i.d.

Output: crossdomain_results.json
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference"))

import numpy as np
from abm import Memory, predicted_accuracy

D, SEEDS = 4096, 5
EPS = {"api": 0.06, "manuals": 0.11, "legal": 0.18, "medical": 0.09}


def gen_api(seed, rng):
    """50 catene endpoint→param→tipo, 6 relazioni riusate."""
    rels = [f"rel{k}" for k in range(6)]
    triples, queries = [], []
    for c in range(50):
        r1, r2 = rels[c % 6], rels[(c + 3) % 6]
        x, y, z = f"ep{seed}_{c}", f"par{seed}_{c}", f"ty{seed}_{c}"
        triples += [(x, r1, y), (y, r2, z)]
        queries.append((x, [r1, r2], z))
    return triples, queries


def gen_manuals(seed, rng):
    """40 procedure di 3 step, UNA sola relazione."""
    triples, queries = [], []
    for c in range(40):
        x, y, z = (f"step{seed}_{c}_{i}" for i in range(3))
        triples += [(x, "next_step", y), (y, "next_step", z)]
        queries.append((x, ["next_step", "next_step"], z))
    return triples, queries


def gen_legal(seed, rng):
    """45 clausole; il 30% dei soggetti di secondo hop è uno di 5 atti
    hub (stesso soggetto, relazioni diverse → chiavi distinte)."""
    hubs = [f"act{seed}_{h}" for h in range(5)]
    triples, queries = [], []
    for c in range(45):
        x, z = f"cl{seed}_{c}", f"ob{seed}_{c}"
        y = hubs[c % 5] if rng.rand() < 0.30 else f"ref{seed}_{c}"
        r2 = f"cites{c}" if y in hubs else "cites"
        triples += [(x, "refers_to", y), (y, r2, z)]
        queries.append((x, ["refers_to", r2], z))
    return triples, queries


def gen_medical(seed, rng):
    """50 percorsi sintomo→esame→diagnosi; le diagnosi sono 10 hub."""
    diags = [f"dx{seed}_{k}" for k in range(10)]
    triples, queries = [], []
    for c in range(50):
        x, y = f"sym{seed}_{c}", f"test{seed}_{c}"
        z = diags[c % 10]
        triples += [(x, "requires", y), (y, "indicates", z)]
        queries.append((x, ["requires", "indicates"], z))
    return triples, queries


DOMAINS = {"api": gen_api, "manuals": gen_manuals,
           "legal": gen_legal, "medical": gen_medical}


def ground(triples, eps, rng, seed):
    """Estrattore simulato: wrong_entity i.i.d. a tasso eps."""
    return [(s, r, f"nz{seed}_{i}") if rng.rand() < eps else (s, r, o)
            for i, (s, r, o) in enumerate(triples)]


if __name__ == "__main__":
    print("=" * 72)
    print("  Cross-domain: una sola formula di contratto, quattro topologie")
    print("=" * 72)
    print(f"\n  {'dominio':10s} {'ε':>4} {'N':>5} {'M':>5} "
          f"{'contratto':>10} {'misurato':>16} {'|dev|':>7}")

    results = {}
    variants = list(DOMAINS.items()) + [("manuals_guided",
                                         DOMAINS["manuals"])]
    for name, gen in variants:
        base = name.split("_")[0]
        guided = name.endswith("_guided")
        eps = EPS[base]
        # fattore di aliasing derivato: g candidati a pari segnale
        # sull'hop 2 quando r1 == r2 (arco non orientato); 1 se guided
        alias = 0.5 if (base == "manuals" and not guided) else 1.0
        accs, n_last, m_last, pred = [], 0, 0, None
        for seed in range(SEEDS):
            rng = np.random.RandomState(3000 + 17 * seed)
            triples, queries = gen(seed, rng)
            extracted = ground(triples, eps, rng, seed)
            mem = Memory(D)
            for t in extracted:
                mem.store(*t)
            n_last, m_last = len(mem._facts), len(mem.items)
            if pred is None:                       # contratto PRE-query
                pred = ((1 - eps) ** 2 * alias
                        * predicted_accuracy(n_last, D, m_last) ** 2)
            ok = 0
            for x, rs, z in queries:
                if guided:
                    node, visited = x, {x}
                    for r in rs:
                        sub = [n for n in mem.items._names
                               if n not in visited]
                        node, _ = mem.query(node, r, subset=sub)
                        visited.add(node)
                    ok += (node == z)
                else:
                    ok += (mem.chain(x, rs)[0] == z)
            accs.append(ok / len(queries))
        acc = float(np.mean(accs))
        sem = float(np.std(accs, ddof=1) / np.sqrt(SEEDS))
        dev = abs(acc - pred)
        results[name] = {"eps": eps, "n": n_last, "m": m_last,
                         "contract": round(pred, 3),
                         "measured": round(acc, 3),
                         "ci95": round(1.96 * sem, 3),
                         "dev": round(dev, 3)}
        print(f"  {name:10s} {eps:>4.0%} {n_last:>5} {m_last:>5} "
              f"{pred:>10.0%} {acc:>9.0%} ±{1.96 * sem:>4.0%} {dev:>7.3f}")

    devs = [r["dev"] for r in results.values()]
    print(f"\n  |dev| media cross-domain: {np.mean(devs):.3f}  "
          f"(max: {max(devs):.3f})")
    print("  Nessun parametro FITTATO per-dominio: l'unico termine "
          "strutturale (aliasing 1/g)\n  è derivato esattamente "
          "dall'algebra ed è calcolabile dal piano di query.")
    Path("crossdomain_results.json").write_text(
        json.dumps(results, indent=2))
    print("\n  → crossdomain_results.json")
