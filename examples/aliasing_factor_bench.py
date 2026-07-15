"""
aliasing_factor_bench.py — Aliasing Factor Hypothesis: Acc = p/g.

STATUS: IPOTESI, non legge — la derivazione algebrica è esatta ma
finora era misurata a un solo punto (g=2, crossdomain_bench). Qui la
curva completa: g ∈ {2, 3, 4, 8}.

COSTRUZIONE: un nodo y è oggetto di (g-1) fatti con relazione r e
soggetto di uno: (x_1,r,y) … (x_{g-1},r,y), (y,r,z). Per la simmetria
dell'encoding (ogni fatto è un arco non orientato), query(y,r) vede g
candidati a pari segnale: z corretto e i g-1 predecessori.

PREDIZIONE (scritta prima, zero parametri): Acc(g) = p(N,M)/g, con
p(N,M) dalla Law IV. A carico leggero p≈1 → Acc ≈ 1/g.

CONTRO-VERIFICA: la proiezione tipata sul sottoinsieme non-visitato
(Projection come operatore di DISAMBIGUAZIONE) deve riportare Acc→p
per ogni g: l'alias è simmetria, non rumore, e infatti lo elimina
Projection, non cleanup.

Setup: D=4096, N=100 fatti (star + filler), 30 istanze/g, 5 seed.
Output: aliasing_factor_results.json
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference"))

import numpy as np
from abm import Memory, predicted_accuracy

D, SEEDS, N_STARS, N_TOTAL = 4096, 5, 30, 100
GS = [2, 3, 4, 8]


def run(g, seed):
    mem = Memory(D)
    stars = []
    n_star_facts = 0
    for i in range(N_STARS):
        y, z = f"y{seed}_{i}", f"z{seed}_{i}"
        for j in range(g - 1):                     # archi entranti (alias)
            mem.store(f"x{seed}_{i}_{j}", "r", y)
        mem.store(y, "r", z)                       # arco corretto
        stars.append((y, z, [f"x{seed}_{i}_{j}" for j in range(g - 1)]))
        n_star_facts += g
    for k in range(max(0, N_TOTAL - n_star_facts)):   # filler
        mem.store(f"f{seed}_{k}", f"fr{k % 9}", f"fo{seed}_{k}")

    naive = guided = 0
    for y, z, xs in stars:
        ans, _ = mem.query(y, "r")
        naive += (ans == z)
        sub = [n for n in mem.items._names if n not in set(xs) | {y}]
        ans_g, _ = mem.query(y, "r", subset=sub)
        guided += (ans_g == z)
    return naive / N_STARS, guided / N_STARS, len(mem._facts), len(mem.items)


if __name__ == "__main__":
    print("=" * 72)
    print("  Aliasing Factor Hypothesis: Acc = p/g   (g = 2, 3, 4, 8)")
    print("=" * 72)
    print(f"\n  {'g':>3} {'pred p/g':>9} {'naive (95% CI)':>18} "
          f"{'|dev|':>7} {'guided':>13} {'pred p':>7}")

    results = {}
    for g in GS:
        nv, gd, n, m = [], [], 0, 0
        for seed in range(SEEDS):
            a, b, n, m = run(g, seed)
            nv.append(a); gd.append(b)
        p = predicted_accuracy(n, D, m)
        pred = p / g
        acc, acc_g = float(np.mean(nv)), float(np.mean(gd))
        ci = 1.96 * float(np.std(nv, ddof=1) / np.sqrt(SEEDS))
        dev = abs(acc - pred)
        results[g] = {"pred": round(pred, 3), "measured": round(acc, 3),
                      "ci95": round(ci, 3), "dev": round(dev, 3),
                      "guided": round(acc_g, 3), "p_theory": round(p, 3),
                      "n": n, "m": m}
        print(f"  {g:>3} {pred:>8.0%} {acc:>11.0%} ±{ci:>4.0%} "
              f"{dev:>7.3f} {acc_g:>12.0%} {p:>6.0%}")

    devs = [r["dev"] for r in results.values()]
    print(f"\n  |dev| media: {np.mean(devs):.3f} (max {max(devs):.3f})")
    print("  guided ≈ p per ogni g ⇒ l'alias è SIMMETRIA (la elimina "
          "Projection),\n  non rumore (il cleanup da solo non può).")
    Path("aliasing_factor_results.json").write_text(
        json.dumps(results, indent=2))
    print("\n  → aliasing_factor_results.json")
