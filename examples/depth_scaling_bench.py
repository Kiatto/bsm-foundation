"""
depth_scaling_bench.py — Test della Proposizione 3.6:
"la profondità di ragionamento è esponenzialmente economica".

Predizione (derivata PRIMA dell'esperimento, FORMALISM.md §3.6):
    D_min(h) = Θ(N·ln M) + Θ(N·ln h)
cioè il D minimo per accuracy di catena ≥95% cresce ~logaritmicamente
con h — non linearmente.

Design chiave: CARICO COSTANTE. Ogni memoria contiene sempre N=120
fatti; catene più lunghe ⇒ meno catene per memoria ⇒ più memorie
indipendenti, finché ogni h ha ≥60 catene di prova. Così la profondità
è isolata dal carico (la tesi: la risorsa limitante è il carico, non h).

Esperimento 2: D fisso (1024), h crescente → Acc(h) confrontata con
p̂^h (p̂ misurato per-hop sulle stesse memorie). Verifica congiunta di
Law V + Prop 3.6.

Output: depth_scaling_results.json
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from bsm.memory.vsa import WorkingMemory

N_LOAD = 120          # fatti per memoria, SEMPRE
MIN_TRIALS = 60       # catene di prova minime per punto
SEEDS = 3
TARGET = 0.95


def chain_accuracy(d, h, seed_base):
    """Accuracy end-to-end e per-hop, a carico costante N_LOAD."""
    per_mem = max(1, N_LOAD // h)          # catene per memoria
    n_mems = int(np.ceil(MIN_TRIALS / per_mem))
    e2e_ok = e2e_tot = hop_ok = hop_tot = 0
    for m in range(n_mems):
        wm = WorkingMemory(d)
        chains = []
        for c in range(per_mem):
            nodes = [f"n{seed_base}_{m}_{c}_{k}" for k in range(h + 1)]
            for k in range(h):
                wm.store(nodes[k], f"r{seed_base}_{m}_{k}", nodes[k + 1])
            chains.append(nodes)
        # riempi fino a N_LOAD con fatti di zavorra (carico costante)
        for j in range(N_LOAD - per_mem * h):
            wm.store(f"pad{seed_base}_{m}_{j}", f"pr{j}",
                     f"po{seed_base}_{m}_{j}")
        for nodes in chains:
            for k in range(h):
                hop_tot += 1
                hop_ok += (wm.query(nodes[k], f"r{seed_base}_{m}_{k}")[0]
                           == nodes[k + 1])
            cur = nodes[0]
            for k in range(h):
                cur, _ = wm.query(cur, f"r{seed_base}_{m}_{k}")
            e2e_tot += 1
            e2e_ok += (cur == nodes[-1])
    return e2e_ok / e2e_tot, hop_ok / hop_tot


def find_dmin(h, seed):
    """Minimo D (griglia geometrica ×1.25 da 256) con accuracy ≥ TARGET."""
    d = 256
    while d <= 16384:
        acc, _ = chain_accuracy(int(d), h, seed)
        if acc >= TARGET:
            return int(d)
        d *= 1.25
    return None


def fig1_dmin():
    print(f"\n[Fig. 1] D_min(h) per accuracy di catena ≥{TARGET:.0%} "
          f"(carico costante N={N_LOAD})")
    print(f"  {'h':>4} {'D_min (media±sd)':>18} {'D_min/ln(h+1)':>14}")
    out = {}
    for h in (1, 2, 4, 8, 16, 32, 64):
        dmins = [find_dmin(h, s) for s in range(SEEDS)]
        dmins = [d for d in dmins if d]
        m, sd = np.mean(dmins), np.std(dmins)
        out[h] = {"dmin": float(m), "sd": float(sd)}
        print(f"  {h:>4} {m:>12.0f} ± {sd:<5.0f} {m/np.log(h+1):>12.0f}")
    hs = np.array(list(out))
    ds = np.array([out[h]["dmin"] for h in out])
    # fit lineare vs logaritmico
    for label, X in (("lineare D=a·h+b", hs.astype(float)),
                     ("log D=a·ln(h)+b", np.log(hs))):
        A = np.vstack([X, np.ones_like(X)]).T
        coef, res, *_ = np.linalg.lstsq(A, ds, rcond=None)
        pred = A @ coef
        r2 = 1 - np.sum((ds - pred) ** 2) / np.sum((ds - ds.mean()) ** 2)
        print(f"  fit {label:18s} R² = {r2:.3f}")
    # rapporto di crescita: D(64)/D(1) — lineare predirebbe ~64×
    print(f"  crescita D_min(64)/D_min(1) = "
          f"{out[64]['dmin']/out[1]['dmin']:.2f}×  "
          f"(lineare ⇒ ~64×, teoria ⇒ piccolo)")
    return out


def fig2_fixed_d(d=1024):
    print(f"\n[Fig. 2] D={d} fisso, carico costante: Acc(h) vs p̂^h")
    print(f"  {'h':>4} {'Acc e2e':>8} {'p̂':>7} {'p̂^h':>7}")
    out = {}
    for h in (1, 2, 3, 4, 6, 8, 12, 16, 24):
        accs, preds = [], []
        for s in range(SEEDS):
            acc, p = chain_accuracy(d, h, 100 + s)
            accs.append(acc)
            preds.append(p ** h)
        a, pr = np.mean(accs), np.mean(preds)
        p_mean = np.mean([chain_accuracy(d, h, 100 + s)[1]
                          for s in range(1)])  # display only
        out[h] = {"acc": float(a), "pred": float(pr)}
        print(f"  {h:>4} {a:>7.0%} {p_mean:>7.0%} {pr:>7.0%}")
    devs = [abs(v["acc"] - v["pred"]) for v in out.values()]
    print(f"  |Acc − p̂^h| medio = {np.mean(devs):.3f}")
    return out


if __name__ == "__main__":
    print("=" * 66)
    print("  Prop. 3.6 — la profondità è esponenzialmente economica?")
    print("=" * 66)
    results = {"dmin": fig1_dmin(), "fixed_d": fig2_fixed_d()}
    Path("depth_scaling_results.json").write_text(
        json.dumps(results, indent=2))
    print("\n  → depth_scaling_results.json")
