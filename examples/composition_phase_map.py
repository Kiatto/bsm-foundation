"""
composition_phase_map.py — Phase Transition of Algebraic Composition.

Domanda: NON "compose funziona?", ma "quando smette, e con quale legge?"

Predizioni (derivate PRIMA dell'esperimento):
  P1. Protocollo RAW (nessun cleanup intermedio: il decodificato rumoroso
      entra direttamente nella chiave successiva): il segnale decade
      geometricamente, z(h) = √D · ρ_N^h con ρ_N = √(2/(πN)).
      → collasso: l'accuracy su tutta la mappa (N × h) è funzione della
      sola variabile z_eff = √D·ρ_N^h.
  P2. Protocollo MEDIATO (cleanup a ogni hop): Acc = p^h (Law V) — il
      reset del cleanup è ciò che rende possibile la profondità.
  P3. Compose a due cleanup espliciti: P = p² (ipotesi del revisore).

Output: composition_phase_results.json (la superficie + il collasso).
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from bsm.memory.vsa import WorkingMemory, bind_xor, permute, hamming

D = 2048
SEEDS = 3
MIN_TRIALS = 40


def build_chains(n_load, h, seed):
    """Memorie a carico costante n_load con catene di h hop."""
    per_mem = max(1, n_load // h)
    n_mems = int(np.ceil(MIN_TRIALS / per_mem))
    mems = []
    for m in range(n_mems):
        wm = WorkingMemory(D)
        chains = []
        for c in range(per_mem):
            nodes = [f"n{seed}_{m}_{c}_{k}" for k in range(h + 1)]
            for k in range(h):
                wm.store(nodes[k], f"r{seed}_{m}_{c}_{k}", nodes[k + 1])
            chains.append(nodes)
        for j in range(n_load - per_mem * h):
            wm.store(f"p{seed}_{m}_{j}", f"pr{j}", f"po{seed}_{m}_{j}")
        mems.append((wm, chains, m))
    return mems


def run_protocols(n_load, h, seed):
    """→ (acc_raw, acc_mediated) sulla stessa popolazione di catene."""
    raw_ok = med_ok = tot = 0
    for wm, chains, m in build_chains(n_load, h, seed):
        for nodes in chains:
            tot += 1
            # RAW: il vettore rumoroso decodificato entra nella chiave
            # successiva SENZA cleanup; un solo cleanup finale.
            state = wm.items.get(nodes[0])
            for k in range(h):
                key = bind_xor(state,
                               permute(wm.items.add(
                                   f"r{seed}_{m}_{chains.index(nodes)}_{k}"), 1))
                state = bind_xor(wm._trace, key)   # decodifica rumorosa
            name, _, _ = wm.items.cleanup(state)
            raw_ok += (name == nodes[-1])
            # MEDIATO: cleanup a ogni hop (Law V)
            cur = nodes[0]
            for k in range(h):
                cur, _ = wm.query(cur,
                                  f"r{seed}_{m}_{chains.index(nodes)}_{k}")
            med_ok += (cur == nodes[-1])
    return raw_ok / tot, med_ok / tot


def phase_map():
    loads = [40, 80, 160, 320]
    hops = [1, 2, 3, 4, 6, 8]
    print(f"\n[Mappa di fase]  D={D}, accuracy RAW / MEDIATO")
    print("  N\\h   " + "   ".join(f"{h:>9}" for h in hops))
    surface = {}
    for n in loads:
        row = []
        for h in hops:
            accs = [run_protocols(n, h, s) for s in range(SEEDS)]
            raw = float(np.mean([a for a, _ in accs]))
            med = float(np.mean([m for _, m in accs]))
            surface[f"{n}:{h}"] = {"raw": raw, "med": med}
            row.append(f"{raw:>4.0%}/{med:<4.0%}")
        print(f"  {n:>4}  " + "  ".join(row))
    return surface, loads, hops


def collapse_test(surface, loads, hops):
    """P1: l'accuracy RAW è funzione della sola z_eff = √D·ρ_N^h ?"""
    print("\n[Collasso]  z_eff = √D·ρ_N^h  (ρ_N = √(2/(πN)))")
    pts = []
    for n in loads:
        rho = np.sqrt(2 / (np.pi * n))
        for h in hops:
            z = np.sqrt(D) * rho ** h
            pts.append((z, surface[f"{n}:{h}"]["raw"], n, h))
    pts.sort()
    print(f"  {'z_eff':>8} {'acc RAW':>8} {'(N,h)':>10}")
    for z, a, n, h in pts:
        print(f"  {z:>8.2f} {a:>7.0%} {f'({n},{h})':>10}")
    # qualità del collasso: fit logistico acc ~ σ(a·ln z + b), R²
    zs = np.array([p[0] for p in pts])
    accs = np.array([p[1] for p in pts])
    mask = zs > 0
    X = np.log(zs[mask])
    # regressione logistica grezza via minimi quadrati sul logit troncato
    y = np.clip(accs[mask], 0.01, 0.99)
    logit = np.log(y / (1 - y))
    A = np.vstack([X, np.ones_like(X)]).T
    coef, *_ = np.linalg.lstsq(A, logit, rcond=None)
    pred = 1 / (1 + np.exp(-(A @ coef)))
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"  collasso su curva unica σ(a·ln z + b): R² = {r2:.3f}")
    return float(r2)


def compose_p_squared():
    """P3: compose con due cleanup espliciti → P = p² ?"""
    print("\n[P = p²]  compose a due cleanup vs prodotto dei singoli")
    rows = []
    for n in (80, 160, 320):
        p_single, p_comp = [], []
        for seed in range(SEEDS):
            for wm, chains, m in build_chains(n, 2, seed):
                for nodes in chains:
                    a, b, c = nodes
                    r0 = f"r{seed}_{m}_{chains.index(nodes)}_0"
                    r1 = f"r{seed}_{m}_{chains.index(nodes)}_1"
                    ok0 = wm.query(a, r0)[0] == b
                    ok1 = wm.query(b, r1)[0] == c
                    p_single += [ok0, ok1]
                    # compose: cleanup dei due oggetti, poi UN XOR
                    b_hat = wm.items.get(wm.query(a, r0)[0])
                    f1 = bind_xor(bind_xor(wm.items.get(a),
                                  permute(wm.items.get(r0), 1)), b_hat)
                    c_hat_name = wm.query(wm.query(a, r0)[0], r1)[0]
                    f2 = bind_xor(bind_xor(b_hat,
                                  permute(wm.items.get(r1), 1)),
                                  wm.items.get(c_hat_name))
                    probe = bind_xor(bind_xor(wm.items.get(a),
                                     permute(wm.items.get(r0), 1)),
                                     permute(wm.items.get(r1), 1))
                    name, _, _ = wm.items.cleanup(
                        bind_xor(bind_xor(f1, f2), probe))
                    p_comp.append(name == c)
        p = float(np.mean(p_single))
        pc = float(np.mean(p_comp))
        rows.append((n, p, p * p, pc))
        print(f"  N={n:>4}: p={p:.2f}  p²={p*p:.2f}  "
              f"P_compose misurato={pc:.2f}")
    return rows


if __name__ == "__main__":
    print("=" * 66)
    print("  Phase Transition of Algebraic Composition")
    print("=" * 66)
    surface, loads, hops = phase_map()
    r2 = collapse_test(surface, loads, hops)
    psq = compose_p_squared()
    Path("composition_phase_results.json").write_text(json.dumps(
        {"surface": surface, "collapse_r2": r2,
         "p_squared": psq}, indent=2))
    print("\n  → composition_phase_results.json")
