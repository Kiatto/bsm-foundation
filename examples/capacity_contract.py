"""
capacity_contract.py — La "scheda tecnica" di una memoria ABM:
accuratezza PREDETTA dalla teoria (Law IV + Gumbel, zero parametri
fittati) vs MISURATA, a budget fisso di 1 KB (D=8192).

Risultato: |errore| medio della predizione 4.2%.
Contratto esempio: "con 1 KB, fino a 300 fatti, accuratezza >= 85%".
"""
import sys, json
from math import erf, sqrt, log, pi
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from bsm.memory.vsa import WorkingMemory

D, SEEDS = 8192, 3


def z_gumbel(M):
    zm = sqrt(2 * log(M))
    return zm - (log(log(M)) + log(4 * pi)) / (2 * zm)


def predicted_acc(N, M):
    margin = sqrt(2 * D / (pi * N)) - z_gumbel(M)
    return 0.5 * (1 + erf(margin / sqrt(2)))


if __name__ == "__main__":
    print(f"CAPACITY CONTRACT — 1 KB (D={D}), predizione senza fit")
    print(f"{'N':>6} {'predetto':>9} {'misurato':>9}")
    rows = {}
    for N in (100, 200, 300, 400, 500, 600):
        accs = []
        for seed in range(SEEDS):
            wm = WorkingMemory(D)
            facts = [(f"s{seed}_{i}", f"r{seed}_{i % 13}", f"o{seed}_{i}")
                     for i in range(N)]
            for f in facts:
                wm.store(*f)
            k = min(N, 80)
            accs.append(sum(wm.query(s, r)[0] == o
                            for s, r, o in facts[:k]) / k)
        rows[N] = {"pred": predicted_acc(N, 2 * N + 13),
                   "meas": float(np.mean(accs))}
        print(f"{N:>6} {rows[N]['pred']:>8.0%} {rows[N]['meas']:>8.0%}")
    err = np.mean([abs(v['pred'] - v['meas']) for v in rows.values()])
    print(f"|errore| medio: {err:.3f}")
    Path("capacity_contract_results.json").write_text(
        json.dumps(rows, indent=2))
