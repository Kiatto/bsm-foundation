"""
compiler_bench.py — Verifica della predizione del Memory Calculus §5.2.

Naïf (interpretato): 2 cleanup sulla traccia T → Law V: p².
Compilato: a sleep-time le composizioni esatte F = f1⊕f2 (Normal Form
Theorem, NF.1: probabilità 1, zero cleanup) vengono consolidate in una
seconda traccia T2; a query time UN cleanup su T2 con chiave composta
statica  k = c_x ⊕ ρ(r1) ⊕ ρ(r2).

Risultato (D=2048, 5 seed): a 80 catene 89% vs 25%; a 160 catene
44% vs 3%. Doppio guadagno previsto: un solo passo probabilistico, e
T2 metà carica di T.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from bsm.memory.vsa import WorkingMemory, bind_xor, permute, bundle

D, SEEDS = 2048, 5


def run(n_chains, seed):
    wm = WorkingMemory(D)
    chains = []
    for c in range(n_chains):
        x, y, z = f"x{seed}_{c}", f"y{seed}_{c}", f"z{seed}_{c}"
        wm.store(x, f"r1_{seed}", y)
        wm.store(y, f"r2_{seed}", z)
        chains.append((x, y, z))
    composed = [bind_xor(wm._facts[i], wm._facts[i + 1])
                for i in range(0, len(wm._facts) - 1, 2)]
    T2 = bundle(composed)
    naive_ok = comp_ok = 0
    r12 = bind_xor(permute(wm.items.get(f"r1_{seed}"), 1),
                   permute(wm.items.get(f"r2_{seed}"), 1))
    for x, y, z in chains:
        b, _ = wm.query(x, f"r1_{seed}")
        naive_ok += (wm.query(b, f"r2_{seed}")[0] == z)
        key = bind_xor(wm.items.get(x), r12)
        comp_ok += (wm.items.cleanup(bind_xor(T2, key))[0] == z)
    p = np.mean([wm.query(x, f"r1_{seed}")[0] == y for x, y, _ in chains])
    return naive_ok / n_chains, comp_ok / n_chains, p


if __name__ == "__main__":
    print(f"{'catene':>7} {'naïf':>6} {'p²':>6} {'compilato':>10}")
    out = {}
    for n_chains in (20, 40, 80, 160):
        res = [run(n_chains, s) for s in range(SEEDS)]
        naive, comp, p = (np.mean([r[i] for r in res]) for i in range(3))
        print(f"{n_chains:>7} {naive:>6.0%} {p*p:>6.0%} {comp:>10.0%}")
        out[n_chains] = {"naive": float(naive), "p2": float(p * p),
                         "compiled": float(comp)}
    Path("compiler_bench_results.json").write_text(json.dumps(out, indent=2))
