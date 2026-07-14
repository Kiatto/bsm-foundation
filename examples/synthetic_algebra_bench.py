"""
synthetic_algebra_bench.py — Caratterizzazione intrinseca dell'algebra.

Universo sintetico completamente controllato (nessun testo, nessun
retrieval, nessun parser): SOLO livello B.

    fatti (s, r, o) ──▶ traccia olografica XOR ──▶ query ──▶ cleanup

Curve misurate:
  A. Capacità: accuracy vs N fatti in UNA traccia, per D ∈ {512..8192};
     stima del punto di collasso N* (accuracy < 50%).
  B. Catene: accuracy end-to-end vs numero di hop (1..10), con cleanup
     tra i hop; confronto con la predizione p^h (hop indipendenti).
  C. Rumore: z-score del cleanup (margine statistico) vs carico.
  D. Branching: accuracy vs numero di relazioni uscenti per nodo.

Output: tabelle + synthetic_bench_results.json (dati per le figure).
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from bsm.memory.vsa import WorkingMemory, hamming

RESULTS = {}


# ---------------------------------------------------------------------------
# A. Capacità e punto di collasso
# ---------------------------------------------------------------------------

def bench_capacity():
    print("\n[A] Capacità olografica: accuracy vs N fatti, per D")
    dims = [512, 1024, 2048, 4096, 8192]
    loads = [10, 20, 40, 80, 160, 320, 640]
    print(f"  {'N':>5} " + " ".join(f"D={d:>5}" for d in dims))
    grid = {}
    collapse = {}
    for n in loads:
        row = []
        for d in dims:
            wm = WorkingMemory(d)
            facts = [(f"s{i}", f"r{i % 11}", f"o{i}") for i in range(n)]
            for f in facts:
                wm.store(*f)
            ok = sum(1 for s, r, o in facts if wm.query(s, r)[0] == o)
            acc = ok / n
            row.append(acc)
            grid[f"{d}:{n}"] = acc
            if d not in collapse and acc < 0.5:
                collapse[d] = n
        print(f"  {n:>5} " + " ".join(f"{a:>6.0%}" for a in row))
    print("  punto di collasso N* (acc<50%): " +
          ", ".join(f"D={d}→N*≈{collapse.get(d, '>640')}" for d in dims))
    RESULTS["capacity"] = grid
    RESULTS["collapse"] = {str(k): v for k, v in collapse.items()}


# ---------------------------------------------------------------------------
# B. Catene: accuracy vs hop
# ---------------------------------------------------------------------------

def bench_chains(d=2048, n_chains=30):
    print(f"\n[B] Catene multi-hop (D={d}, {n_chains} catene, "
          f"cleanup tra i hop)")
    print(f"  {'hop':>5} {'acc end-to-end':>15} {'p^h previsto':>13} "
          f"{'acc per-hop':>12}")
    out = {}
    for hops in (1, 2, 3, 5, 7, 10):
        wm = WorkingMemory(d)
        chains = []
        for c in range(n_chains):
            nodes = [f"n{c}_{k}" for k in range(hops + 1)]
            for k in range(hops):
                wm.store(nodes[k], f"rel{k}", nodes[k + 1])
            chains.append(nodes)
        # accuracy per-hop (su tutti i singoli fatti)
        hop_ok = hop_tot = 0
        for nodes in chains:
            for k in range(hops):
                hop_tot += 1
                if wm.query(nodes[k], f"rel{k}")[0] == nodes[k + 1]:
                    hop_ok += 1
        p = hop_ok / hop_tot
        # accuracy end-to-end (catena intera, cleanup a ogni hop)
        e2e = 0
        for nodes in chains:
            cur = nodes[0]
            for k in range(hops):
                cur, _ = wm.query(cur, f"rel{k}")
            e2e += (cur == nodes[-1])
        acc = e2e / n_chains
        pred = p ** hops
        print(f"  {hops:>5} {acc:>14.0%} {pred:>12.0%} {p:>11.0%}")
        out[hops] = {"e2e": acc, "per_hop": p, "pred": pred,
                     "n_facts": hops * n_chains}
    RESULTS["chains"] = out
    print("  (il carico cresce con i hop: 10 hop × 30 catene = 300 fatti)")


# ---------------------------------------------------------------------------
# C. Rumore: z-score del cleanup vs carico
# ---------------------------------------------------------------------------

def bench_noise(d=2048):
    print(f"\n[C] Margine statistico del cleanup vs carico (D={d})")
    print(f"  {'N fatti':>8} {'dist corretto':>14} {'dist rumore':>12} "
          f"{'z-score':>8}")
    out = {}
    std = np.sqrt(d) / 2
    for n in (10, 40, 160, 640):
        wm = WorkingMemory(d)
        facts = [(f"s{i}", f"r{i % 11}", f"o{i}") for i in range(n)]
        for f in facts:
            wm.store(*f)
        dists = []
        for s, r, o in facts[: min(n, 60)]:
            name, dist = wm.query(s, r)
            if name == o:
                dists.append(dist)
        d_ok = float(np.mean(dists)) if dists else float("nan")
        z = (d / 2 - d_ok) / std
        print(f"  {n:>8} {d_ok:>13.0f} {d / 2:>11.0f} {z:>7.1f}σ")
        out[n] = {"correct_dist": d_ok, "z": z}
    RESULTS["noise"] = out


# ---------------------------------------------------------------------------
# D. Branching factor
# ---------------------------------------------------------------------------

def bench_branching(d=2048, n_nodes=25):
    print(f"\n[D] Branching: relazioni uscenti per nodo (D={d}, "
          f"{n_nodes} nodi)")
    print(f"  {'B':>4} {'accuracy':>9} {'N fatti':>8}")
    out = {}
    for b in (1, 2, 4, 8, 16):
        wm = WorkingMemory(d)
        facts = []
        for i in range(n_nodes):
            for j in range(b):
                facts.append((f"node{i}", f"rel{j}", f"obj{i}_{j}"))
        for f in facts:
            wm.store(*f)
        ok = sum(1 for s, r, o in facts if wm.query(s, r)[0] == o)
        acc = ok / len(facts)
        print(f"  {b:>4} {acc:>8.0%} {len(facts):>8}")
        out[b] = {"acc": acc, "n_facts": len(facts)}
    RESULTS["branching"] = out


if __name__ == "__main__":
    print("=" * 66)
    print("  Synthetic Algebra Benchmark — livello B puro")
    print("=" * 66)
    bench_capacity()
    bench_chains()
    bench_noise()
    bench_branching()
    Path("synthetic_bench_results.json").write_text(
        json.dumps(RESULTS, indent=2))
    print("\n  → synthetic_bench_results.json")
