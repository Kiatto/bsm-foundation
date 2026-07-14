"""
seed_robustness.py — Le leggi con intervalli di confidenza al 95%.

Ogni legge osservata viene ristimata su S seed indipendenti (i nomi di
entità/relazioni incorporano il seed → tutti gli hypervector cambiano):

  Law IV  Capacità algebrica:  N* = c·D    → c ± CI95
  Law V   Composizione hop:    Acc(h) = p^h → R² e deviazione media ± CI95
  ProofWriter (depth 0/2/5): accuracy ± CI95 su seed

Output: seed_robustness_results.json
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from bsm.memory.vsa import WorkingMemory

T95 = {4: 2.776, 5: 2.571, 9: 2.262}   # t di Student, due code


def ci95(vals):
    v = np.asarray(vals, dtype=float)
    n = len(v)
    t = T95.get(n - 1, 2.0)
    return float(v.mean()), float(t * v.std(ddof=1) / np.sqrt(n))


# ---------------------------------------------------------------------------
# Law IV — N* = c·D
# ---------------------------------------------------------------------------

def accuracy_at(d, n, seed):
    wm = WorkingMemory(d)
    facts = [(f"s{seed}_{i}", f"r{seed}_{i % 11}", f"o{seed}_{i}")
             for i in range(n)]
    for f in facts:
        wm.store(*f)
    ok = sum(1 for s, r, o in facts if wm.query(s, r)[0] == o)
    return ok / n


def find_nstar(d, seed):
    """Interpolazione del crossing al 50% su una griglia geometrica."""
    prev_n, prev_a = None, None
    n = max(10, d // 16)
    while n <= d:
        a = accuracy_at(d, n, seed)
        if a < 0.5:
            if prev_n is None:
                return n
            # interpolazione lineare tra (prev_n, prev_a) e (n, a)
            return prev_n + (prev_a - 0.5) * (n - prev_n) / (prev_a - a)
        prev_n, prev_a = n, a
        n = int(n * 1.4)
    return d


def law4(seeds=5):
    print("\n[Law IV] N* = c·D  —  stima di c su seed indipendenti")
    dims = [512, 1024, 2048, 4096]
    per_d = {}
    cs = []
    for d in dims:
        nstars = [find_nstar(d, s) for s in range(seeds)]
        m, ci = ci95(nstars)
        per_d[d] = (m, ci)
        cs += [x / d for x in nstars]
        print(f"  D={d:>5}: N* = {m:6.0f} ± {ci:4.0f}   (c = {m/d:.3f})")
    c_m, c_ci = ci95(cs)
    print(f"  → c = {c_m:.3f} ± {c_ci:.3f}   (N* = ({c_m:.3f}±{c_ci:.3f})·D)")
    # linearità: R² del fit N* ~ D
    ds = np.repeat(dims, seeds).astype(float)
    ns = np.array([find_nstar(d, s + 100) for d in dims
                   for s in range(seeds)], dtype=float)
    c_fit = float((ds @ ns) / (ds @ ds))
    ss_res = float(np.sum((ns - c_fit * ds) ** 2))
    ss_tot = float(np.sum((ns - ns.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    print(f"  linearità (fit N*=cD su run indipendenti): R² = {r2:.3f}")
    return {"c": c_m, "c_ci95": c_ci, "r2_linear": r2,
            "per_d": {str(k): v for k, v in per_d.items()}}


# ---------------------------------------------------------------------------
# Law V — Acc(h) = p^h
# ---------------------------------------------------------------------------

def law5(seeds=5, d=2048, n_chains=30):
    print(f"\n[Law V] Acc(h) = p^h  —  D={d}, {n_chains} catene, "
          f"{seeds} seed")
    devs, r2s = [], []
    for seed in range(seeds):
        obs, pred = [], []
        for hops in (1, 2, 3, 4, 5, 7):
            wm = WorkingMemory(d)
            chains = []
            for c in range(n_chains):
                nodes = [f"x{seed}_{c}_{k}" for k in range(hops + 1)]
                for k in range(hops):
                    wm.store(nodes[k], f"rel{seed}_{k}", nodes[k + 1])
                chains.append(nodes)
            hop_ok = hop_tot = 0
            for nodes in chains:
                for k in range(hops):
                    hop_tot += 1
                    hop_ok += (wm.query(nodes[k], f"rel{seed}_{k}")[0]
                               == nodes[k + 1])
            p = hop_ok / hop_tot
            e2e = 0
            for nodes in chains:
                cur = nodes[0]
                for k in range(hops):
                    cur, _ = wm.query(cur, f"rel{seed}_{k}")
                e2e += (cur == nodes[-1])
            acc = e2e / n_chains
            obs.append(acc)
            pred.append(p ** hops)
        obs, pred = np.array(obs), np.array(pred)
        devs.append(float(np.mean(np.abs(obs - pred))))
        mask = (obs > 0) & (pred > 0)
        lo, lp = np.log(obs[mask]), np.log(pred[mask])
        ss_res = float(np.sum((lo - lp) ** 2))
        ss_tot = float(np.sum((lo - lo.mean()) ** 2))
        r2s.append(1 - ss_res / max(ss_tot, 1e-12))
    dev_m, dev_ci = ci95(devs)
    r2_m, r2_ci = ci95(r2s)
    print(f"  deviazione media |Acc − p^h| = {dev_m:.3f} ± {dev_ci:.3f}")
    print(f"  R² (log Acc vs h·log p)      = {r2_m:.3f} ± {r2_ci:.3f}")
    return {"mean_abs_dev": dev_m, "dev_ci95": dev_ci,
            "r2": r2_m, "r2_ci95": r2_ci}


# ---------------------------------------------------------------------------
# ProofWriter multi-seed
# ---------------------------------------------------------------------------

def proofwriter_seeds(seeds=5, n=100):
    print(f"\n[ProofWriter] accuracy ± CI95 su {seeds} seed "
          f"({n} domande/depth)")
    import pyarrow.parquet as pq
    from proofwriter_eval import parse_theory, AlgebraicProver, RE_Q, PARQUET

    rows = [r for r in pq.read_table(PARQUET).to_pylist()
            if r["id"].startswith("AttNoneg")]
    out = {}
    for depth in (0, 2, 5):
        sample = [r for r in rows if r["config"] == f"depth-{depth}"][:n * 3]
        accs = []
        for seed in range(seeds):
            oks = []
            for r in sample:
                if len(oks) >= n:
                    break
                facts, rules, cov = parse_theory(r["theory"])
                mq = RE_Q.match(r["question"].strip())
                if not mq or cov < 0.99:
                    continue
                ent, neg, attr = (mq.group(1).lower(), bool(mq.group(2)),
                                  mq.group(3).lower())
                salt = f"seed{seed}:"
                prover = AlgebraicProver(state_dim=4096)
                for e, a in facts:
                    prover.add_fact(salt + e, salt + a)
                prover.forward_chain(
                    [([salt + p for p in ps], salt + c) for ps, c in rules])
                provable = prover.member(salt + ent, salt + attr)
                pred = (("False" if provable else "Unknown") if neg
                        else ("True" if provable else "Unknown"))
                oks.append(pred == str(r["answer"]))
            accs.append(sum(oks) / len(oks))
        m, ci = ci95(accs)
        print(f"  depth {depth}: {m:.1%} ± {ci:.1%}")
        out[depth] = {"acc": m, "ci95": ci}
    return out


if __name__ == "__main__":
    print("=" * 66)
    print("  Seed robustness — le leggi con intervalli di confidenza")
    print("=" * 66)
    results = {"law4": law4(), "law5": law5(),
               "proofwriter": proofwriter_seeds()}
    Path("seed_robustness_results.json").write_text(
        json.dumps(results, indent=2))
    print("\n  → seed_robustness_results.json")
