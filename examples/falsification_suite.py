"""
falsification_suite.py — Tentativi deliberati di rompere le leggi.

Fase adversariale: non cerchiamo conferme, cerchiamo smentite.

  F1. Test di ln M (LA previsione della Law IV): N* a D fisso deve
      contrarsi come 1/ln M quando il codebook cresce.
      k = N*·ln(M)/D deve restare costante entro i CI.
  F2. Distribuzione non uniforme (Zipf): la Law IV descrive ancora la
      media? Il degrado si concentra sui nodi pesanti (Law VI)?
  F3. Fatti ridondanti: la ripetizione pesa il majority vote — aiuta i
      fatti ripetuti e danneggia gli altri? Di quanto?
  F4. Rumore intenzionale: flip di una frazione ε dei bit della traccia
      → il segnale deve scalare come (1−2ε) (degrado dolce, predicibile).
  F5. Cleanup imperfetta: codebook corrotto di ε bit per item.

Output: falsification_results.json
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from bsm.memory.vsa import WorkingMemory, random_hv

D = 2048
SEEDS = 3
RESULTS = {}


def build(n, seed, subjects=None):
    """WorkingMemory con n fatti; subjects opzionale per distribuzioni."""
    wm = WorkingMemory(D)
    facts = []
    for i in range(n):
        s = subjects[i] if subjects else f"s{seed}_{i}"
        facts.append((s, f"r{seed}_{i}", f"o{seed}_{i}"))
    for f in facts:
        wm.store(*f)
    return wm, facts


def acc_of(wm, facts, k=None):
    sample = facts if k is None else facts[:k]
    return sum(wm.query(s, r)[0] == o for s, r, o in sample) / len(sample)


# ---------------------------------------------------------------------------
# F1 — la previsione: N* ∝ 1/ln M a D fisso
# ---------------------------------------------------------------------------

def f1_lnM():
    print("\n[F1] Test di ln M — N* a D=2048 con codebook gonfiato")
    print(f"  {'M_extra':>8} {'N*':>6} {'M_tot':>7} {'k=N*·lnM/D':>11}")
    out = {}
    for m_extra in (0, 1000, 4000, 16000):
        nstars = []
        for seed in range(SEEDS):
            # distrattori nel codebook (item mai usati nei fatti)
            lo, hi, n = 20, D, 120
            # ricerca del crossing 50% (griglia geometrica + interp.)
            prev = None
            n = 40
            nstar = D
            while n <= 900:
                wm, facts = build(n, seed)
                for j in range(m_extra):
                    wm.items.add(f"distr{seed}_{j}")
                a = acc_of(wm, facts, k=min(n, 80))
                if a < 0.5:
                    if prev is None:
                        nstar = n
                    else:
                        pn, pa = prev
                        nstar = pn + (pa - 0.5) * (n - pn) / (pa - a)
                    break
                prev = (n, a)
                n = int(n * 1.45)
            nstars.append(nstar)
        m = float(np.mean(nstars))
        m_tot = 2 * m + m_extra + m  # subj+obj+rel ≈ 3N* + distrattori
        k = m * np.log(m_tot + m_extra) / D
        m_tot = 3 * m + m_extra
        k = m * np.log(m_tot) / D
        out[m_extra] = {"nstar": m, "M": m_tot, "k": k}
        print(f"  {m_extra:>8} {m:>6.0f} {m_tot:>7.0f} {k:>11.3f}")
    ks = [v["k"] for v in out.values()]
    spread = (max(ks) - min(ks)) / np.mean(ks)
    print(f"  k: media {np.mean(ks):.3f}, spread {spread:.1%} "
          f"{'→ LEGGE REGGE' if spread < 0.15 else '→ LEGGE IN DIFFICOLTÀ'}")
    RESULTS["f1_lnM"] = out


# ---------------------------------------------------------------------------
# F2 — distribuzione Zipf dei soggetti
# ---------------------------------------------------------------------------

def f2_zipf(n=120):
    print(f"\n[F2] Soggetti Zipf vs uniformi (N={n}, D={D})")
    rng = np.random.RandomState(0)
    out = {}
    for label, subjects_fn in (
        ("uniforme", lambda seed: None),
        ("zipf(1.2)", lambda seed: [
            f"z{seed}_{min(int(x), 49)}"
            for x in rng.zipf(1.2, n)]),
    ):
        accs, heavy_accs, light_accs = [], [], []
        for seed in range(SEEDS):
            subs = subjects_fn(seed)
            wm, facts = build(n, seed, subjects=subs)
            accs.append(acc_of(wm, facts))
            if subs:
                from collections import Counter
                deg = Counter(s for s, _, _ in facts)
                heavy = [(s, r, o) for s, r, o in facts if deg[s] >= 5]
                light = [(s, r, o) for s, r, o in facts if deg[s] <= 2]
                if heavy:
                    heavy_accs.append(acc_of(wm, heavy))
                if light:
                    light_accs.append(acc_of(wm, light))
        row = f"  {label:12s} acc={np.mean(accs):.0%}"
        if heavy_accs:
            row += (f"   nodi pesanti(≥5)={np.mean(heavy_accs):.0%} "
                    f"leggeri(≤2)={np.mean(light_accs):.0%}")
        print(row)
        out[label] = {"acc": float(np.mean(accs)),
                      "heavy": float(np.mean(heavy_accs)) if heavy_accs else None,
                      "light": float(np.mean(light_accs)) if light_accs else None}
    RESULTS["f2_zipf"] = out


# ---------------------------------------------------------------------------
# F3 — ridondanza
# ---------------------------------------------------------------------------

def f3_redundancy(n=100, dup=5, n_dup=10):
    print(f"\n[F3] Ridondanza: {n_dup} fatti ripetuti {dup}× su {n} totali")
    accs_dup, accs_rest, accs_ctrl = [], [], []
    for seed in range(SEEDS):
        wm = WorkingMemory(D)
        facts = [(f"s{seed}_{i}", f"r{seed}_{i}", f"o{seed}_{i}")
                 for i in range(n)]
        for i, f in enumerate(facts):
            for _ in range(dup if i < n_dup else 1):
                wm.store(*f)
        accs_dup.append(acc_of(wm, facts[:n_dup]))
        accs_rest.append(acc_of(wm, facts[n_dup:]))
        wm2, facts2 = build(n, seed + 50)
        accs_ctrl.append(acc_of(wm2, facts2))
    print(f"  fatti ripetuti: {np.mean(accs_dup):.0%}   "
          f"altri: {np.mean(accs_rest):.0%}   "
          f"controllo (nessuna ripetizione): {np.mean(accs_ctrl):.0%}")
    RESULTS["f3_redundancy"] = {
        "duplicated": float(np.mean(accs_dup)),
        "others": float(np.mean(accs_rest)),
        "control": float(np.mean(accs_ctrl))}


# ---------------------------------------------------------------------------
# F4 — rumore nella traccia
# ---------------------------------------------------------------------------

def f4_trace_noise(n=80):
    print(f"\n[F4] Flip di ε bit della traccia (N={n}, D={D}) — "
          f"predizione: segnale ∝ (1−2ε)")
    print(f"  {'ε':>6} {'accuracy':>9}")
    out = {}
    rng = np.random.RandomState(1)
    for eps in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40):
        accs = []
        for seed in range(SEEDS):
            wm, facts = build(n, seed)
            if eps > 0:
                idx = rng.choice(D, int(eps * D), replace=False)
                wm._trace = wm._trace.copy()
                wm._trace[idx] = -wm._trace[idx]
            accs.append(acc_of(wm, facts))
        print(f"  {eps:>6.0%} {np.mean(accs):>8.0%}")
        out[eps] = float(np.mean(accs))
    RESULTS["f4_noise"] = out


# ---------------------------------------------------------------------------
# F5 — cleanup imperfetta
# ---------------------------------------------------------------------------

def f5_dirty_cleanup(n=80):
    print(f"\n[F5] Codebook corrotto di ε bit per item (N={n}, D={D})")
    print(f"  {'ε':>6} {'accuracy':>9}")
    out = {}
    rng = np.random.RandomState(2)
    for eps in (0.0, 0.05, 0.10, 0.20, 0.30):
        accs = []
        for seed in range(SEEDS):
            wm, facts = build(n, seed)
            if eps > 0:
                for i, st in enumerate(wm.items._states):
                    idx = rng.choice(D, int(eps * D), replace=False)
                    st = st.copy()
                    st[idx] = -st[idx]
                    wm.items._states[i] = st
            accs.append(acc_of(wm, facts))
        print(f"  {eps:>6.0%} {np.mean(accs):>8.0%}")
        out[eps] = float(np.mean(accs))
    RESULTS["f5_dirty_cleanup"] = out


if __name__ == "__main__":
    print("=" * 66)
    print("  Falsification suite — proviamo a rompere le leggi")
    print("=" * 66)
    f1_lnM()
    f2_zipf()
    f3_redundancy()
    f4_trace_noise()
    f5_dirty_cleanup()
    Path("falsification_results.json").write_text(
        json.dumps(RESULTS, indent=2))
    print("\n  → falsification_results.json")
