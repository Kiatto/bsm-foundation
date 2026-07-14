"""
scale_bench.py — La "tabella dei tre numeri": costo di ingestione,
latenza di query, errore del contratto — su due configurazioni
dimensionate DALLA TEORIA (Inspector), non a mano.

  edge    D scelto per ~250 fatti a pressione < 0.5  (target: KB, non GB)
  server  D scelto per ~2000 fatti a pressione < 0.5

La metrica proprietaria è l'ultima colonna: |contratto − misurato|,
dove il contratto è emesso PRIMA delle query, a zero parametri fittati.

Nota implementativa: la reference ricostruisce il bundle a ogni store
(corretto per la specifica, O(N²) per l'ingestione bulk). Qui, da
livello applicativo, accumuliamo i fatti e consolidiamo una volta —
stesso identico stato finale, come da definizione di bundle.

Output: scale_bench_results.json
"""

import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference"))

import numpy as np
from abm import Memory, bundle
from inspector import stats, contract

CONFIGS = {"edge": 250, "server": 2000}
N_QUERIES = 500


def recommended_dim(n_facts, codebook):
    """La dimensione la sceglie la teoria: minima potenza di 2 con
    pressione <= 0.5 (stessa formula dell'Inspector)."""
    from abm import capacity
    d = 1024
    while n_facts > 0.5 * capacity(d, codebook) and d < 2 ** 20:
        d *= 2
    return d


def run(label, n_facts):
    codebook_est = 2 * n_facts + n_facts // 10
    dim = recommended_dim(n_facts, codebook_est)
    facts = [(f"s{i}", f"r{i % (n_facts // 10)}", f"o{i}")
             for i in range(n_facts)]

    t0 = time.perf_counter()
    mem = Memory(dim)
    for s, r, o in facts:
        mem._facts.append(mem.fact_hv(s, r, o))     # bulk ingest
    mem._trace = bundle(mem._facts)                  # un solo consolidamento
    t_ingest = time.perf_counter() - t0

    s = stats(mem)                                   # contratto PRE-query
    print(f"\n  [{label}] D={dim} ({dim // 8} byte/traccia), "
          f"{n_facts} fatti, codebook={len(mem.items)}")
    print("  " + contract(mem).replace("\n", "\n  "))

    rng = np.random.RandomState(42)
    n_q = min(N_QUERIES, n_facts)
    idx = rng.choice(n_facts, n_q, replace=False)
    t0 = time.perf_counter()
    ok = sum(mem.query(facts[i][0], facts[i][1])[0] == facts[i][2]
             for i in idx)
    t_query = (time.perf_counter() - t0) / n_q
    measured = ok / n_q

    ram_mb = (dim // 8 + len(mem.items) * dim) / 1e6  # traccia + codebook int8
    row = {
        "dimension": dim,
        "facts": n_facts,
        "trace_bytes": dim // 8,
        "ram_mb_unpacked": round(ram_mb, 1),
        "ingest_s": round(t_ingest, 2),
        "ingest_ms_per_fact": round(1000 * t_ingest / n_facts, 2),
        "query_ms": round(1000 * t_query, 2),
        "contract_accuracy": s["expected_accuracy"],
        "measured_accuracy": measured,
        "contract_error": round(abs(s["expected_accuracy"] - measured), 3),
        "pressure": s["pressure"],
    }
    print(f"  ingest {t_ingest:.2f}s ({row['ingest_ms_per_fact']}ms/fatto) | "
          f"query {row['query_ms']}ms | RAM {row['ram_mb_unpacked']}MB | "
          f"contratto {s['expected_accuracy']:.0%} vs misurato "
          f"{measured:.0%} → errore {row['contract_error']:.1%}")
    return row


if __name__ == "__main__":
    print("=" * 72)
    print("  Scale bench — ingestione, latenza, errore del contratto")
    print("=" * 72)
    results = {label: run(label, n) for label, n in CONFIGS.items()}
    Path("scale_bench_results.json").write_text(json.dumps(results, indent=2))
    print("\n  → scale_bench_results.json")
