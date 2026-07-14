"""
inspector.py — ABM Inspector: la teoria esposta come API.

Non tocca la reference congelata (abm.py): la osserva. Ogni campo di
stats() è una formula del formalismo, non una statistica descrittiva —
è la differenza tra mostrare numeri e mostrare garanzie.

    from abm import Memory
    from inspector import stats, contract, report

    stats(mem)                      → dict con i campi del Memory Contract
    contract(mem, grounding=0.93)   → la specifica firmabile
    report(mem)                     → testo leggibile
"""

import sys
from math import sqrt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from abm import (Memory, capacity, predicted_accuracy, z_gumbel, hamming,
                 confidence)


def stats(mem: Memory, extractor_precision: float = 1.0,
          horizon: int = 100, hops: int = 2) -> dict:
    """Il Memory Contract calcolato, non osservato."""
    n = len(mem._facts)
    m = max(len(mem.items), 2)
    cap = capacity(mem.dim, m)
    acc_now = predicted_accuracy(n, mem.dim, m) if n else 1.0
    acc_h = predicted_accuracy(n + horizon, mem.dim,
                               m + 2 * horizon) if n else 1.0
    # margine di confidenza: z del segnale meno la soglia del codebook
    z_signal = sqrt(2 * mem.dim / (np.pi * max(n, 1)))
    margin = z_signal - z_gumbel(m)
    # dimensione raccomandata: la minima potenza di 2 con pressione ≤ 0.5
    rec = mem.dim
    while n > 0.5 * capacity(rec, m) and rec < 2 ** 20:
        rec *= 2
    pg = extractor_precision ** hops
    return {
        "facts": n,
        "dimension": mem.dim,
        "codebook": m,
        "estimated_capacity": int(cap),
        "utilization": round(n / cap, 2) if cap else 0.0,
        "pressure": round(n / cap, 2) if cap else 0.0,
        "expected_accuracy": round(acc_now, 3),
        "grounding_contract": round(pg, 3),
        "projected_accuracy": round(pg * acc_now, 3),   # Composition Law
        "capacity_remaining": max(int(cap) - n, 0),
        "expected_after_horizon": round(acc_h, 3),
        "horizon": horizon,
        "confidence_margin_sigma": round(margin, 1),
        "recommended_dimension": rec,
        "dominant_error": ("grounding" if pg < acc_now else "capacity"),
        "max_reasoning_depth_p50": (
            int(np.floor(np.log(0.5) / np.log(acc_now)))
            if 0 < acc_now < 1 else 99),
    }


def contract(mem: Memory, grounding: float = 1.0) -> str:
    """La specifica firmabile prima del deploy."""
    s = stats(mem, extractor_precision=grounding)
    return (
        f"MEMORY CONTRACT\n"
        f"  Capacity        <= {s['estimated_capacity']} facts "
        f"(D={s['dimension']}, codebook={s['codebook']})\n"
        f"  Expected accuracy >= {s['expected_accuracy']:.0%} "
        f"at current load ({s['facts']} facts)\n"
        f"  Grounding       >= {grounding:.0%} "
        f"(projected end-to-end {s['projected_accuracy']:.0%})\n"
        f"  Confidence      calibrated (0.5 = chance), "
        f"margin {s['confidence_margin_sigma']}σ\n"
        f"  Max depth (p50) =  {s['max_reasoning_depth_p50']} hops\n"
        f"  Pressure        =  {s['pressure']:.2f}"
        + ("  ← consider D=" + str(s['recommended_dimension'])
           if s['recommended_dimension'] > s['dimension'] else "")
    )


def report(mem: Memory, extractor_precision: float = 1.0) -> str:
    s = stats(mem, extractor_precision)
    lines = [f"{k:.<28}{v}" for k, v in s.items()]
    return "ABM INSPECTOR\n" + "\n".join("  " + line for line in lines)


if __name__ == "__main__":
    mem = Memory(2048)
    for i in range(287):
        mem.store(f"s{i}", f"r{i % 17}", f"o{i}")
    print(report(mem, extractor_precision=0.93))
    print()
    print(contract(mem, grounding=0.93))
