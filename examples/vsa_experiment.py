"""
vsa_experiment.py — Il ragionamento come algebra binaria: 3 condizioni.

Task 1  Capacità olografica: N fatti in UN vettore, query via XOR.
Task 2  Multi-hop puro: catene a 2 hop risolte con solo unbinding+cleanup
        (zero euristiche testuali, zero indici).
Task 3  Generalizzazione alias (la tesi Working/Semantic): l'addressing
        con binding XOR (esatto) vs proiezione random vs proiezione
        LOSSY FITTATA — su query con nomi varianti mai visti.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from bsm.memory.vsa import (WorkingMemory, SemanticMemory, ItemMemory,
                            random_hv, bind_xor, bundle, hamming)
from bsm.memory.encoder.entity_encoder import _minhash_sketch


# ===========================================================================
# Task 1 — capacità olografica
# ===========================================================================

def task1_capacity(state_dim=1024):
    print(f"\n[Task 1] Capacità olografica (D={state_dim}): N fatti in UN vettore")
    print(f"  {'N fatti':>8} {'accuracy':>9}")
    rng = np.random.RandomState(7)
    for n in (5, 10, 20, 40, 80, 160):
        wm = WorkingMemory(state_dim)
        facts = [(f"s{i}", f"r{i % 7}", f"o{i}") for i in range(n)]
        # registra anche item distrattori nel codebook
        for i in range(n):
            wm.items.add(f"o{i}")
        for s, r, o in facts:
            wm.store(s, r, o)
        ok = sum(1 for s, r, o in facts if wm.query(s, r)[0] == o)
        print(f"  {n:>8} {ok / n:>8.0%}")


# ===========================================================================
# Task 2 — multi-hop algebrico
# ===========================================================================

def task2_multihop(state_dim=1024, n_chains=25):
    print(f"\n[Task 2] Multi-hop puro (D={state_dim}, {n_chains} catene a 2 hop)")
    wm = WorkingMemory(state_dim)
    chains = []
    for i in range(n_chains):
        x, y, z = f"prodotto{i}", f"azienda{i}", f"città{i}"
        wm.store(x, "sviluppato_da", y)
        wm.store(y, "sede_a", z)
        chains.append((x, y, z))

    ok_hop1 = ok_hop2 = 0
    for x, y, z in chains:
        bridge, _ = wm.query(x, "sviluppato_da")       # hop 1: XOR + cleanup
        if bridge == y:
            ok_hop1 += 1
        answer, _ = wm.query(bridge, "sede_a")          # hop 2: XOR + cleanup
        if answer == z:
            ok_hop2 += 1
    print(f"  hop 1: {ok_hop1}/{n_chains}   catena completa: {ok_hop2}/{n_chains}")
    print("  (nessuna euristica testuale: la query è T ⊕ chiave → cleanup)")


# ===========================================================================
# Task 3 — generalizzazione alias: XOR vs proj random vs proj fittata
# ===========================================================================

# (nome canonico, [varianti di query], hq, founder)
ENTITIES = [
    ("google llc",            ["google", "google inc", "the google company"],
     "mountain view", "larry page"),
    ("google cloud platform", ["google cloud"],
     "seattle", "diane greene"),
    ("microsoft corporation", ["microsoft", "microsoft corp"],
     "redmond", "bill gates"),
    ("apple inc",             ["apple", "apple computer"],
     "cupertino", "steve jobs"),
    ("apple records ltd",     ["apple records"],
     "london", "the beatles"),
    ("amazon.com inc",        ["amazon", "amazon inc"],
     "seattle", "jeff bezos"),
    ("amazon web services",   ["aws amazon services"],
     "arlington", "andy jassy"),
    ("tesla inc",             ["tesla", "tesla motors"],
     "austin", "elon musk"),
    ("netflix inc",           ["netflix"],
     "los gatos", "reed hastings"),
    ("meta platforms inc",    ["meta", "meta platforms"],
     "menlo park", "mark zuckerberg"),
    ("oracle corporation",    ["oracle", "oracle corp"],
     "austin", "larry ellison"),
    ("intel corporation",     ["intel", "intel corp"],
     "santa clara", "gordon moore"),
]


def _sketch(name: str, d: int) -> np.ndarray:
    return _minhash_sketch(set(name.split()), d)


def task3_alias(state_dim=1024, rank=16):
    print(f"\n[Task 3] Generalizzazione alias (D={state_dim}, rank fit={rank})")
    canon_states = [_sketch(e[0], state_dim) for e in ENTITIES]

    conditions = {
        "XOR (invertibile)":        SemanticMemory(state_dim, binding="xor"),
        "proiezione random":        SemanticMemory(state_dim, binding="proj"),
        f"proiezione fittata r={rank}":
            SemanticMemory(state_dim, binding="proj").fit(canon_states, rank),
    }

    results = {}
    for label, sm in conditions.items():
        for (canon, _, hq, founder), st in zip(ENTITIES, canon_states):
            sm.store(st, "hq", hq, s_name=canon)
            sm.store(st, "founder", founder, s_name=canon)
        ok = tot = 0
        margins = []
        for canon, variants, hq, founder in ENTITIES:
            for v in variants:
                vs = _sketch(v, state_dim)
                for rel, expected in (("hq", hq), ("founder", founder)):
                    got, d, margin = sm.query(vs, rel)
                    tot += 1
                    if got == expected:
                        ok += 1
                        margins.append(margin)
        results[label] = (ok, tot, np.mean(margins) if margins else 0)

    print(f"  {'condizione':<28} {'accuracy':>10} {'margine medio':>14}")
    for label, (ok, tot, m) in results.items():
        print(f"  {label:<28} {ok}/{tot} ({ok / tot:.0%}) {m:>10.1f} bit")
    return results


def rank_sweep(state_dim=1024):
    print("\n[Task 3b] Sensibilità al rango della proiezione fittata")
    canon_states = [_sketch(e[0], state_dim) for e in ENTITIES]
    print(f"  {'rank':>6} {'accuracy':>9} {'margine':>9}")
    for rank in (4, 8, 12, 16, 24, 32):
        sm = SemanticMemory(state_dim, binding="proj").fit(canon_states, rank)
        for (canon, _, hq, founder), st in zip(ENTITIES, canon_states):
            sm.store(st, "hq", hq, s_name=canon)
            sm.store(st, "founder", founder, s_name=canon)
        ok = tot = 0
        margins = []
        for canon, variants, hq, founder in ENTITIES:
            for v in variants:
                vs = _sketch(v, state_dim)
                for rel, expected in (("hq", hq), ("founder", founder)):
                    got, d, margin = sm.query(vs, rel)
                    tot += 1
                    if got == expected:
                        ok += 1
                        margins.append(margin)
        print(f"  {rank:>6} {ok / tot:>8.0%} {np.mean(margins):>7.1f} bit")


if __name__ == "__main__":
    print("=" * 66)
    print("  VSA su BSM — ragionamento come algebra binaria")
    print("=" * 66)
    task1_capacity()
    task2_multihop()
    task3_alias()
    rank_sweep()
