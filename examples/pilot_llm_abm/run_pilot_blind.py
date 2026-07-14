"""
run_pilot_blind.py — Pilota question-blind: schema di relazioni GENERICO
(nessuna relazione cucita sulla domanda), estrazione esaustiva, inverse
automatiche, catene fino a 3 hop. Risultato: 10/10 (confidence più
basse del run question-aware: il carico raddoppia con le inverse).
"""
import sys, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np
from bsm.memory.vsa import WorkingMemory, hamming
from bsm.memory.encoder.entity_encoder import _minhash_sketch
from bsm.memory.reasoning_engine import calibrated_confidence

D, SK = 2048, 256
HERE = Path(__file__).resolve().parent


def norm(s):
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def tokens(s):
    return {w.rstrip("s") for w in norm(s).split() if w and w not in
            {"of", "in", "and", "on", "to", "by", "for", "with"}}


def answer(q):
    wm = WorkingMemory(D)
    ents, sketches = [], []
    for s, r, o in [t[:3] for t in q["triples"]]:
        wm.store(s, r, o)
        wm.store(o, r + "_inv", s)
        for name in (s, o):
            if name not in ents:
                ents.append(name)
                sketches.append(_minhash_sketch(tokens(name), SK))

    def ground(anchor):
        probe = _minhash_sketch(tokens(anchor), SK)
        d = [hamming(probe, sk) for sk in sketches]
        i = int(np.argmin(d))
        return ents[i], 1 - 2 * d[i] / SK

    plan, conf = q["plan"], 1.0
    if plan["type"] == "chain":
        node, j = ground(plan["anchor"])
        if j <= 0.1:
            return None, 0.0
        for rel in plan["chain"]:
            name, dist = wm.query(node, rel)
            c = calibrated_confidence(dist, D)
            if c < 0.55:
                return None, 0.0
            conf *= c
            node = name
        return node, conf
    years = {}
    for a in plan["anchors"]:
        node, _ = ground(a)
        y, dist = wm.query(node, plan["rel"])
        conf *= calibrated_confidence(dist, D)
        m = re.search(r"\d{4}", y)
        years[a] = int(m.group()) if m else None
    if not all(years.values()):
        return None, 0.0
    return min(years, key=years.get), conf


if __name__ == "__main__":
    data = json.loads(
        (HERE / "pilot_extraction_blind.json").read_text())["questions"]
    ok = 0
    for q in data:
        ans, conf = answer(q)
        hit = ans is not None and norm(q["gold"]) in norm(ans)
        ok += hit
        print(f"  {'OK ' if hit else 'MISS'} {q['q'][:48]:48s} -> "
              f"{(ans or '—')[:40]:40s} c={conf:.2f}")
    print(f"\n  question-blind end-to-end: {ok}/{len(data)}")
