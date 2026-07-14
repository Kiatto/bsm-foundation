"""
run_pilot.py — Pilota di applicabilità LLM+ABM su HotpotQA (10 domande).

Divisione dei livelli (Law VIII):
  Livello A (LLM): triple + piano di query in pilot_extraction.json
      (estratte da un LLM leggendo TUTTI i contesti, gold + distrattori).
  Livello B (ABM): grounding MinHash + catena XOR + cleanup + confidence
      calibrata — INVARIATO rispetto ai benchmark interni.
  Livello C (controller): solo il confronto di anni per le domande di
      comparazione (dichiarato).

Risultato: 10/10 end-to-end (regex grounding: 7%; retrieval top-1: 13%).
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
    return {w for w in norm(s).split() if w}


def answer_question(q):
    wm = WorkingMemory(D)
    prov, ents, sketches = {}, [], []
    for s, r, o, src in q["triples"]:
        wm.store(s, r, o)
        prov[(s, r)] = src
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
            return None, 0.0, ""
        steps = [node]
        for rel in plan["chain"]:
            name, dist = wm.query(node, rel)
            c = calibrated_confidence(dist, D)
            if c < 0.55:
                return None, 0.0, ""
            conf *= c
            node = name
            steps.append(node)
        src = prov.get((steps[-2], plan["chain"][-1]), "")
        return f"{node} | {src}", conf, " -> ".join(steps)
    # compare_older: confronto al livello controller (dichiarato)
    years = {}
    for a in plan["anchors"]:
        node, _ = ground(a)
        y, dist = wm.query(node, plan["rel"])
        conf *= calibrated_confidence(dist, D)
        m = re.search(r"\d{4}", y)
        years[a] = int(m.group()) if m else None
    if not all(years.values()):
        return None, 0.0, ""
    oldest = min(years, key=years.get)
    return oldest, conf, str(years)


if __name__ == "__main__":
    data = json.loads((HERE / "pilot_extraction.json").read_text())["questions"]
    ok = 0
    for q in data:
        ans, conf, chain = answer_question(q)
        hit = ans is not None and norm(q["gold"]) in norm(ans)
        ok += hit
        print(f"  {'OK ' if hit else 'MISS'} {q['q'][:50]:50s} -> "
              f"{(ans or '—')[:40]:40s} c={conf:.2f}")
    print(f"\n  end-to-end: {ok}/{len(data)}")
