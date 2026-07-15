"""Estrazione question-blind delle 40 domande del campione (PROTOCOL.md).
Scrive extractions.jsonl in modo incrementale; riprende da dove era.
La chiave API arriva da $OR_KEY, mai committata."""

import json, os, sys, time, urllib.request, urllib.error
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np

HERE = Path(__file__).resolve().parent
PARQUET = ("/tmp/claude-1000/-var-www-html-BitKore/"
           "c904bff8-7b97-4d4b-9e76-49f65ca6a95e/scratchpad/hotpot_val.parquet")
MODELS = ["nvidia/nemotron-3-ultra-550b-a55b:free",
          "nvidia/nemotron-3-nano-30b-a3b:free",
          "openai/gpt-oss-20b:free"]

RELS = ("directed_by, written_by, produced_by, starred, born_in, born_on, "
        "died_in, nationality, occupation, member_of, part_of, located_in, "
        "capital_of, founded_by, founded_in, created_by, published_in, "
        "released_in, genre, instance_of, known_for, spouse_of, child_of, "
        "works_for, plays_for, album_of, song_by, author_of, developed_by, "
        "based_on")

PROMPT = """Extract factual triples from the following encyclopedia paragraphs.
Rules:
- Output ONLY a JSON array of [subject, relation, object] triples, nothing else. No markdown fences.
- Use these relations when applicable: {rels}. Otherwise invent a short snake_case relation.
- Subjects/objects: canonical entity names as written (prefer the paragraph title for its main entity).
- Extract exhaustively: every fact in every paragraph, including dates, places, roles.

{paras}"""


def sample():
    t = pq.read_table(PARQUET)
    rng = np.random.RandomState(77)
    idx = rng.choice(t.num_rows, 200, replace=False)
    rows = t.take(idx).to_pylist()
    return [r for r in rows if r["type"] == "bridge"][:41]


def call(model, prompt, max_tok=16000):
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps({"model": model, "max_tokens": max_tok,
                         "reasoning": {"effort": "low"},
                         "messages": [{"role": "user",
                                       "content": prompt}]}).encode(),
        headers={"Authorization": "Bearer " + os.environ["OR_KEY"],
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        msg = json.load(resp)["choices"][0]["message"]
        txt = msg.get("content") or ""
        if "[" not in txt:                     # risposta finita nel reasoning
            txt = msg.get("reasoning") or txt
        return txt


if __name__ == "__main__":
    bridge = sample()[1:]                     # #0 = calibrazione, esclusa
    out = HERE / "extractions.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.open() if l.strip()}
    print(f"{len(bridge)} domande, {len(done)} già fatte", flush=True)
    for i, r in enumerate(bridge):
        if r["id"] in done:
            continue
        ctx = r["context"]
        paras = "\n\n".join(f"[{ti}] " + " ".join(s)
                            for ti, s in zip(ctx["title"], ctx["sentences"]))
        prompt = PROMPT.format(rels=RELS, paras=paras[:12000])
        triples, model_used, err = None, None, None
        for attempt in range(3):               # budget: max 3 tentativi
            model = MODELS[min(attempt, len(MODELS) - 1)]
            try:
                txt = call(model, prompt)
                s = txt[txt.find("["):txt.rfind("]") + 1]
                cand = json.loads(s)
                triples = [t3 for t3 in cand
                           if isinstance(t3, list) and len(t3) == 3]
                model_used = model
                break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"[:120]
                time.sleep(20 * (attempt + 1))
        rec = {"id": r["id"], "i": i, "triples": triples,
               "model": model_used, "error": None if triples else err}
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  [{i+1}/40] {r['id']} → "
              f"{len(triples) if triples else 'FAIL:' + str(err)} "
              f"({model_used})", flush=True)
        time.sleep(5)
    print("done", flush=True)
