"""
proofwriter_eval.py — ProofWriter (OWA, attributi, no negazione nel
corpus): l'inferenza logica può poggiare sull'algebra XOR?

Divisione rigorosa dei livelli (vedi docs/hotpotqa_report.md):
  Livello A (banale by design): il linguaggio di ProofWriter è
      controllato → parsing con 4 pattern, copertura ~100%.
  Livello B (sotto esame): i fatti vivono SOLO nella traccia olografica
      XOR; il forward-chaining usa come UNICO oracolo il test di
      membership algebrico:
          member(s, attr)  ⇔  hamming(chiave(s,is)⊕item(attr), T) < soglia
      Le regole derivano fatti nuovi che vengono ri-scritti nella
      traccia (⇒ il rumore cresce con le derivazioni: è parte del test).

Predizione: True se il fatto della domanda è derivabile, False se la
domanda è negata e il fatto è derivabile, Unknown altrimenti (OWA).

Uso: python proofwriter_eval.py [n_per_depth]
"""

import sys, re, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow.parquet as pq

from bsm.memory.vsa import WorkingMemory, bind_xor, hamming

PARQUET = ("/tmp/claude-1000/-var-www-html-BitKore/"
           "c904bff8-7b97-4d4b-9e76-49f65ca6a95e/scratchpad/"
           "proofwriter_val.parquet")

RE_FACT = re.compile(r"^(\w+) is (\w+)\.?$")
RE_RULE_IF = re.compile(
    r"^If (?:someone|something) is (.+?) then (?:they|it) (?:are|is) "
    r"(\w+)\.?$")
RE_RULE_PEOPLE = re.compile(r"^(?:All )?(.+?) (?:people|things) are (\w+)\.?$")
RE_Q = re.compile(r"^(\w+) is (not )?(\w+)\.?$")


def parse_theory(theory: str):
    """→ (fatti [(ent, attr)], regole [([premesse], concl)], copertura)"""
    facts, rules, parsed, total = [], [], 0, 0
    for s in re.split(r"(?<=\.)\s+", theory.strip()):
        s = s.strip()
        if not s:
            continue
        total += 1
        m = RE_FACT.match(s)
        if m:
            facts.append((m.group(1).lower(), m.group(2).lower()))
            parsed += 1
            continue
        m = RE_RULE_IF.match(s)
        if m:
            prem = re.split(r" and ", m.group(1))
            rules.append(([p.strip().lower() for p in prem],
                          m.group(2).lower()))
            parsed += 1
            continue
        m = RE_RULE_PEOPLE.match(s)
        if m:
            prem = re.split(r",\s*| and ", m.group(1))
            rules.append(([p.strip().lower() for p in prem],
                          m.group(2).lower()))
            parsed += 1
            continue
    return facts, rules, parsed / max(total, 1)


class AlgebraicProver:
    """Forward-chaining il cui unico oracolo è la membership algebrica.

    La conoscenza è UNA traccia olografica XOR; member() non consulta
    mai una lista di fatti: confronta l'hypervector del fatto candidato
    con la traccia (correlazione di maggioranza)."""

    def __init__(self, state_dim: int = 4096, z_min: float = 3.0):
        self.wm = WorkingMemory(state_dim)
        self.d = state_dim
        self.z_min = z_min
        self._entities: set = set()

    def add_fact(self, ent: str, attr: str):
        self.wm.store(ent, "is", attr)
        self._entities.add(ent)

    def _fact_hv(self, ent: str, attr: str):
        return bind_xor(self.wm._key(ent, "is"), self.wm.items.add(attr))

    def member(self, ent: str, attr: str) -> bool:
        """Test algebrico: il fatto è nella traccia olografica?"""
        if self.wm._trace is None:
            return False
        dist = hamming(self._fact_hv(ent, attr), self.wm._trace)
        z = (self.d / 2 - dist) / (np.sqrt(self.d) / 2)
        return z >= self.z_min

    def forward_chain(self, rules, max_rounds: int = 6):
        """Applica le regole fino al punto fisso (o max_rounds).
        Le premesse sono verificate SOLO via member()."""
        derived = set()
        for _ in range(max_rounds):
            new = []
            for ent in self._entities:
                for premises, concl in rules:
                    if (ent, concl) in derived or self.member(ent, concl):
                        continue
                    if all(self.member(ent, p) for p in premises):
                        new.append((ent, concl))
            if not new:
                break
            for ent, attr in new:
                self.add_fact(ent, attr)
                derived.add((ent, attr))
        return derived


def eval_row(row, state_dim=4096):
    facts, rules, coverage = parse_theory(row["theory"])
    mq = RE_Q.match(row["question"].strip())
    if not mq or coverage < 0.99:
        return None                       # fuori grammatica: dichiarato
    ent, neg, attr = mq.group(1).lower(), bool(mq.group(2)), \
        mq.group(3).lower()

    prover = AlgebraicProver(state_dim=state_dim)
    for e, a in facts:
        prover.add_fact(e, a)
    prover.forward_chain(rules)

    provable = prover.member(ent, attr)
    if neg:
        pred = "False" if provable else "Unknown"
    else:
        pred = "True" if provable else "Unknown"
    return pred == str(row["answer"]), row["QDep"], len(facts), len(rules)


def main():
    n_per_depth = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    rows = pq.read_table(PARQUET).to_pylist()
    rows = [r for r in rows if r["id"].startswith("AttNoneg")]

    print(f"ProofWriter OWA AttNoneg — {n_per_depth} domande per depth")
    print(f"  {'depth':>6} {'accuracy':>9} {'coperte':>8} "
          f"{'fatti+regole medi':>18}")
    results = {}
    for depth in (0, 1, 2, 3, 5):
        sample = [r for r in rows if r["config"] == f"depth-{depth}"]
        sample = sample[:n_per_depth * 3]      # margine per gli skip
        outs, skipped = [], 0
        for r in sample:
            if len(outs) >= n_per_depth:
                break
            out = eval_row(r)
            if out is None:
                skipped += 1
                continue
            outs.append(out)
        if not outs:
            print(f"  {depth:>6}   (nessuna domanda coperta)")
            continue
        acc = sum(o[0] for o in outs) / len(outs)
        nf = np.mean([o[2] + o[3] for o in outs])
        print(f"  {depth:>6} {acc:>8.0%} {len(outs):>5}/{len(outs)+skipped:<3}"
              f" {nf:>15.1f}")
        results[depth] = {"acc": acc, "n": len(outs), "skipped": skipped}

    # baseline: classe di maggioranza sulle stesse domande
    answers = [str(r["answer"]) for r in rows[:2000]]
    maj = max(set(answers), key=answers.count)
    print(f"  baseline classe di maggioranza ('{maj}'): "
          f"{answers.count(maj)/len(answers):.0%}")
    Path("proofwriter_results.json").write_text(json.dumps(results, indent=2))
    print("  → proofwriter_results.json")


if __name__ == "__main__":
    main()
