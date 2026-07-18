"""abm — command-line interface (SDK tooling, not part of the frozen
specification).

    abm inspect triples.json [--dim 8192] [--grounding 0.93]
    abm demo

`triples.json` is a JSON array of [subject, relation, object] triples —
the natural output format of an LLM extractor.
"""

import argparse
import json
import sys


def cmd_inspect(args):
    from . import Memory, contract, report
    try:
        triples = json.loads(open(args.file).read())
    except Exception as e:
        print(f"error: cannot read {args.file}: {e}", file=sys.stderr)
        return 1
    bad = [t for t in triples
           if not (isinstance(t, (list, tuple)) and len(t) == 3)]
    if bad:
        print(f"error: {len(bad)} entries are not [s, r, o] triples "
              f"(first: {bad[0]!r})", file=sys.stderr)
        return 1
    mem = Memory(dim=args.dim)
    for s, r, o in triples:
        mem.store(str(s), str(r), str(o))
    print(report(mem, extractor_precision=args.grounding))
    print()
    print(contract(mem, grounding=args.grounding))
    return 0


def cmd_demo(args):
    from . import Memory, contract
    mem = Memory(dim=4096)
    facts = [("payment_service", "requires", "auth_service"),
             ("auth_service", "writes_to", "session_store"),
             ("session_store", "deployed_in", "eu_west")]
    for f in facts:
        mem.store(*f)
    print("stored:", *[f"  {s} --{r}--> {o}" for s, r, o in facts],
          sep="\n")
    ans, conf = mem.chain("payment_service", ["requires", "writes_to"])
    print(f"\nchain(payment_service, [requires, writes_to]) "
          f"= {ans} ({conf:.0%})\n")
    print(contract(mem, grounding=0.93))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="abm",
        description="ABM — algebraic memory runtime with predictive "
                    "Memory Contracts")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect",
                        help="load triples and print the Memory Contract")
    pi.add_argument("file", help="JSON array of [s, r, o] triples")
    pi.add_argument("--dim", type=int, default=8192,
                    help="trace size in bits (default 8192)")
    pi.add_argument("--grounding", type=float, default=1.0,
                    help="audited precision of your extractor (0..1)")
    pi.set_defaults(fn=cmd_inspect)

    pd = sub.add_parser("demo", help="30-second tour")
    pd.set_defaults(fn=cmd_demo)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
