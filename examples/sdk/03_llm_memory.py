"""Example 3 — LLM Memory: document → triples → ABM → queries.

Your LLM reads documents and emits (subject, relation, object) triples;
ABM stores them and answers with a predictive contract. Here the
"LLM output" is inlined so the example runs offline — replace
EXTRACTED with your extractor's output.

    python 03_llm_memory.py
"""

from abm import Memory, stats

# --- what your LLM extracted from the docs (question-blind) ----------
EXTRACTED = [
    ("payment_service", "requires", "auth_service"),
    ("payment_service", "owned_by", "team_payments"),
    ("auth_service", "writes_to", "session_store"),
    ("auth_service", "requires", "user_db"),
    ("session_store", "deployed_in", "eu_west"),
    ("user_db", "deployed_in", "eu_west"),
    ("checkout_page", "calls", "payment_service"),
    ("team_payments", "on_call", "alice"),
]

# --- ingest -----------------------------------------------------------
mem = Memory(dim=4096)
for s, r, o in EXTRACTED:
    mem.store(s, r, o)

# --- the projected contract, given your extractor's audited precision -
s = stats(mem, extractor_precision=0.93, hops=2)
print(f"facts: {s['facts']}  capacity: {s['estimated_capacity']}  "
      f"pressure: {s['pressure']}")
print(f"memory accuracy (theory): {s['expected_accuracy']:.0%}")
print(f"end-to-end with your extractor: {s['projected_accuracy']:.0%}")
print(f"dominant bottleneck: {s['dominant_error']}")
print()

# --- query it ---------------------------------------------------------
questions = [
    ("who is on call for payments?",
     lambda: mem.chain("payment_service", ["owned_by", "on_call"])),
    ("where does auth write state?",
     lambda: mem.query("auth_service", "writes_to")),
    ("what region is the session store in?",
     lambda: mem.query("session_store", "deployed_in")),
]
for text, ask in questions:
    answer, conf = ask()
    print(f"Q: {text}\nA: {answer}  (confidence {conf:.0%})\n")
