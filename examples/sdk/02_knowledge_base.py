"""Example 2 — Knowledge Base with a Memory Contract.

Store a service-dependency knowledge base, then ask the memory itself
how reliable it will be — BEFORE running any query.

    python 02_knowledge_base.py
"""

from abm import Memory, contract, report

mem = Memory(dim=8192)

services = [f"service_{i}" for i in range(60)]
for i, s in enumerate(services):
    mem.store(s, "depends_on", services[(i + 7) % 60])
    mem.store(s, "owned_by", f"team_{i % 8}")
    mem.store(s, "deployed_in", f"region_{i % 3}")

# The contract: computed from the memory's own state, not measured.
print(contract(mem))
print()

# Use it
answer, conf = mem.chain("service_3", ["depends_on", "owned_by"])
print(f"service_3's dependency is owned by: {answer} ({conf:.0%})")

# Full diagnostic view (capacity, pressure, recommended size, ...)
print()
print(report(mem))
