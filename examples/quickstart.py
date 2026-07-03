"""
quickstart.py — BSM Foundation in 15 lines.

Usage:
    python examples/quickstart.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bsm import BSM

bsm = BSM(encoder="hash", state_dim=256)

texts = [
    "the cat sat on the mat",
    "a dog ran in the park",
    "the weather is sunny today",
    "solve the equation for x",
]

for t in texts:
    state = bsm.encode(t)
    bsm.observe(state, {"text": t, "category": "story"})

state = bsm.encode("a cat is sleeping on a mat")
results = bsm.recall(state, k=2)

print("Top matches:")
for payload, dist, meta in results:
    print(f"  dist={dist}: {payload}")

print(f"\nInfo: {bsm.info()}")
print(f"Health: {bsm.health()}")
