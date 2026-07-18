"""ABM — Algebraic Binary Memory.

A deterministic algebraic memory runtime for compiled symbolic
knowledge. Five concepts:

    from abm import Memory

    mem = Memory()
    mem.store("payment_service", "requires", "auth_service")
    mem.query("payment_service", "requires")   # ("auth_service", 0.99)
    mem.chain("payment_service", ["requires", "writes_to"])
    print(contract(mem))                       # the Memory Contract

The package maps onto the frozen reference implementation
(FORMALISM v2.1): `abm.abm` is the executable specification,
`abm.inspector` is the Memory Contract API.
"""

from .abm import (Memory, ItemMemory, bind, bundle, permute, random_hv,
                  hamming, phi, confidence, capacity, predicted_accuracy,
                  z_gumbel, __version__)
from .inspector import stats, contract, report, aliasing

__all__ = ["Memory", "ItemMemory", "bind", "bundle", "permute",
           "random_hv", "hamming", "phi", "confidence", "capacity",
           "predicted_accuracy", "z_gumbel", "stats", "contract",
           "report", "aliasing", "__version__"]
