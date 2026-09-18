"""Digest riproducibile dei codeword, per confrontare piattaforme diverse.

COMPATIBILITY.md dichiara che `random_packed_hv` usa
`np.packbits(...).view(np.uint64)`, che dipende dall'endianness: su una macchina
con endianness diversa i codeword differirebbero. Questo script stampa un digest
dei codeword generati; la CI lo calcola su ogni OS/architettura e verifica che
sia identico. Non prova nulla su big-endian (nessun runner GitHub lo e'): prova
che x86-64, arm64 e Windows producono gli stessi bit.
"""
import hashlib
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from bsm.memory.bitpack import random_packed_hv

NAMES = ["payment_service", "auth_service", "session_store", "requires", "writes_to"]
DIMS = [1024, 2048, 8192]

h = hashlib.sha256()
for dim in DIMS:
    for name in NAMES:
        h.update(random_packed_hv(name, dim).tobytes())
digest = h.hexdigest()

print(json.dumps({
    "digest": digest,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "byteorder": sys.byteorder,
    "python": platform.python_version(),
    "numpy": np.__version__,
}, indent=2))

Path("codeword_digest.txt").write_text(digest + "\n")
