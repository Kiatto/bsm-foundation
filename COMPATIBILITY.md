# COMPATIBILITY.md — matrice di compatibilità misurata

Ogni riga è stata **eseguita**, non dedotta. Suite completa: `bsm/tests` +
`reference`, 166 test. Data: 2026-08-06, commit `db137b8` + modifiche di sessione.

Riprodurre una riga:

```bash
uv run --python 3.12 --with "numpy==2.0.2" --with pytest --with fastapi --with httpx \
  python -m pytest bsm/tests reference -q
```

## Matrice

| Python | NumPy | popcount | Test | Prestazioni |
|---|---|---|---|---|
| 3.9 | qualsiasi | — | **30 falliti** | non supportato |
| 3.10 | 1.24.4 | LUT | 166 passed | **degradate**, vedi sotto |
| 3.10 | 1.26.4 | LUT | 166 passed | **degradate** |
| 3.11 | 1.26.4 | LUT | 166 passed | **degradate** |
| 3.11 | 2.0.2 | `bitwise_count` | 166 passed | nominali |
| 3.12 | 2.0.2 | `bitwise_count` | 166 passed | nominali |
| 3.12 | 2.5.1 | `bitwise_count` | 166 passed | nominali |
| 3.13 | 2.5.1 | `bitwise_count` | 166 passed | **configurazione di riferimento** |

## Due livelli di supporto, non uno

La distinzione conta più della matrice stessa: **funzionante** e **performante
come documentato** non coincidono.

- **Supportato (funzionale):** Python ≥ 3.10, NumPy ≥ 1.24. Tutti i 166 test
  passano, i risultati sono identici bit per bit.
- **Raccomandato (prestazioni nominali):** Python ≥ 3.10, **NumPy ≥ 2.0**. È la
  configurazione a cui si riferiscono tutti i numeri di
  [`PERFORMANCE.md`](PERFORMANCE.md).

Costo misurato del path senza `np.bitwise_count` (Python 3.12, D=2048, mediana di
3 ripetizioni):

| operazione | NumPy 1.26 (LUT) | NumPy 2.5 (`bitwise_count`) | penalità |
|---|---|---|---|
| `batch_hamming_packed` (M=1001) | 598.9 µs | 49.6 µs | **12.1x** |
| `query_calibrated` | 308.4 µs | 34.7 µs | **8.9x** |
| `member` | 2.31 µs | 2.52 µs | nessuna |

`member` non è toccato perché usa `int.bit_count()` di Python, non il popcount di
numpy. Il cleanup sì: su NumPy 1.x la LUT espande ogni word uint64 in 8 lookup
uint8, cioè 8x più elementi da ridurre.

**Non abbiamo alzato il floor a `numpy>=2.0`.** Funziona su 1.24 e escludere chi è
bloccato su NumPy 1.x per un fattore di prestazioni sarebbe sproporzionato. Se
emergesse un utente reale su NumPy 1.x per cui la latenza è un problema, la
risposta è alzare il floor o pubblicare un extra, non ottimizzare la LUT.

## Perché Python 3.9 non è supportato

`pyproject.toml` dichiarava `requires-python = ">=3.9"`. **Era falso, e da prima
di questa sessione.** `int.bit_count()` (Python ≥ 3.10) è usato in codice
committato dal commit `0b81b27`: `bsm/memory/store/memory_store.py:108`,
`bsm/__init__.py:207`, `bitpack.py`, e cinque file in `bsm/experiments/`. Sotto
3.9 fallivano 30 test.

Il floor è stato corretto a `>=3.10`. Non è stata rimossa una compatibilità
funzionante: è stata resa vera un'affermazione che non lo era. Python 3.9 è anche
EOL da ottobre 2025.

Nessun fallback è stato aggiunto per 3.9 di proposito: un floor sbagliato deve
fallire in modo rumoroso, non degradare in silenzio. Per numpy vale il contrario —
lì il fallback esiste ed è testato, perché `numpy>=1.24` è una dipendenza dichiarata
che vogliamo davvero onorare.

## Cosa questa matrice non copre

Dichiarato per non essere sopravvalutato:

- **Un solo OS e una sola CPU** (Linux x86-64, Intel i7-9750H). Nessun test su
  macOS, Windows, ARM/Apple Silicon. Il codice non usa intrinseche esplicite, ma
  `packbits`/`bitwise_count` non sono stati verificati su altra endianness o
  architettura. `random_packed_hv` usa `np.packbits(...).view(np.uint64)`, che
  **dipende dall'endianness**: su una macchina big-endian i codeword differirebbero.
  Non verificato.
- **Un solo BLAS** (scipy-openblas 0.3.33). La voce A3 di `PERFORMANCE.md` dipende
  dal gemv; con MKL o un BLAS di riferimento il rapporto può cambiare.
- **Nessuna versione intermedia** di NumPy 2.1–2.4, né PyPy, né free-threaded
  CPython (dove il `RandomState` condiviso della voce C3 andrebbe riverificato).
- **Solo la suite di test.** Non è un test di carico né di lunga durata.

Aggiungere una riga solo dopo averla eseguita.

## In corso di copertura: la CI multi-piattaforma

`.github/workflows/compatibility.yml` riesegue a ogni commit le sette righe della
matrice qui sopra e aggiunge l'unica dimensione che il portatile di sviluppo non
può coprire: **macOS x86-64, macOS arm64 (Apple Silicon) e Windows**, ciascuno sul
path LUT (NumPy 1.26) e sul path `bitwise_count` (NumPy 2.5).

Un job dedicato confronta il **digest dei codeword** (`.github/codeword_digest.py`)
prodotto su ogni piattaforma e fallisce se non coincidono: se `packbits().view()`
si comportasse diversamente su arm64 o su Windows, un artifact prodotto su una
macchina non sarebbe leggibile su un'altra. Il digest è già verificato identico in
locale su Python 3.10–3.13 e NumPy 1.24–2.5
(`682d558d068a17519a54da9fc692cefc9a269ab981f70513061bac9e6a915ada`).

**Stato: in attesa della prima esecuzione.** Finché la CI non è passata, le righe
macOS e Windows non esistono: la regola sopra vale anche per chi l'ha scritta. Il
caveat big-endian resta comunque intatto — nessun runner GitHub è big-endian, e
nessuna quantità di CI gratuita lo risolve.
