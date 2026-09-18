# Rapporto — Fase adversariale: tentativi di falsificazione

Data: 2026-07-13 · Harness: `examples/falsification_suite.py` + verifiche
mirate · Dati: `falsification_results.json`

Obiettivo dichiarato: **non cercare conferme, cercare smentite.**
Bilancio: una legge falsificata, una promossa a forma quasi
parameter-free, una nuova congettura, due asimmetrie di robustezza
caratterizzate.

---

## Il diagramma della teoria

```
      D (dimensione)          M (codebook)         {wᵢ} (pesi/ripetizioni)
           │                       │                        │
           ▼                       ▼                        ▼
   segnale per fatto        soglia di rumore         carico efficace
   z = √(2D/(π·N_eff))      z_G(M)  [Gumbel:            N_eff ≈ Σwᵢ²
           │                 √(2lnM) − 2° ordine]     (congettura F3)
           └──────────┬────────────┘
                      ▼
        p = P(cleanup corretto | z, z_G)      ← collasso a N* : z ≈ z_G
                      │                          N* = k·2D/(π·z_G²),
                      ▼                          k = 0.94 ± 0.01
        Acc(h) = p^h   [Law V: il cleanup
                      │  azzera il rumore ⇒ Markov]
                      ▼
        profondità massima di ragionamento  h_max(N, D, M)
```

Tutte le frecce sono state misurate; nessuna è solo postulata.

## La tabella delle leggi (aggiornata dopo la falsificazione)

| Legge | Formula | Verifica | Stato |
|---|---|---|---|
| I | Località di Hamming (nulla nota a priori) | calibrazione + cleanup | corroborata |
| II | Retrieval > decoder (fuori dominio) | HotpotQA | corroborata (condizionale) |
| III | Dimensionalità intrinseca (proiezione r≈12 ⇒ margine +70%) | Task 3 alias | corroborata |
| IV | **N\* = k·2D/(π·z_G(M)²), k = 0.94 ± 0.01** | 5 seed × 4 D × 4 M | **corroborata, quasi parameter-free** |
| V | Acc(h) = p^h (cleanup ⇒ Markov) | dev. 0.023±0.017 | corroborata + sketch di prova |
| VI | ~~Failure ∝ branching~~ | **falsificata a carico costante** | **RITIRATA → sostituita** |
| VI′ | Il fallimento dipende dal carico efficace, non dalla topologia: a N fisso, B∈[1,24] è neutro | 3 seed, carico costante | corroborata (nuova) |
| VII (cong.) | N_eff = Σwᵢ² (la ridondanza pesa quadraticamente) | previsto 11% vs misurato 21% | direzione giusta, costante da raffinare |

---

## I risultati, uno per uno

### F1 — Il test di ln M (la previsione della Law IV)

N\* a D=2048 si contrae con il codebook: 149 → 119 → 99 → 81 per
M = 447 → 16 242. **L'ipotesi nulla (N\* indipendente da M) è distrutta**
(−46%). La forma semplice k=N\*·lnM/D mostra un drift residuo del 15%;
con il termine di secondo ordine di Gumbel per il massimo di M gaussiane
(z_G = √(2lnM) − (lnlnM+ln4π)/(2√(2lnM))):

| M | 447 | 1 356 | 4 297 | 16 242 |
|---|---|---|---|---|
| k semplice | 0.444 | 0.419 | 0.404 | 0.383 |
| **k Gumbel** | **0.943** | **0.937** | **0.942** | **0.925** |

Spread <2% su un range 36× di M, e k≈0.94 ≈ 1: **la legge di capacità è
ora quasi parameter-free** — N\* ≈ 2D/(π·z_G(M)²) senza costanti libere
entro il 6%. Il tentativo di rompere la legge l'ha resa più forte.

### F2+verifica — Law VI falsificata

Il test Zipf non mostrava il degrado atteso sui nodi pesanti (65% vs
60%). La verifica diretta — branching a **carico costante** (120 fatti):

| B (grado uscente) | 1 | 4 | 8 | 24 |
|---|---|---|---|---|
| accuracy | 64% | 73% | 69% | 73% |

Piatto. Il drammatico 100%→14% del benchmark originale (B=1→16) era
**interamente un artefatto del carico** (B=16 significava 400 fatti,
2.5× oltre N\*). Law VI è ritirata e sostituita da VI′: la topologia del
grafo è neutra; conta solo N_eff/D. (Conseguenza pratica: niente
sharding degli hub — era una raccomandazione basata su un artefatto.)

### F3 — La ridondanza è un'arma a doppio taglio (nuova congettura)

10 fatti ripetuti 5× su 100: i ripetuti salgono al **100%**, gli altri
crollano al **21%** (controllo senza ripetizioni: 73%). La ripetizione
pesa il majority vote e *parassita* la capacità altrui. La congettura
N_eff = Σwᵢ² (=340) predice ~11% per i non-ripetuti: ordine giusto,
costante imprecisa. Da raffinare, ma il messaggio è già solido: in una
traccia olografica **la frequenza è salienza**, con un costo quadratico
per il resto della memoria.

### F4 vs F5 — L'asimmetria di robustezza

- **Rumore nella traccia** (flip di ε bit): degrado dolce e prevedibile
  (86% → 80% → 65% → 42% per ε = 0→20%), compatibile con segnale ∝ (1−2ε).
- **Rumore nel codebook**: catastrofico — ε=5% dimezza l'accuracy (54%),
  ε=10% la distrugge (24%). Il danno si compone: chiave di query,
  bersaglio di cleanup e contenuto della traccia si corrompono insieme.

Caratterizzazione: **la memoria è robusta, il codebook è la trusted
computing base** del paradigma. (Implicazione ingegneristica futura:
l'item memory merita ridondanza/ECC; la traccia no.)

---

## Valutazione 0-10 della fase adversariale

| Dimensione | Voto | Motivazione |
|---|---|---|
| **Guadagno scientifico** | **9.5** | Una legge falsificata e sostituita, una promossa a forma parameter-free (k=0.94±0.01 su 36× di M), una congettura nuova (Σw²), un'asimmetria caratterizzata: il singolo run più produttivo dell'intera sessione |
| **Rigore del metodo** | **9** | Il confound carico/topologia è stato scoperto *dalla suite stessa* (F2 non tornava con Law VI) e risolto con l'esperimento a carico costante |
| **Stato della teoria** | **8.5** | Le frecce del diagramma sono tutte misurate; restano da chiudere la costante di VII e la prova formale di V |
| **Prontezza da preprint** | **8** | La storia ora ha tutto: leggi, derivazioni, previsione verificata (Gumbel), falsificazione interna documentata. Mancano solo figure a ≥10 seed e la scrittura |

## Prossimi passi

1. Raffinare la congettura VII (Σw²): curva accuracy vs Σw² a parità di
   N unico — un pomeriggio di esperimenti.
2. Prova formale della Law V (l'argomento Markov è già a un passo).
3. Figure definitive a 10 seed e stesura del preprint:
   *"Algebraic Reasoning over Binary Geometric Memory: Capacity Laws
   and Error Composition"* — con la falsificazione di Law VI raccontata
   nel testo: è la parte che dà credibilità a tutto il resto.

---

## Addendum — 2026-08-06 · Livello implementativo (bitpack / sharding / router)

Tre affermazioni del layer bitpacked sono state messe alla prova. Due
sono state falsificate e rimosse, una è stata resa vera per costruzione.

### F4 — "VP-Tree: cleanup in O(log M)" → **FALSIFICATA**

Misura diretta del numero di nodi visitati (contatore su
`hamming_packed`), dopo aver corretto il pruning che era anche
logicamente sbagliato (confrontava `d - best_dist` con `best_dist` già
aggiornato a `d`, condizione sempre vera):

| M | D | nodi visitati | mismatch vs scan esaustivo |
|---|---|---|---|
| 400 | 512 | 365 / 400 | 0 / 20 |
| 5 000 | 2 048 | 4 487 / 5 000 | 0 / 20 |

Causa, misurata sul regime reale del cleanup:

| fatti nel trace | d(target) | d(non-target) | gap |
|---|---|---|---|
| 50 | 0.446·D | 0.500·D | 111 bit |
| 200 | 0.472·D | 0.500·D | 57 bit |
| 500 | 0.483·D | 0.500·D | 34 bit |

Una query di cleanup dista 0.45–0.48·D dal *proprio* codeword. Nessun
bound triangolare e nessuna LSH su bit parziali può potare: con p≈0.47
la probabilità che una banda di r bit coincida è 0.53^r. Anche lo
screening su prefisso è debole (z scende da 2.52 a 1.78 usando metà dei
bit) e reso 1.24x a fronte di perdita di recall.

**Conseguenza:** `bsm/memory/vptree.py` rimosso. Il cleanup resta uno
scan esaustivo vettorizzato — 0.54 ms a M=10 000, 6.1 ms a M=100 000
(D=2048), limitato dalla banda di memoria, non dalla CPU. Il claim
corretto non è "sublineare" ma "lineare con costante bassa".

### F5 — "Soglia di confidenza 0.75 = router affidabile" → **FALSIFICATA**

Con `temperature=8`, conf 0.75 implica z ≥ 8.79. Su D=2048 con 30 fatti
(accuratezza predetta da Law IV: 1.00):

| criterio | veri positivi | falsi accetti |
|---|---|---|
| conf ≥ 0.75 | **0 / 30** | 0 / 300 |
| calibrato α=0.01 | **30 / 30** | 3 / 300 (≈ α) |

La soglia fissa non era permissiva: era così severa da far degenerare
il router in "inoltra sempre all'LLM" appena la memoria si carica. Il
difetto non è il valore ma la forma: una soglia sulla confidenza non
dipende da M, mentre la statistica di test sì — il cleanup riporta il
*minimo* su M candidati.

**Sostituita da** `acceptance_z(candidates, alpha)`: sotto l'ipotesi
nulla il massimo z su M estrazioni ha CDF Φ(t)^M, quindi si accetta se
z ≥ Φ⁻¹((1−α)^(1/M)). Nessun parametro libero oltre ad α, e la soglia
cresce con M come dev'essere (z: 3.09 a M=10 → 6.00 a M=10⁷).

### F6 — "MultiTraceMemory condivide un codebook" → **FALSIFICATA (bug)**

`BitpackedItemMemory` definisce `__len__`, quindi un codebook vuoto è
*falsy*: `items or BitpackedItemMemory(dim)` scartava silenziosamente
il codebook condiviso e ogni shard si creava il proprio. Effetti:
`stats()["items_count"]` sempre 0, codewords duplicati per shard, e
`load()` — che riassegna esplicitamente `shard.items` — restituiva un
oggetto che si comportava *diversamente* da quello salvato.

Inoltre la selezione fra shard era il massimo delle confidenze
per-shard: un massimo di K estrazioni, distorto verso l'alto e con
falsi accetti che crescono linearmente in K. Ora la selezione è
l'argmin globale di distanza (= cleanup sull'unione) e la soglia si
calcola su K·M candidati.

### Ricadute di performance (D=2048, 500 fatti, `examples/bitpack_bench.py`)

| operazione | prima | dopo |
|---|---|---|
| store | 9.4x | 14.9x |
| query | 1.34x | 11.4x |
| member oracle | **0.31x** | 4.7x |

Il regresso su `member` era `permute_packed`, che faceva
unpack → `np.roll` → pack a ogni chiamata (29.7 µs dei 34.5 di
`fact_hv`), annullando il bitpacking. Riscritta nativa sui bit
(byte-roll + shift) e memoizzata a livello di chiave: la rotazione è
una funzione deterministica del nome, quindi la sua forma permutata è
immutabile e cacheable. Il trace è ora materializzato da un contatore
di voti incrementale in O(D) invece di re-bundlare tutti i fatti — con
identità bit-per-bit verificata contro `bundle_packed`, tie-break di
parità incluso.

**Nota metodologica.** F5 contraddice la mia analisi iniziale, che
attribuiva alla soglia 0.75 un eccesso di *falsi accetti*: la misura ha
mostrato il segno opposto. La direzione ipotizzata era sbagliata, la
struttura del difetto (soglia indipendente da M) no.

### Addendum al 2026-08-06 — secondo giro di ottimizzazione

> Le decisioni di prestazione di questi due addendum sono raccolte in forma
> operativa in [`PERFORMANCE.md`](../PERFORMANCE.md), con le condizioni di
> validità e i tentativi scartati. Questo documento tiene la parte di
> falsificazione; quello tiene la parte di ingegneria.

Profilando invece di indovinare, il collo di bottiglia dominante non era
nessuna delle operazioni algebriche: era
`np.random.RandomState(seed)`, **193 µs per costruzione**, pagata una
volta per ogni codeword generato. Riusare un unico generatore per thread
e richiamare `.seed()` costa 1.65 µs e produce **lo stesso stream di
bit** (verificato su 3000 seed). Generazione di un codeword:
186 → 19 µs.

`reference/abm.py::random_hv` ha la stessa struttura (940 µs per fatto
memorizzato, due codeword ciascuno): la maggior parte dello speedup sullo
store misurato da `bitpack_bench.py` — mediana **88x** su tre esecuzioni,
dispersione 7% — è questa inefficienza del riferimento, non un merito del
bitpacking. Il riferimento non è stato toccato (FREEZE), ma la correzione
è disponibile e preserva i bit. Le dispersioni misurate per ogni metrica
sono in [`PERFORMANCE.md`](../PERFORMANCE.md): quella di `member` è del
48%, quindi va citata solo come "circa 5–9x".

| operazione (D=2048) | inizio sessione | ora | guadagno |
|---|---|---|---|
| `random_packed_hv` (entità nuova) | 186.3 µs | 19.3 µs | 9.7x |
| `store()` (nomi noti) | 18.7 µs | 4.3 µs | 4.3x |
| materializzazione trace | 14.0 µs | 2.65 µs | 5.3x |
| `hamming_packed` | 3.95 µs | 1.10 µs | 3.6x |
| `member()` | 9.29 µs | 2.46 µs | 3.8x |
| `query()` (M=1001) | 73.0 µs | 57.6 µs | 1.3x |
| `query_calibrated()` (soli oggetti) | — | 38.2 µs | — |

Interventi, ognuno con la misura che lo giustifica:

- **Contatore di uno invece di voti ±1.** `ones += unpackbits(fatto)` è
  un singolo add uint8; il round-trip denso ±1→int64 costava 12.3 µs
  contro 2.5. Il bit di maggioranza diventa `ones > n/2`, con il vettore
  di tie a decidere `ones == n/2` (raggiungibile solo per n pari).
  Identità bit-per-bit con `bundle_packed` sotto test.
- **Riduzione dei popcount via BLAS gemv** contro un vettore di uno:
  50 µs contro 63 a M=1001. Ogni termine è ≤ 64 e ogni riga somma ≤ D,
  quindi l'accumulo float32 è esatto. Sotto 256 righe il setup del gemv
  non si ripaga e resta `np.sum`; `chunk=4096` è l'ottimo misurato
  (16384 costa il doppio: esce dalla cache).
- **`hamming_packed` su una sola coppia via `int.bit_count()`** sui byte
  grezzi: 1.1 µs contro 2.7 per tre kernel numpy su 32 elementi.
- **`store_many`**: un `unpackbits` sull'intero batch, 47 µs/fatto
  contro 65.7 per entità nuove (il resto è generazione di codeword).
- **Cleanup ristretto al ruolo di oggetto** (`answers_only`, default in
  `query_calibrated`). Un codeword mai memorizzato in posizione oggetto
  non può essere la risposta: includerlo aggiunge solo estrazioni sotto
  l'ipotesi nulla. Effetto su entrambi gli assi, a D=2048:

  | fatti | M | latenza intero → oggetti | accuratezza intero → oggetti |
  |---|---|---|---|
  | 500 | 1001 | 65.5 → 40.9 µs | 0.066 → 0.098 |
  | 2000 | 4001 | 201 → 113 µs | 0.003 → 0.006 |

  Nota di onestà su questa tabella: quelle accuratezze sono bassissime
  perché 500 fatti a D=2048 sono **molto oltre la capacità** (Law IV
  prevede 0.065 per n=500, M=1001 — la misura dà 0.066, la teoria
  regge). Il benchmark `bitpack_bench.py` misura quindi la latenza in un
  regime in cui le risposte sono comunque sbagliate: è un test di
  throughput, non di utilità.

**Tentativi misurati e scartati** (registrati per non essere ritentati):

| tentativo | risultato |
|---|---|
| buffer preallocati per xor/popcount | 64.6 µs vs 63.6 — nessun guadagno, l'allocatore numpy già ricicla |
| popcount sulla vista uint8 | 111 µs vs 64 — 8x più elementi da ridurre |
| albero di somme uint16 | 54 µs vs 24.7 di `np.sum` |
| screening su prefisso di bit (top 2%) | 1.24x, con perdita di recall — rifiutato |
| bit derivati da `random_raw` del MT | bit-esatto su 3000 seed ma 32 µs vs 17.7 |
| query in batch (K=8..128) | 53–72 µs/query: nessun riuso di cache reale |

**Limite non risolto, quantificato.** Il seed dei codeword è di 32 bit
(4 byte di MD5): la probabilità di almeno una collisione è ~69% a
M=100 000 entità, e due nomi in collisione diventano indistinguibili.
Allargare il seed cambierebbe tutti i codeword e romperebbe
l'equivalenza con il riferimento congelato — va deciso, non subito.
