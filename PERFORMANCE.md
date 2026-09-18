# PERFORMANCE.md — registro delle decisioni di prestazione

Scopo: **non ripetere esperimenti già fatti.** Ogni voce è una decisione con la
misura che la giustifica. Le voci scartate valgono quanto quelle adottate — sono
la parte che fa risparmiare tempo.

## Ambiente delle misure

Tutti i numeri di questo file provengono da questo ambiente, salvo indicazione
contraria. Chi rimisura su un ambiente diverso aggiunga un blocco, non sovrascriva
questo: i **rapporti** sono in genere trasferibili, i valori assoluti no.

```
Data          : 2026-08-06
Commit        : db137b8 (+ modifiche non committate della sessione)
CPU           : Intel Core i7-9750H @ 2.60GHz, 12 thread
OS            : Linux 7.0.0-28-generic x86_64
Python        : 3.13.14 (CPython)
NumPy         : 2.5.1   (np.bitwise_count disponibile)
BLAS          : scipy-openblas 0.3.33.112.0
```

Perché questi campi e non altri: **NumPy ≥ 2.0** abilita `np.bitwise_count` (voci
A3, A5 — senza, il codice cade sul path LUT e i rapporti cambiano); il **BLAS**
determina la voce A3 (riduzione gemv); la **cache della CPU** determina la voce A4
(`chunk = 4096`); **CPython** determina A5 (`int.bit_count()` su big-int).

⚠️ **Questi numeri valgono solo su NumPy ≥ 2.0.** Su NumPy 1.x il codice funziona
(166/166 test) ma il cleanup è **12x più lento** — misurato, con la matrice
completa, in [`COMPATIBILITY.md`](COMPATIBILITY.md).

Regole per chi aggiunge una voce:

1. Nessuna voce senza una misura. "Sembra più veloce" non è una misura.
2. Indicare la **condizione** di validità (D, M, numero di fatti, hardware).
   Una misura senza regime non è confrontabile.
3. Dichiarare la **riproducibilità**: quante ripetizioni e con quale dispersione.
   Un numero da una singola esecuzione va marcato come tale.
4. Registrare anche i tentativi **falliti**, con il numero. È il motivo per cui
   questo file esiste.
5. Prima di ottimizzare, **profilare**. Il caso più costoso di questo progetto
   (`RandomState`, voce A1) non era in nessuna delle funzioni algebriche che
   sembravano il collo di bottiglia.

**Ambito di queste regole.** Servono a supportare decisioni di sviluppo, non a
produrre risultati statisticamente pubblicabili. La soglia del 10% è una soglia
decisionale interna, non una procedura statistica: non citarla come metodo in un
paper e non trattarla come un test di significatività. Le affermazioni destinate
a uscire dal repository stanno in [`docs/falsification_report.md`](docs/falsification_report.md),
che ha standard di evidenza diversi (più seed, ipotesi dichiarate, falsificazioni
esplicite). Questo file è il registro di chi deve decidere se una modifica vale il
suo costo.

Le misure sono single-thread salvo il gemv BLAS (voce A3). Ambiente completo in
testa al file.

**Queste cinque regole sono congelate insieme alla fase.** La soglia del 10% per
"riproducibile" è arbitraria: serve che *esista un numero*, non che sia il numero
giusto. Non spendere tempo a discutere se debba essere 8%, 15% o un coefficiente
di variazione — una regola abbastanza buona da evitare errori sistematici va
congelata, non ottimizzata. Cambiarla solo se produce una decisione sbagliata
osservata, non per renderla più elegante.

---

## Sinossi

Mediana di 5 ripetizioni per i microbench, 3 esecuzioni complete per gli speedup.
"Riproducibile" = dispersione (max−min)/mediana sotto il 10%.

| Decisione | Misura | Esito | Riproducibile? |
|---|---|---|---|
| A1 · RandomState riusato | 193 → 1.65 µs; codeword 186 → **18.9 µs** | adottata | **sì** (spread 3.1%) |
| A2 · contatore di uno | store 12.3 → 2.5 µs; trace 14.0 → **2.63 µs** | adottata | sì (spread 10.9%, al limite) |
| A3 · riduzione gemv BLAS | **47.5** vs 59.0 µs a M=1001 | adottata | **sì** (spread 1.8% / 2.4%) |
| A4 · `chunk = 4096` | 5.85 ms vs 11.82 senza chunk (M=100k) | adottata | sì (curva monotona su 6 valori) |
| A5 · `int.bit_count()` su coppia | **1.12** vs 2.67 µs | adottata | **sì** (spread 1.5%) |
| A6 · permute nativa + memoizzazione | 29.7 → 22.2 µs nativa; **0.14 µs** memoizzata | adottata | sì |
| A7 · `store_many` | **61.7 vs 73.2** µs/fatto = 1.19x su entità nuove | adottata | **no** — spread 69% / 40% |
| A8 · cleanup ristretto agli oggetti | 60.0 → **35.8 µs** (1.68x); accuratezza 0.072 → 0.100 | adottata | **sì** (spread 6–8%) |
| A9 · `lru_cache` su `acceptance_z` | query calibrata 74 → **37.1 µs** | adottata | **sì** (spread 3.0%) |
| B1 · buffer preallocati | 64.6 vs 63.6 µs | scartata | sì |
| B2 · popcount su vista uint8 | 111 vs 64 µs | scartata | sì |
| B3 · albero di somme uint16 | 54 vs 24.7 µs | scartata | sì |
| B4 · screening su prefisso | 1.24x, con perdita di recall | scartata | sì |
| B5 · bit da `random_raw` del MT | bit-esatto su 3000 seed, 32.4 vs 17.7 µs | scartata | sì (esattezza verificata) |
| B6 · query in batch (K=8..128) | 53–72 µs/query contro 60 a K=1 | scartata | sì |
| B7 · VP-Tree / indici metrici | 4487 nodi su 5000 visitati | scartata | sì (falsificata strutturalmente) |

### Riproducibilità degli speedup contro il reference

Tre esecuzioni complete di `bitpack_bench.py` a D=2048:

| metrica | valori | mediana | dispersione |
|---|---|---|---|
| store | 83.4 / 89.4 / 88.3 | **88x** | 7% |
| query | 12.55 / 12.48 / 12.23 | **12.5x** | 3% |
| member | 5.37 / 8.71 / 6.97 | **7x** | **48%** |

**Non citare lo speedup di `member` con più di una cifra significativa: "circa
5–9x".** La dispersione del 48% viene dal lato reference della misura. In una
sessione precedente lo store è stato osservato a 110x: era un valore estremo, la
mediana è 88x. Per lo stesso motivo la sinossi riporta mediane, non il best run.

---

## Come leggere i benchmark

`examples/bitpack_bench.py` stampa tre blocchi. Solo il secondo è confrontabile
fra run diversi:

| blocco | confrontabile fra D/M diversi? |
|---|---|
| latenza grezza + speedup | **no** — scala con D e M |
| normalizzato (per fatto, per candidate-bit, per bit) | sì |
| regime (pressure N/N*, accuratezza predetta) | è il contesto che rende leggibili gli altri due |

Due avvertenze permanenti stampate dal benchmark stesso:

- **Lo speedup contro il reference include una inefficienza nota del reference
  congelato** (voce A1). Va letto come "packed ABM contro il reference così
  com'è", non come il valore del bitpacking.
- Il costo **per bit scende al crescere di D** (11.2 → 6.9 → 8.9 ns/fact-bit a
  D=1024/2048/4096): a D piccolo domina l'overhead fisso per chiamata, quindi
  quel regime è *overhead-bound*, non *bandwidth-bound*. Non aspettarsi che le
  colonne normalizzate siano costanti.

---

## A — Adottate

### A1 · Riuso di un `RandomState` per thread invece di costruirne uno per codeword

- **Misura:** `np.random.RandomState(seed)` = **193 µs**; `rng.seed(seed)` su un
  oggetto riusato = **1.65 µs**. Codeword completo: 186 → 19 µs (**9.7x**).
- **Condizione:** qualsiasi D. Vale per ogni entità nuova, quindi per store,
  query e member su nomi mai visti.
- **Tradeoff:** il generatore è ora stato mutabile condiviso → `threading.local()`
  per isolarlo. Test dedicato verifica determinismo con 8 thread concorrenti.
- **Perché è la voce più importante:** era il costo dominante di *tutto il
  sistema* e non stava in nessuna funzione algebrica. Le ottimizzazioni fatte
  prima di profilare (bundling, permute) agivano sul 5% del tempo.
- **Bit invariati:** verificato, lo stream è identico. Nessun artefatto invalidato.

### A2 · Contatore di bit a uno invece di voti ±1 per il trace

- **Misura:** aggiornamento per store 12.3 → **2.5 µs**; materializzazione del
  trace 14.0 → **2.65 µs** (5.3x).
- **Condizione:** qualsiasi D. Il guadagno cresce col numero di store.
- **Come:** con `ones` bit a uno su `n` voti, la somma ±1 è `2*ones - n`, quindi
  il bit di maggioranza è `ones > n/2`; il vettore di tie decide `ones == n/2`,
  raggiungibile solo per `n` pari.
- **Tradeoff:** nessuno. Identità bit-per-bit con `bundle_packed` sotto test
  (n=1..24, pesi misti, entrambe le parità).

### A3 · Riduzione dei popcount via gemv BLAS

- **Misura:** M=1001, D=2048 → **50 µs** contro 63 di `np.sum`. M=100k → 5.23 ms
  contro 6.55.
- **Condizione:** **M ≥ 256**. Sotto quella soglia il setup del gemv non si
  ripaga (M=100: 11.8 µs contro 9.9) → il codice tiene `np.sum`.
- **Esattezza:** ogni termine è un popcount di word ≤ 64 e ogni riga somma ≤ D,
  quindi l'accumulo float32 è esatto e il cast a int32 è senza perdita.
- **Tradeoff:** dipendenza dal BLAS di numpy per una riduzione di interi. Il
  fallback `np.sum` resta nel codice per il path senza `bitwise_count`.

### A4 · `chunk = 4096` righe nello scan del cleanup

- **Misura** (M=100k, D=2048): 512→6.65 ms, 1024→6.19, 2048→5.89, **4096→5.85**,
  16384→11.82, senza chunk→10.76.
- **Condizione:** ottimo piatto fra 2048 e 4096; oltre, il buffer esce dalla
  cache e il costo raddoppia.
- **Beneficio collaterale:** il temporaneo di picco passa da 25.6 MB a 0.8 MB.

### A5 · `hamming_packed` su coppia singola via `int.bit_count()`

- **Misura:** **1.10 µs** contro 2.67 di tre kernel numpy su 32 elementi (2.4x).
- **Condizione:** solo per il confronto di **una** coppia. Per molti vettori usare
  `batch_hamming_packed`: la versione big-int perde subito.
- **Perché funziona:** un'unica chiamata a livello interprete invece di tre
  kernel su array minuscoli.

### A6 · `permute_packed` nativa sui bit + memoizzazione delle chiavi

- **Misura:** permute nativa 29.7 → 22.2 µs (**solo 1.4x**: a D=2048 domina
  l'overhead numpy su 256 byte). La memoizzazione di chiavi e permutazioni porta
  il costo effettivo a **0.14 µs** ed è ciò che ha spostato la query da 1.34x a
  ~11x contro il reference.
- **Condizione:** il codeword è funzione deterministica del nome, quindi la sua
  forma permutata è immutabile e cacheabile. I risultati vanno trattati come
  read-only.
- **Lezione:** riscrivere il kernel rendeva l'1.4x; **non chiamarlo** rendeva il
  resto. Prima di ottimizzare una funzione, chiedersi se serve invocarla.

### A7 · `store_many` — un `unpackbits` per batch

- **Misura, con riserva:** mediana **61.7 µs/fatto** contro 73.2 di `store()` per
  entità nuove (1.19x), ma **dispersione 69% e 40%** su 5 ripetizioni: il rapporto
  non è affidabile a questa precisione. Su nomi già noti il guadagno è netto e
  stabile: 1.6 contro 4.5 µs.
- **Condizione:** su entità nuove il costo è dominato dalla generazione dei
  codeword (voce A1, ~19 µs per nome), che è anche la fonte della dispersione. Il
  vantaggio di `store_many` si vede solo quando i nomi sono già nel codebook.
- **Nota:** una misura precedente riportava 47.0 vs 65.7 µs/fatto da una singola
  esecuzione. Con le ripetizioni il rapporto scende da 1.4x a 1.19x — esempio di
  perché la regola 3 esiste.
- **Equivalenza:** trace, answers e query identici a `store()` uno per uno; con
  `weight > 1` cambia solo l'ordine interno dei fatti, da cui il bundle di
  maggioranza non dipende.

### A8 · Cleanup ristretto ai codeword in ruolo di oggetto

- **Misura** (D=2048, 500 fatti, mediana di 5 ripetizioni): latenza 60.0 →
  **35.8 µs** (1.68x, spread 6–8%); accuratezza 0.072 → **0.100**. A 2000 fatti,
  singola esecuzione: 201 → 113 µs, 0.003 → 0.006.
- **Riserva sull'accuratezza:** le distribuzioni si sfiorano (intero max 0.080,
  ristretto min 0.072). Il guadagno in mediana è reale, ma non è un miglioramento
  separato al di là di ogni ripetizione.
- **Condizione — importante:** ottimizzazione valida per il **query planner
  corrente**, che risolve solo `(subject, relation) -> object`. **Non** è una
  proprietà generale di ABM. Un planner che risolvesse
  `(object, relation) -> subject` o `(?, relation, object)` scarterebbe risposte
  valide: serve una view per posizione risolta, o `answers_only=False`.
- **Tradeoff:** la lista dei nomi in ruolo oggetto non è derivabile dai fatti
  compressi → va persistita in `meta.json`. Artefatti scritti prima di questa
  voce hanno `_answers = None`, che fa fallback sul codebook intero invece di
  rispondere silenziosamente nulla.
- **Default:** attivo in `query_calibrated`, non in `query()` (che resta
  equivalente al reference).

### A9 · `lru_cache` su `acceptance_z`, `math.exp` in `confidence_packed`

- **Misura:** `query_calibrated` era **più lenta** di `query` (74 vs 58 µs)
  perché ricalcolava 200 bisezioni su `erf` a ogni chiamata. Con la cache:
  soglia 0.12 µs, `confidence_packed` 0.46 → 0.24 µs, query calibrata **38 µs**.
- **Condizione:** `acceptance_z` è pura in `(candidates, alpha)`; `maxsize=4096`
  copre qualsiasi codebook realistico.

---

## B — Scartate (non ritentare senza un motivo nuovo)

| tentativo | misura | perché scartato | riprod. |
|---|---|---|---|
| **Buffer preallocati** per xor/popcount/out | 64.6 µs contro 63.6 | nessun guadagno: l'allocatore di numpy già ricicla i buffer di questa taglia | sì |
| **Popcount sulla vista uint8** invece che uint64 | 111 µs contro 64 | 8x più elementi da ridurre; il popcount per-elemento non compensa | sì |
| **Albero di somme uint16** per la riduzione | 54 µs contro 24.7 di `np.sum` | troppe chiamate numpy su array via via più piccoli | sì |
| **Screening su prefisso di bit** (metà bit, top 2% candidati) | 1.24x, con perdita di recall | separazione insufficiente: z scende da 2.52 a 1.78 usando metà dei bit | sì |
| **Bit derivati da `random_raw` del Mersenne Twister** | bit-esatto su 3000 seed, ma 32.4 µs contro 17.7 | le operazioni di derivazione costano più dei 2048 float64 che evitano | sì, esattezza inclusa |
| **Query in batch** (K = 8, 32, 128) | 53–72 µs/query contro 60 a K=1 | nessun riuso di cache reale: il traffico sulla matrice resta K volte | sì |
| **VP-Tree / indici metrici** per il cleanup | 4487 nodi visitati su 5000 | falsificato strutturalmente, vedi sotto | sì, a due M diversi |

Nessuno di questi scarti dipende da un dettaglio della macchina: i margini sono
larghi (1.7x nel caso più stretto) e le cause sono strutturali, non marginali.

### Perché nessun indice sublineare funziona sul cleanup

Non è un problema di implementazione, è il regime delle distanze. Misurato a
D=2048:

| fatti nel trace | d(target) | d(non-target) | gap |
|---|---|---|---|
| 50 | 0.446·D | 0.500·D | 111 bit |
| 200 | 0.472·D | 0.500·D | 57 bit |
| 500 | 0.483·D | 0.500·D | 34 bit |

Una query di cleanup dista 0.45–0.48·D dal **proprio** codeword. Nessun bound
triangolare pota (con p ≈ 0.47 le distanze sono concentrate) e nessuna LSH
funziona (una banda di r bit coincide con probabilità 0.53^r). Anche il bound
esatto a due stadi è inutile: la distanza parziale su metà dei bit non esclude
praticamente nessuna riga.

**Conseguenza accettata:** il cleanup è uno scan esaustivo lineare con costante
bassa — 0.05 ms a M=1000, 0.44 ms a M=10k, **4.8 ms a M=100k** (D=2048),
limitato dalla banda di memoria. Il claim corretto è "lineare con costante
bassa", non "sublineare". Dettagli in `docs/falsification_report.md`, addendum F4.

---

## C — Limiti noti non risolti (per scelta)

### C1 · Seed dei codeword a 32 bit — limite noto del formato v1

4 byte di MD5 come seed: probabilità di almeno una collisione ~**69% a M=100.000
entità**, e due nomi in collisione condividono il codeword, diventando
indistinguibili. Inoltre `name.lower()` rende `"Rome"` e `"rome"` lo stesso
codeword.

**Non si corregge sotto il freeze.** Allargare il seed cambia *ogni* codeword e
invalida ogni artefatto salvato e ogni misura pubblicata. Appartiene a un
eventuale **formato v2**, non a una patch.

### C2 · Costo della query nel multi-shard è O(shards)

Una query non porta con sé una chiave su cui instradare, quindi vanno interrogati
tutti gli shard. Lo sharding compra **accuratezza** (ogni trace resta sotto il
proprio noise floor), non costo di query. Un routing deterministico richiederebbe
di conservare le triple per il rehash alla crescita, cosa che il formato attuale
non fa.

### C3 · `RandomState` condiviso — debito tecnico esplicito, non un bug

La voce A1 ha introdotto **stato mutabile condiviso**: un `RandomState` per thread
in una `threading.local()`. È l'unica cosa in questa fase che può rompersi in modi
non locali.

Registrato come **debito tecnico accettato**, non come difetto da correggere, in
base a tre condizioni che oggi sono soddisfatte:

1. **Documentato** — qui e nel docstring di `_seeded_rng`.
2. **Testato sul caso principale** — `test_codeword_bits_are_stable_and_thread_safe`
   verifica determinismo con 8 thread concorrenti e confronta i bit contro un
   generatore indipendente.
3. **Con condizioni di riapertura** — sotto.

**Cosa il test non copre, dichiarato:** `multiprocessing` con `fork` (il figlio
erediterebbe lo stato del generatore), e una generazione di codeword annidata
dentro un'altra sezione che usi lo stesso oggetto. Sono **rischi plausibili, non
osservati**. Non vengono affrontati adesso deliberatamente: costruire una
soluzione per un problema ipotetico è la dinamica che questo progetto cerca di
evitare.

**Riaprire solo con:** un bug riproducibile, oppure un caso d'uso reale che usi
`fork` o accesso concorrente non coperto dal test. Un'ipotesi non basta.

### C4 · Il benchmark misura throughput oltre la capacità

A D=2048 con 500 fatti la pressione è **N/N\* = 2.91** e l'accuratezza predetta
è **0.15**: le latenze sono misurate in un regime in cui le risposte sono
comunque sbagliate. È un test di throughput valido, ma non dice nulla
sull'utilità. Il benchmark ora stampa pressione e accuratezza predetta accanto
alle latenze proprio per rendere questo impossibile da ignorare.

---

## D — Stato: fase chiusa

Le prestazioni sono considerate **sufficienti** e questa fase è chiusa. La
domanda utile non è più "possiamo renderlo più veloce" ma "c'è qualcuno per cui
38 µs invece di 57 µs cambia una decisione". Finché la risposta è no, un altro
ciclo di micro-ottimizzazioni ha valore atteso inferiore al capire se il runtime
risolve un problema che qualcuno vuole risolto.

Riaprire questo file solo se: (a) un profilo su un carico **reale** mostra un
collo di bottiglia nuovo, (b) cambia il formato (v2) e le voci C1/C2 tornano in
gioco, o (c) qualcuno documenta un caso d'uso in cui la latenza attuale è il
vincolo.

Il congelamento vale anche per **il processo descritto qui**, non solo per il
codice: le regole in testa al file sono abbastanza buone da evitare errori
sistematici, e affinarle ulteriormente è la stessa trappola dei rendimenti
decrescenti da cui questa fase è uscita. Se questo file viene riaperto, che sia
per una misura, non per la sua metodologia.
