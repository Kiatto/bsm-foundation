# Rapporto — Il ragionamento come algebra binaria (Phase IV, prototipo)

Data: 2026-07-13 · Test: **105/105 passed** · Esperimento: `examples/vsa_experiment.py`

## La tesi

> **L'operatore di binding è una proprietà del sottosistema di memoria,
> non dell'algebra.** La Working memory binda con XOR (invertibile,
> manipolazione esatta); la Semantic memory binda con proiezioni lossy
> fittate (l'astrazione è codificata nell'operatore di scrittura, non è
> un processo a valle). Il consolidamento è un cambio di operatore.

Rispetto alla VSA classica (Kanerva, MBAT): il binding a proiezione esiste
in letteratura, ma sempre pensato quasi-invertibile; qui la *lossiness è la
feature* (filtro d'informazione per ruolo) e l'algebra è deliberatamente
**eterogenea tra sottosistemi** — formulazione che non risulta occupata.

## Cosa è stato costruito

[bsm/memory/vsa.py](../bsm/memory/vsa.py): `bind_xor` (in {-1,+1} il prodotto
È lo XOR, auto-inverso), `permute` (ρ di Kanerva), `bundle` (majority vote,
riuso del meccanismo dei prototipi), `ItemMemory` (cleanup), `RoleProjection`
(random o fittata via SVD a rango ridotto), `WorkingMemory` (traccia
olografica: TUTTI i fatti in UN vettore, query = `T ⊕ chiave → cleanup`),
`SemanticMemory` (addressing con operatore configurabile — il banco di prova).

## Risultati sperimentali (D=1024)

### Task 1 — Capacità olografica
Una sola traccia da 1024 bit che contiene N fatti; query via XOR puro:

| N fatti | 5 | 10 | 20 | 40 | 80 | 160 |
|---|---|---|---|---|---|---|
| accuracy | 100% | 100% | 100% | 90% | 60% | 19% |

Degrado dolce, come da teoria (capacità ~D/(2·ln D)). **128 byte che
"contengono" 20-40 fatti interrogabili algebricamente.**

### Task 2 — Multi-hop come XOR
25 catene a 2 hop (`prodotto → azienda → città`): hop 1 **22/25**, catena
completa **20/25** — senza estrazione di entità, senza stop-words, senza
boost di keyword, senza indici. La query è letteralmente due XOR e due cleanup.

### Task 3 — La tesi alla prova: addressing con alias mai visti
42 query con nomi varianti ("google", "apple computer", "tesla motors"…) su
fatti memorizzati coi nomi canonici. Tre condizioni, stesso identico task:

| Condizione | Accuracy | Margine medio |
|---|---|---|
| XOR (invertibile) | 90% | 159 bit |
| Proiezione random (lossy, non informata) | 81% | 109 bit |
| **Proiezione fittata r≥12 (lossy, informata)** | **90%** | **270 bit (+70%)** |

Il rank sweep mostra la "manopola dell'astrazione": rank 4 → 57% (astrae
troppo, collassa entità distinte), rank 12-32 → plateau a 90% col margine
massimo. **La proiezione lossy informata non perde accuratezza e rende
l'addressing il 70% più robusto** — la separazione prevista dalla tesi:
lossy-informata > esatta > lossy-cieca.

### Scoperta collaterale (di valore autonomo)
Il MinHash a 1 bit su CRC32 produceva distanze assurde (763 dove la teoria
dice 512): **CRC32 è lineare su GF(2)** e le parità dei minimi risultano
correlate tra insiemi. Corretto con md5 in
[entity_encoder.py](../bsm/memory/encoder/entity_encoder.py) — fix che
migliora anche l'EntityEncoder di produzione (Task 3 è passato da 71% a 90%
sulla condizione XOR col solo cambio di hash). È il secondo bug di substrato
scovato dall'algebra (il primo: `unpack` non-inverso di `pack`).

## Valutazione 0-10

| Dimensione | Voto | Motivazione |
|---|---|---|
| **Innovazione** | **8.5** | La tesi "binding eterogeneo per sottosistema + lossiness come operatore di astrazione" non risulta formulata così in letteratura VSA; i componenti sì (onestà: MBAT, permutation binding, thinning esistono) |
| **Evidenza sperimentale** | **7** | Esperimento a 3 condizioni con predizione verificata (+70% margine a parità di accuracy, ordine lossy-informata > esatta > lossy-cieca); ma 42 query su 12 entità costruite da noi — è una dimostrazione di fattibilità, non una validazione |
| **Cambio di paradigma effettivo** | **6.5** | Il multi-hop *è* diventato algebra (Task 2), ma vive in un prototipo parallelo: il ReasoningEngine di produzione usa ancora le euristiche testuali. Il paradigma è dimostrato, non ancora adottato |
| **Solidità ingegneristica** | **8** | 16 test nuovi (105 totali), tutto deterministico, zero dipendenze nuove; 2 bug di substrato scovati e corretti |
| **Coerenza con l'identità BSM** | **9** | Tutto è {-1,+1}, XOR, majority vote, Hamming: nessun tensore float nel percorso di ragionamento; il PrototypeIndex diventa la cleanup memory canonica |
| **Rischi residui** | **5/10 gravità** | Rumore per hop limita le catene lunghe (20/25 già a 2 hop olografici); capacità olografica modesta a D=1024; la SemanticMemory decodifica per nearest-fact O(n); tutto da ripetere su dati esterni |
| **Potenziale di pubblicazione/differenziazione** | **7.5** | "Una memoria da 128 byte interrogabile via XOR + consolidamento come cambio di operatore di binding" è una storia che nessun RAG può raccontare; serve il benchmark esterno per difenderla |

**Complessivo: 7.5/10** come prototipo di cambio di paradigma — la tesi ha
superato il suo primo esperimento discriminante, e il progetto ora contiene
tre livelli coerenti: substrato geometrico (Hamming), memoria ibrida
(retrieval calibrato), algebra cognitiva (VSA eterogenea).

## Prossimi passi (in ordine di leva)

1. **Integrare la WorkingMemory nel ReasoningEngine** come primo percorso
   di risoluzione multi-hop (fallback alle euristiche): il paradigma passa
   da dimostrato ad adottato.
2. **Consolidamento vero Working→Semantic in `sleep()`**: i fatti confermati
   dal feedback vengono ri-bindati con le proiezioni fittate — il cambio di
   operatore diventa il meccanismo di apprendimento.
3. **Benchmark esterno** (HotpotQA ridotto): l'unico giudice che conta.
4. **Scaling della capacità**: D=8192, sparse block-codes, o traccia
   multipla con routing — per superare il muro dei ~40 fatti/vettore.
5. **Decodifica resonator** per la SemanticMemory (iterativa invece che
   nearest-fact O(n)).
