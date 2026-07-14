# Rapporto — Il paradigma adottato: percorso algebrico nel ReasoningEngine

Data: 2026-07-13 · Test: **118/118 passed** · Benchmark: `examples/algebraic_demo.py`

## Cosa è cambiato

Il paradigma VSA (vedi [vsa_report.md](vsa_report.md)) è passato da
*dimostrato in un esperimento* ad *adottato dal motore di produzione*:
il `ReasoningEngine` accetta ora un `AlgebraicReasoner`
([bsm/memory/algebraic.py](../bsm/memory/algebraic.py)) come **primo
percorso di risoluzione**, con le euristiche testuali declassate a fallback.

Architettura (il simbolico ai confini, l'algebra nel mezzo):

```
testo KB ──TripleExtractor──▶ triple (s, rel, o) + inverse
                                  │  grounding: sketch MinHash
                                  ▼
                           WorkingMemory XOR olografica (D=2048)
query ──QueryPlanner──▶ catena di relazioni
                                  │  per hop: T ⊕ chiave → cleanup
                                  │  accettato se confidence calibrata ≥ soglia
                                  ▼
                       risposta + frase sorgente (provenienza)
                       — altrimenti fallback alle euristiche —
```

Nel loop di ragionamento non ci sono più stop-words, boost di keyword né
indici testuali: ogni hop è uno XOR e un cleanup, e la decisione
algebrico-vs-fallback la prende la confidence calibrata sulla distribuzione
nulla di Hamming (i tre livelli costruiti in questa sessione si incastrano:
sketch MinHash → algebra XOR → calibrazione statistica).

## I tre numeri (stesso benchmark multihop di sempre, 54 frasi, 10 domande)

| | Baseline euristico | Integrato (algebra + fallback) |
|---|---|---|
| **Accuracy** | 9/10 | **10/10** |
| **Quota risolta per via algebrica** | 0/10 | **9/10** |
| **Latenza mediana** | 2.6 ms | **1.4 ms** |

Dettagli che contano:

- La domanda che le euristiche sbagliavano ("Who founded the company based
  in Seattle?", 12-21 ms di beam search per una risposta errata) è risolta
  dall'algebra in **1.0 ms**: `seattle → amazon.com inc → jeff bezos`.
- L'unica domanda rimasta alle euristiche ("…makes the iPhone?") è un caso
  in cui il grounding non supera la soglia: **il fallback fa il suo lavoro**
  e la risposta resta corretta. Il sistema degrada, non fallisce.
- Ogni risposta algebrica porta la **provenienza** (la frase sorgente del
  fatto finale), quindi resta verificabile.
- Estrazione: 52 triple da 54 frasi (96%), con le inverse (`made_by`,
  `hq_of`, `founder_of`) che rendono il grafo percorribile nei due sensi.

## Bug scovati durante l'integrazione (e corretti)

1. Lo split delle frasi spezzava su "Inc." (abbreviazione ≠ confine di
   frase): metà del KB non veniva estratto.
2. Gli sketch di grounding tenevano la punteggiatura nei token
   ("seattle," ≠ "seattle").
3. Soglia di accettazione hop tarata sopra i valori reali dei hop corretti
   (0.70 vs ~0.72 osservato con 52 fatti): portata a 0.60 (≈3σ sopra il
   rumore), il fallback copre il resto.

## Valutazione 0-10

| Dimensione | Voto | Motivazione |
|---|---|---|
| **Cambio di paradigma effettivo** | **8** | Era 6.5: ora il 90% delle query del benchmark passa dall'algebra nel motore di produzione, non in un esperimento. Il loop di ragionamento non contiene più euristiche testuali |
| **Risultato empirico** | **8** | Strettamente meglio del baseline su tutte e tre le metriche (accuracy 10/10 vs 9/10, latenza −46%); ma sempre sul KB interno |
| **Robustezza del design** | **8** | Fallback a cascata governato da una soglia statistica, non da if testuali; provenienza preservata; 13 test nuovi (118 totali) |
| **Innovazione incrementale** | **6** | L'integrazione in sé è ingegneria; il valore innovativo sta nel sistema complessivo (sketch → XOR → calibrazione) che ora lavora come un pezzo unico |
| **Debolezza: copertura dei pattern** | **6/10 gravità** | TripleExtractor e QueryPlanner sono regex su inglese SVO: fuori da quei pattern l'algebra è cieca (by design: il fallback copre). È il punto dove serve un estrattore vero (o un piccolo LLM al confine) |
| **Debolezza: scala** | **4/10 gravità** | 52 triple in D=2048 è comodo (capacità ~130); KB reali richiedono tracce multiple o D maggiore — già previsto nel piano |
| **Validazione esterna** | **3** | Invariata: il giudice vero resta un benchmark non scritto da noi |

**Complessivo: 8/10** per questo passo — il progetto ora ha la storia
completa e funzionante: *128-256 byte di traccia olografica, ragionare è
uno XOR, astrarre è una proiezione, e il motore di produzione la usa davvero.*

## Stato del progetto a fine sessione

| Fase | Stato |
|---|---|
| Substrato geometrico (Hamming, packed, calibrato) | maturo, 2 bug core corretti |
| Layer ibrido (MinHash, GraphCache geometrico, prototipi, RRF adattivo) | integrato, testato |
| Algebra VSA (Working/Semantic, binding eterogeneo) | dimostrata sperimentalmente |
| **Paradigma nel motore di produzione** | **adottato: 9/10 query via XOR** |
| Validazione esterna | **mancante — prossimo passo obbligato** |

## Prossimi passi

1. **Benchmark esterno** (HotpotQA/2Wiki ridotto): ora che il sistema vero
   esiste, è l'unico numero che manca alla storia.
2. Consolidamento Working→Semantic in `sleep()` (il cambio di operatore
   come meccanismo di apprendimento — la tesi completa).
3. Estrattore di triple più generale (dependency parsing leggero o LLM al
   confine) per alzare la copertura dei pattern.
4. Scaling: tracce olografiche multiple con routing per KB oltre ~100 fatti.
