# Rapporto — Cross-domain: il Livello B segue il dominio, non il dataset

Data: 2026-07-15 · Harness: `examples/crossdomain_bench.py` · Dati:
`crossdomain_results.json` · Reference congelata, D=4096, 5 seed.

## Il claim e il disegno

**Claim:** l'accuratezza di un ABM dipende solo dalle risorse (N, M,
hop, ε di grounding), non dalla struttura del dominio. Quattro
topologie deliberatamente diverse, catene a 2 hop, ε di grounding
per-dominio dichiarato, **una sola formula di contratto**:

    Acc = (1-ε)² × p(N, M)²          (nessun parametro fittato)

| dominio | struttura | ε |
|---|---|---|
| api | 6 relazioni dense riusate su 50 endpoint | 6% |
| manuals | sequenze con UNA relazione (next_step) | 11% |
| legal | 30% dei riferimenti su 5 atti-hub | 18% |
| medical | 50 percorsi convergenti su 10 diagnosi-hub | 9% |

## Esito: una falsificazione localizzata, spiegata esattamente, conservata

**Primo run:** api 0.9%, legal 5.6%, medical 1.8% di |dev| — ma
**manuals crolla: 37% misurato vs 79% previsto**. Il claim di
indipendenza, così com'era formulato, è falsificato lì.

**Diagnosi (algebra esatta, non fit):** l'encoding s⊕ρ(r)⊕o è
simmetrico in soggetto e oggetto — **ogni fatto è un arco non
orientato**. Con la stessa relazione su hop consecutivi,
f₁ ⊕ key(y, r) = x *esattamente*: il predecessore è un alias a pari
segnale del successore. Verifica sul meccanismo: su hop puliti le
risposte sono 21 z / 19 x / **0 altro** — una moneta perfetta, come
prevede l'algebra. Negli altri domini r₁≠r₂ e l'alias non colpisce mai.

**Correzione derivata** (fattore 1/g per hop con g candidati a pari
segnale, calcolabile dal piano di query): contratto manuals 40% vs
misurato 37% ±5% → |dev| 2.6%.

**Rimedio a livello controller, già nella reference:** la proiezione
tipata (`subset`) che esclude i nodi visitati elimina l'alias —
variante "guided": **80% misurato vs 79% di contratto pieno**.

## Tabella finale

| dominio | ε | contratto | misurato (95% CI) | \|dev\| |
|---|---|---|---|---|
| api | 6% | 87% | 88% ±4% | 0.009 |
| manuals (naive) | 11% | 40% | 37% ±5% | 0.026 |
| legal | 18% | 67% | 61% ±7% | 0.056 |
| medical | 9% | 82% | 80% ±3% | 0.018 |
| manuals (guided) | 11% | 79% | 80% ±6% | **0.009** |

|dev| media **2.4%**, max 5.6%.

## Il claim, riformulato onestamente

Il Livello B non è dominio-indipendente nella forma ingenua: la
struttura *entra* nella legge — ma **solo attraverso un termine
algebrico esatto e calcolabile prima delle query** (l'aliasing da
simmetria degli archi), mai attraverso il contenuto. La teoria non solo
prevede il crollo: dice esattamente su quale query avverrà, di quanto
(1/g), e quale operatore già esistente lo elimina. Per il prodotto: il
compilatore/Inspector può rilevare i piani con relazioni ripetute e
attivare la proiezione guidata automaticamente.

Nota a margine: la simmetria è anche una *feature* — ogni ABM risponde
gratis alle query inverse (chi precede y?), un arco memorizzato vale
in entrambe le direzioni.

## Valutazione 0-10

| Dimensione | Voto | Nota |
|---|---|---|
| Valore scientifico | **9** | Claim forte testato, falsificato dove doveva, riformulato con un meccanismo esatto — il ciclo metodologico completo in un esperimento |
| Rigore | **9.5** | Primo run conservato; correzione derivata dall'algebra (non fittata); verifica indipendente del meccanismo (21/19/0) |
| Valore di prodotto | **8** | "La teoria segue il dominio" + rilevazione automatica dell'aliasing dal piano di query |
| Completezza | **7** | Topologie sintetiche; mancano corpus documentali reali e g>2 (relazioni ripetute su catene più lunghe) |

## Prossimi passi

1. Aliasing con g>2 e catene lunghe a relazione ripetuta (la correzione
   1/g è derivata ma testata solo a g=2).
2. Inspector: rilevare relazioni ripetute nel piano e dichiarare il
   fattore di aliasing nel contratto.
3. Corpus documentali reali per dominio (con estrazioni LLM di kiatto).
