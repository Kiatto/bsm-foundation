# Rapporto — Prima validazione esterna: HotpotQA (distractor)

Data: 2026-07-13 · Harness: `examples/hotpotqa_eval.py` · Dati: HotpotQA
validation (distractor), 200 domande *bridge*, 10 paragrafi Wikipedia
per domanda (2 gold + 8 distrattori), ~42 frasi/domanda.

Metrica: **containment** (la risposta gold è contenuta nel testo
restituito) — il sistema restituisce frasi, non span, quindi EM/F1
classici non si applicano; l'oracle è al 100% (la metrica è sempre
raggiungibile) e riportiamo il **caso** come riferimento minimo.

## I numeri

| Metrica | Caso (random) | Risultato | vs caso |
|---|---|---|---|
| Oracle (risposta nel contesto) | — | 100% | — |
| Baseline single-hop top-1 | 4.9% | **13%** | 2.6× |
| Recall@5 / Recall@10 | — / 34.7% | 30% / 48% | — / 1.4× |
| **Motore integrato top-1** | 4.9% | **7%** | 1.4× |
| Latenza mediana | — | 11.5 ms | — |

Stadi della pipeline:

| Stadio | Valore |
|---|---|
| Triple estratte per domanda (da ~42 frasi) | **1.9** (~4% delle frasi) |
| Query riconosciute dal planner | **1/200 (0.5%)** |
| Risolte per via algebrica | **0/200 (0%)** |

## Le tre scoperte (in ordine di importanza)

### 1. Il paradigma algebrico non è stato falsificato — non è mai stato ingaggiato
Su testo Wikipedia reale, il confine simbolico (TripleExtractor e
QueryPlanner a regex, scritti sui pattern SVO del KB interno) ha copertura
~0: 1.9 triple da 42 frasi, 1 query pianificata su 200. **L'algebra XOR non
ha mai avuto la palla.** Distinzione scientifica cruciale: il risultato
dice che *il confine* non copre il dominio, non che *il meccanismo* non
regge. La domanda "quali classi di ragionamento sono rappresentabili come
algebra su stati discreti?" resta aperta — questo run dice solo che i
regex non sono il modo di alimentarla.

### 2. Le euristiche multi-hop SOTTRAGGONO valore su dati reali
Il motore integrato fa **7%** dove il banale top-1 single-hop fa **13%**:
la decomposizione a 2 hop (estrazione entità per maiuscole, bridge
entity, boost di keyword) — che sul KB interno vale il 90% — su dati
reali *distrugge* metà del segnale che il retrieval aveva già trovato.
L'overfitting al demo, sospettato dalla prima review di questa sessione,
è ora quantificato: −6 punti assoluti rispetto a non fare nulla.

### 3. Il substrato geometrico è la parte che regge meglio (ma non basta)
13% top-1 = 2.6× il caso, con 32 byte/frase, zero training, encoder
fittato al volo su 42 frasi. Reale ma modesto; e il recall@10 (48%) è
solo 1.4× il caso — su questo va detto che BM25 sullo stesso task fa
tipicamente molto di più. Il substrato è efficiente, non ancora
competitivo in accuratezza.

## Valutazione 0-10 di questo passo

| Dimensione | Voto | Motivazione |
|---|---|---|
| **Valore scientifico del risultato** | **8** | Primo numero esterno del progetto; separa con precisione dove il sistema si rompe (confine simbolico: copertura 0.5-4%) da dove regge (substrato: 2.6× caso); quantifica l'overfitting del demo (−6 punti) |
| **Onestà metodologica** | **9** | Baseline casuale calcolato dai dati, oracle dichiarato, limiti della metrica dichiarati (containment ≠ EM), distinzione falsificato/non-testato esplicita |
| **Esito per il paradigma** | **4** | Non falsificato ma nemmeno corroborato: 0% di ingaggio. Il claim "10/10 sul benchmark interno" oggi non è trasferibile |
| **Esito per il sistema attuale** | **3** | 7% sotto il proprio baseline single-hop: su dati reali il reasoning euristico va disattivato o rifatto |
| **Direzione resa evidente** | **9** | Il collo di bottiglia è uno e misurato: la copertura del confine simbolico. Non "serve più codice", serve UN componente (estrazione triple/query reale) e la stessa harness rifarà il verdetto |

## Cosa direbbe un ricercatore esterno

Il comportamento "non me lo aspettavo" per ora non c'è — c'è il suo
prerequisito: una harness riproducibile su dati pubblici con baseline
oneste. Il candidato più vicino a quel risultato resta il claim del
benchmark interno (multi-hop in 1 ms via XOR da una traccia di 256 byte,
10/10) — ma finché non si ripete su domande non nostre, è un demo.

## Prossimi passi (regola: niente feature, solo misura)

1. **Chiudere il gap di ingaggio con il minimo indispensabile**: un solo
   componente di estrazione triple/query generale (anche un piccolo
   modello open al confine — il paradigma non vieta ML *ai confini*, lo
   vieta *nel meccanismo*). Poi ri-eseguire questa stessa harness: è
   l'esperimento che decide se il paradigma regge fuori casa.
2. **Spegnere le euristiche multi-hop su dati reali** (o gate più severo):
   il fallback giusto oggi è il single-hop, che fa +6 punti.
3. **RuleTaker/ProofWriter prima di MuSiQue**: sono benchmark di
   ragionamento *sintetico con struttura regolare* — il terreno dove il
   confine simbolico copre quasi il 100% e si testa davvero l'algebra
   (la domanda di ricerca), non l'estrattore.
4. Stessa harness su 2WikiMultiHopQA (ha triple gold annotate: permette
   di testare l'algebra con estrazione *oracolare* — l'ablazione chiave).

Punto 3 e 4 sono i più promettenti per il "primo risultato inatteso":
l'algebra con ingresso pulito, misurata dove il confine non è la variabile.
