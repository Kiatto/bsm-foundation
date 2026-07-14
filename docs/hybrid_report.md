# Rapporto — Layer Ibrido Geometrico (Phase III.5)

Data: 2026-07-13 · Test: **89/89 passed** · Benchmark multihop: **90% (9/10)** invariato

## Contesto

La Phase III aveva introdotto reasoning multi-hop efficace ma quasi interamente
simbolico: il substrato binario era ridotto a indice. Questo intervento riporta
la geometria di Hamming dentro i meccanismi (confidence, entity matching, cache,
consolidamento, fusione), senza sacrificare l'accuratezza sul benchmark.

Divisione dei ruoli dichiarata: **il simbolico decide cosa chiedere, il binario
decide cosa è simile.**

---

## 1. Confidence calibrata sulla distribuzione nulla di Hamming

`calibrated_confidence(dist, D)` — z-score contro Binomial(D, 0.5) ≈ N(D/2, √D/2),
mappato con logistica a temperatura 8 (scelta perché la pendenza locale a livello
di chance coincide con la scala legacy `1-d/D`: stesso potere discriminante,
valore assoluto interpretabile: **0.5 = indistinguibile dal rumore**).

Evidenza: sostituita in tutti i punti di scoring del ReasoningEngine; accuratezza
multihop invariata (90%); range confidence su 30 query: 0.55–0.75.

| Dimensione | Voto | Nota |
|---|---|---|
| Innovazione | **7** | Sfrutta una proprietà che solo lo spazio binario ha (distribuzione nulla nota a priori, indipendente dal modello) |
| Impatto attuale | **6** | Confidence finalmente interpretabile e confrontabile tra store di dimensioni diverse; ranking preservato per costruzione |
| Solidità | **8** | 4 test dedicati; taratura τ derivata analiticamente, non a occhio |
| Debolezza residua | **4/10 gravità** | La distribuzione reale delle distanze in un KB non è esattamente la nulla (i documenti non sono indipendenti); una calibrazione empirica per-store sarebbe il passo successivo |

## 2. EntityEncoder via one-bit MinHash (Jaccard → Hamming)

Riscritto: lo sketch MinHash a D bit ha la proprietà
`E[hamming(A,B)] = D·(1-J)/2`. La distanza di Hamming tra sketch **è** una stima
del Jaccard: eliminata la `_EntityMemory` dedicata (ora usa il `MemoryStore`
core), eliminata la metrica travestita, l'encoder rispetta il contratto comune
(int8 {-1,+1}) ed è un cittadino legittimo dell'ensemble.

Evidenza: test empirici della relazione Jaccard↔Hamming (insiemi disgiunti →
~D/2; J=0.5 → ~D/4); usabile in un BSM standard; determinismo verificato.

| Dimensione | Voto | Nota |
|---|---|---|
| Innovazione | **8** | Il fix concettualmente giusto: risultato classico (MinHash) applicato esattamente dove serviva; elimina il debito architetturale più grave |
| Impatto attuale | **7** | Ensemble ora omogeneo (3 spazi, stessa metrica); via la memoria speciale e il caso particolare in `_recall_encoder` |
| Solidità | **8** | 5 test, incluse le proprietà statistiche dello sketch |
| Debolezza residua | **3/10 gravità** | CRC32 per bit×entità è O(D·|entità|) per encode (accettabile a questa scala); il matching resta word-level: typo dentro la singola parola non recuperati (vedi §3) |

## 3. GraphCache con lookup e merge geometrici

`GraphCache(encoder=EntityEncoder(...))` (opt-in, default invariato): al miss di
lookup esatto, ricerca nella palla di Hamming (raggio 0.3·D) tra le entità note;
merge di `sleep()` basato su Jaccard stimato dagli sketch (`J ≈ 1-2d/D`) invece
che sulle parole.

Evidenza: `lookup_entity("Google")` e `"google llc"` → HIT su edge memorizzato
come "Google LLC"; typo `"Googl LLC"` → MISS (J=1/3 sotto soglia); merge
geometrico verificato; comportamento default invariato (4 test).

| Dimensione | Voto | Nota |
|---|---|---|
| Innovazione | **6** | Cache addressing geometrico invece che per stringa: alias e varianti gratis, e language-agnostic (non dipende più solo dal lowercase inglese) |
| Impatto attuale | **5** | Aliasing funziona; il valore pieno emerge su KB grandi con molte varianti di entità |
| Solidità | **7** | Testato, opt-in, zero regressioni sul comportamento esistente |
| Debolezza residua | **5/10 gravità** | Lookup geometrico O(n. entità) lineare; typo *dentro* la parola non coperti (servirebbero n-gram di caratteri nello sketch — estensione naturale); raggio 0.3 fissato a mano |

## 4. PrototypeIndex — consolidamento majority-vote e recall gerarchico

Nuovo modulo: leader clustering in Hamming + centroide = **voto di maggioranza
bit a bit** (quasi gratis in binario). Recall in due stadi: prototipi → membri
dei top-`n_probe` cluster, con distanze esatte.

Evidenza: su KB da 50 doc → 31 prototipi (compressione 1.61×, 21 singleton);
top-1 gerarchico = flat su 24/30 query con `n_probe=3`; distanze esatte
preservate. Ha fatto emergere (e correggere) un **bug preesistente del core**:
`MemoryStore.unpack` non era l'inverso di `pack` (bit order MSB vs LSB).

| Dimensione | Voto | Nota |
|---|---|---|
| Innovazione | **9** | L'idea più "BSM-nativa" del progetto: astrazione che emerge dalla pura geometria, senza euristiche testuali. È la potenziale tesi del progetto |
| Impatto attuale | **4** | Su 50 doc eterogenei la compressione è modesta e il recall gerarchico è approssimato (24/30); il payoff vero è su store grandi e ridondanti — non ancora dimostrato |
| Solidità | **7** | 4 test + bugfix core scovato; ma non integrato nel ciclo `sleep()` del CognitiveEngine (per scelta: prima va validato) |
| Debolezza residua | **6/10 gravità** | Leader clustering dipende dall'ordine di inserimento; `n_probe` basso sacrifica recall; serve un benchmark su 10k+ entry per giustificarlo |

## 5. Pesi RRF adattivi dal feedback

`EnsembleRetriever.reward(query, answer, correct)`: gli encoder che avevano
rankato in alto la risposta premiata guadagnano peso (credito 1/rank, floor 0.1,
media rinormalizzata a 1). Collegato a `ReasoningEngine.feedback()`: il feedback
ora scende fino al layer geometrico.

Evidenza: dopo 10 feedback positivi i pesi divergono
(projection 1.13, hash 0.99, entity 0.88 — coerente col fatto che il projection
è l'encoder più preciso su questo KB); recall funzionante dopo l'adattamento;
i pesi non collassano mai (50 feedback negativi → tutti > 0).

| Dimensione | Voto | Nota |
|---|---|---|
| Innovazione | **6** | Non nuovo in letteratura (bandit-style fusion), ma chiude un'asimmetria vera: prima l'esperienza aggiornava solo il grafo simbolico |
| Impatto attuale | **5** | Si adatta in modo sensato e misurabile; con 3 soli encoder il margine è limitato |
| Solidità | **7** | 4 test; learning rate e floor scelti conservativamente |
| Debolezza residua | **4/10 gravità** | Il credito 1/rank è euristico; nessun decay dei pesi nel tempo; andrebbe valutato se l'adattamento migliora l'accuracy end-to-end su un benchmark esterno |

---

## Valutazione complessiva del progetto (post-intervento)

| Aspetto | Voto | Commento |
|---|---|---|
| Coerenza architetturale | **8** | L'ibrido è ora una divisione di ruoli dichiarata, non un compromesso; il travestimento Jaccard è eliminato |
| Fedeltà all'idea originale (binario) | **7.5** | Era ~5: confidence, entity matching, cache addressing, consolidamento e fusione ora passano dalla geometria |
| Robustezza | **8** | 89 test, bug core (unpack) e 8 bug del layer cognitivo corretti in sessione |
| Innovazione | **7.5** | MinHash-nell'ensemble e prototipi majority-vote sono contributi non banali; il resto è ingegneria solida |
| Validazione esterna | **3** | Il tallone d'Achille invariato: tutti i numeri vengono dal KB scritto insieme al codice. Serve HotpotQA/2Wiki |
| Multilinguismo | **4** | Il layer geometrico ora è language-agnostic, ma estrazione entità e boost di intent restano inglesi |
| Pronto per il commit | **9** | Suite verde, demo invariati, API retrocompatibili (breaking: rimossi i knob no-op del ReasoningEngine) |

**Media pesata: ~7/10** — era ~5.5 prima della sessione.

## Prossimi passi consigliati (in ordine)

1. Benchmark esterno (100+ domande HotpotQA) — sblocca ogni altra valutazione.
2. Sketch a n-gram di caratteri nell'EntityEncoder → typo-robustness vera nel GraphCache.
3. PrototypeIndex su store 10k+ entry: misurare speedup e recall@k reali.
4. Calibrazione empirica per-store della confidence (distribuzione delle distanze osservata, non teorica).
5. Integrare `PrototypeIndex.build()` nel ciclo `sleep()` una volta validato il punto 3.
