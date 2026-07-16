# Protocollo pre-registrato — Pilot con estrattore LLM esterno (OpenRouter)

Data di registrazione: 2026-07-15, PRIMA di qualsiasi estrazione.

## Scopo

Primo test del contratto ABM con dati esterni non generati da noi
(HotpotQA validation, contesti Wikipedia reali) e compilatore LLM
esterno reale (modello free su OpenRouter). Il claim testato NON è
"l'accuratezza sarà alta": è "il contratto, emesso prima delle query,
predice l'accuratezza entro l'incertezza dichiarata".

## Disegno

- **Campione:** 40 domande `type=bridge` estratte con seed fisso 77
  dal validation set (7405 righe). Bridge = catene, il caso che il
  runtime modella; la restrizione è dichiarata qui, prima di guardare
  i dati.
- **Estrattore (question-blind):** 1 chiamata LLM per domanda; il
  modello vede SOLO i 10 paragrafi di contesto, mai la domanda.
  Schema di relazioni generico fissato nel prompt.
- **Planner (context-blind):** chiamate LLM batch; il modello vede
  SOLO la domanda e il vocabolario di relazioni, mai i contesti né
  le triple estratte.
- **Memoria:** una WorkingMemory D=2048 per domanda, triple + inverse
  automatiche (protocollo identico al pilot blind precedente).
- **Audit (per il contratto):** sulle prime 15 domande (ordine del
  campionamento, non scelte), si verifica simbolicamente — a meno di
  inverse — se esiste un percorso dalle anchor del piano alla risposta
  gold nel grafo estratto. Pg = frazione con percorso presente E piano
  eseguibile. SE binomiale su n=15.
- **Contratto (pre-query):** Acc = Pg × Pr, con Pr dalla teoria al
  carico medio delle memorie. CI del contratto = 1.96·SE(Pg)·Pr + 2%.
- **Misura:** le 40 domande end-to-end; risposta corretta = match
  token-normalizzato con la gold.

## Criteri di esito (decisi ora)

1. **SUCCESSO del claim:** |misurato − contratto| ≤ CI dichiarato.
2. **FALLIMENTO del claim:** fuori dal CI. Va nel paper come
   falsificazione, qualunque sia la direzione.
3. L'accuratezza assoluta NON è un criterio: un estrattore free con
   Pg=0.3 e contratto rispettato è un successo del claim; un Pg=0.9
   con contratto violato è un fallimento.
4. Nessuna modifica a prompt, soglie o matching dopo aver visto i
   risultati end-to-end. Le iterazioni di prompt sono consentite SOLO
   sulla singola domanda di calibrazione (id #0 del campione), che
   viene poi ESCLUSA dalle 40.

## Vincoli dichiarati

- Modello free (~50 richieste/giorno): 41 chiamate estrazione + ≤4
  planning. Se il rate limit tronca il run, si riporta il campione
  effettivo con il suo SE.
- La chiave API è dell'utente, usata via env, mai committata.

---

## Fase 1 pre-registrata — Multi-compilatore reale (registrata 2026-07-15)

**Varia SOLO il compilatore.** Congelati: corpus (le 33 domande con
estrazione riuscita del compilatore A, identificate da `sample_meta` ∩
`extractions.jsonl`), planner v1 e i suoi piani (`plans.json`, chiave =
id domanda), esecutore, audit (prime 15 in ordine di campionamento),
D=8192, matching. Un compilatore per run, analisi prima del successivo.

Compilatori (famiglie indipendenti): A nvidia/nemotron-3-ultra-550b ·
B tencent/hy3 · C google/gemma-4 (26b o 31b) · D openai/gpt-oss-20b.

Per ogni compilatore si registra la riga completa:
Pg (audit n=15) | N_eff | M | pressure | aliasing | Pr previsto |
accuracy prevista (=Pg×Pr×alias) ± CI | accuracy osservata | errore.

**Domanda:** il contratto segue la misura per OGNI compilatore?
Esiti entrambi utili: (A) sì → evidenza che la legge descrive
l'architettura; (B) un compilatore rompe sistematicamente il contratto
→ limite della teoria, da riportare come tale.

**Linguaggio dell'esito (vincolante):** in caso positivo si scrive
"su questo protocollo e su questo campione la legge descrive il
comportamento di compilatori eterogenei" — NON "la legge è
indipendente dal compilatore".

**Fase 2 (solo dopo):** planner a 2 hop, a compilatore fissato.

### Emendamento pre-dati (registrato prima dei risultati di B)

- Formulazione vincolante dell'eventuale esito positivo: "Across
  heterogeneous extraction pipelines evaluated under a fixed protocol,
  contract predictions remained consistent with observed performance
  within experimental uncertainty."
- Ordine di lettura dei risultati: PRIMA la decomposizione delle
  risorse (Pg, N_eff, M, pressure, aliasing, Pr) e la sua coerenza con
  la legge, POI il contratto vs misurato.
- Caveat dichiarati: (1) la categoria C (schema di piano, 58% dei miss
  di A) è comune a tutti i compilatori → rischio di Pg schiacciati in
  una stessa regione bassa, che non discriminerebbe il modello; in tal
  caso il test del range richiede la Fase 2 ripetuta su tutti i
  compilatori. (2) I compilatori sono modelli eterogenei ma con prompt
  e schema di relazioni identici (nostri): l'eterogeneità è del
  modello, non della pipeline di prompting.

---

## Fase 1A — CHIUSA: saturation check (2026-07-15)

Conclusione sperimentale: con planner v1, due famiglie di compilatori
molto diverse (nemotron-550b, hy3) producono profili di risorse
indistinguibili (Δ<1% su ogni colonna) e le stesse 2 risposte
corrette. Il planner v1 domina Pg; il compilatore è variabile di
secondo ordine entro la sensibilità del protocollo. C e D non vengono
eseguiti nelle stesse condizioni (valore atteso ~nullo).

## Fase 1B pre-registrata — planner sensitivity (registrata PRIMA di scrivere il planner v2)

**Varia SOLO il planner** (v1 → v2). Congelati: estrazioni del
compilatore A (extractions_A.jsonl, già raccolte), corpus 33 domande,
esecutore semantico esteso ai piani a 2 hop, audit (prime 15), D=8192,
matching. Planner v2: piani con catena ≤2 relazioni + vincolo
opzionale; riceve la domanda E l'elenco dei NOMI di relazione
effettivamente estratti (non i contesti: il context-blind resta —
i nomi di relazione non contengono i documenti; dichiarato).

**Predizioni scritte ora:**
- L'attribuzione dà il tetto: eliminare la categoria C (58% dei miss)
  porterebbe Pg verso ~0.6; realisticamente il planner v2 ne
  eliminerà una parte → atteso Pg in [0.15, 0.5] (largo, dichiarato).
- Caso A: Pg↑, Pr~, contratto segue → evidenza forte (CI stretto).
- Caso B: Pg↑ ma con più aliasing/carico effettivo → il contratto
  DEVE incorporarli e seguire comunque.
- Caso C: Pg↑ ma la misura non segue → FALSIFICAZIONE della forma
  per-query su dati esterni; va nel paper come tale.
- Criterio identico: |misurato − contratto| ≤ CI dichiarato.

### Fase 1B — esito e emendamento v2.1 (registrato prima della valutazione v2.1)

ESITO v2 (criterio pre-registrato): Pg audit = 0/15, contratto 0% ± 2%,
misurato 3% (1/33) → formalmente VIOLATO di 1 punto. DUE note oneste:
(1) la formula del CI degenera a Pg=0 (SE binomiale=0) — difetto di
disegno del NOSTRO harness, scoperto dal caso limite; con l'intervallo
esatto Clopper-Pearson (0/15 → upper 21.8%, contratto upper ~20%) la
misura è compatibile. Si riportano ENTRAMBE le letture, senza
sostituire il criterio a posteriori. (2) Diagnosi: i piani v2 sono
semanticamente corretti ma falliscono sul MATCH ESATTO del nome di
relazione (frammentazione del vocabolario dell'estrattore:
traded_for/traded_to, works_for/writes_for). La categoria C si
raffina: non struttura del piano, ma allineamento del vocabolario a
livello di istanza.

EMENDAMENTO v2.1 (una sola modifica, meccanica): esecutore e audit
groundano anche le RELAZIONI con lo stesso matcher token-based già
usato per le entità (per ogni hop, si accettano le relazioni estratte
con overlap di token col nome pianificato; soglia: ≥1 token condiviso
dopo normalizzazione). Predizione: se la frammentazione è la causa
vera, Pg sale in modo netto; criterio invariato, CI con
Clopper-Pearson d'ora in poi (correzione di harness dichiarata).
