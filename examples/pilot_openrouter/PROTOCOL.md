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
