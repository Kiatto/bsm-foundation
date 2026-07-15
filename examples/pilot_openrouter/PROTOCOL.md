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
