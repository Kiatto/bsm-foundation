# Video 3 minuti — script (da registrare: screen capture + voce)

Regola: zero formule, zero teoria, zero storia del progetto.
Ogni scena = una cosa sullo schermo.

## 0:00–0:20 — Il problema (schermo: terminale vuoto)
"Ogni sistema di memoria per AI ti dice quanto è andato bene DOPO.
ABM ti dice come andrà PRIMA — e firma un contratto sul numero."

## 0:20–0:40 — Install (schermo: terminale)
    pip install abm-runtime
    abm demo
Lasciare scorrere l'output fino al MEMORY CONTRACT. Non commentare
ogni riga: indicare solo "Expected accuracy" e "Pressure".

## 0:40–1:20 — Documento → memoria (schermo: demo web, step 1-3)
Incollare 10 righe di manualistica. Click "Extract facts" →
"Store in binary memory". Inquadrare la traccia disegnata:
"Dodici fatti. Duecentocinquantasei byte. Deterministico:
stessi fatti, stessi bit, stesse risposte."

## 1:20–2:00 — Domande multi-hop (demo web, step 4)
Domanda a 2 hop ("chi chiamo se si rompe payment_service?").
Mostrare l'explain: "ogni risposta arriva con il percorso,
le distanze e una confidence calibrata: 0.5 significa 'sto
tirando a caso' — e te lo dice."

## 2:00–2:40 — Il momento del contratto (demo web, step 5)
Click "Verify the contract".
"Il pannello a destra era stato CALCOLATO prima di qualunque query.
Promesso 99.8, misurato 100. Se questi due numeri divergono,
non è sfortuna: è un bug, e lo trattiamo come tale."

## 2:40–3:00 — Chiusura (schermo: landing)
"ABM non sostituisce il tuo LLM: il tuo LLM compila, ABM ricorda.
pip install abm-runtime. Il contratto è il prodotto."

## Note di produzione
- Terminale con font grande (16pt+), tema scuro.
- Nessun voiceover sopra gli output: pause vere.
- Durata target 2:50 — se supera 3:00, tagliare la scena 4, non la 5.
