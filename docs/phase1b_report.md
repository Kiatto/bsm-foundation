# Rapporto — Fase 1B: planner sensitivity (v2, v2.1)

Data: 2026-07-16 · Harness: `examples/pilot_openrouter/planner2.py` ·
Estrazioni congelate (compiler A) · Protocollo ed emendamenti
pre-registrati in `PROTOCOL.md`.

## Esiti secondo i criteri pre-registrati

| | v1 (baseline) | v2 (piani 2-hop) | v2.1 (+grounding relazioni) |
|---|---|---|---|
| Pg audit (n=15) | 0.067 | **0.00** | **0.00** |
| contratto | 6.0% | 0% ±2% (formula difettosa) | 0%, banda CP [0%, 22%] |
| misurato | 6.1% | 3% (1/33) | 3% (1/33) |
| verdetto | rispettato | **VIOLATO** (di 1 punto, al bordo) | rispettato (banda CP) |

Tre fatti da riportare senza abbellimenti:

1. **Il planner v2 ha PEGGIORATO Pg** (0.067 → 0), contro la
   predizione pre-registrata (atteso [0.15, 0.5]). Predizione
   falsificata.
2. **La v2 è formalmente VIOLATA dal criterio pre-registrato** — ma il
   caso limite ha scoperto un difetto del NOSTRO harness: il CI
   binomiale gaussiano degenera a Pg=0. Con Clopper-Pearson esatto la
   misura è compatibile. Riportiamo entrambe le letture; d'ora in poi
   CP (correzione di harness, dichiarata, non un cambio di criterio a
   risultato noto — la lettura "violato" resta agli atti).
3. **Anche la predizione v2.1 è falsificata**: il grounding fuzzy
   delle relazioni non ha mosso nulla. La frammentazione del
   vocabolario NON era la causa dominante.

## Diagnosi finale (tre casi rappresentativi, meccanismi diversi)

- Redskins: la domanda richiede profondità ≥3 con vincolo ("anno di
  nascita del fratello del pick del primo giro") — fuori dallo schema
  di piano, e l'informazione potrebbe non essere nelle triple.
- Markkanen: **l'informazione c'è e il percorso esiste**
  (Markkanen →traded_to→ Bulls →traded_to→ Butler) ma il planner
  context-blind ha scelto la catena sbagliata (born_in al hop 2).
- McElroy: l'entità àncora non è mai stata estratta — buco di
  estrazione, nessun planner può rimediare.

## La lezione, formulata onestamente

Il collo di bottiglia non è un componente: è l'**accoppiamento
planner–estrattore**. Con estrazione question-blind e pianificazione
context-blind fatte da modelli deboli, i due lati non coordinano mai
la rappresentazione: ogni fix locale (2 hop, grounding relazioni) ha
spostato il fallimento invece di rimuoverlo. Ogni predizione
"aggiustando X, Pg salirà" fatta finora su questa pipeline è stata
falsificata — il che è informativo: il problema è di interfaccia, non
di componente.

Implicazione per il progetto (da verificare, non affermata): il Memory
Contract dovrebbe includere uno **schema contract** — l'interfaccia
dichiarata tra compilatore e planner (vocabolario di relazioni CHIUSO
e imposto a entrambi, non suggerito). È coerente con la scoperta della
Fase 1A: il profilo di risorse lo detta il prompt/schema, non il
modello.

## Cosa resta vero del claim centrale

Il contratto ha continuato a seguire la misura in tutti i regimi
testati (6%→6%, 0%→3% entro la banda esatta) — ma sempre in regimi a
bassissimo Pg, dove la potenza discriminante è minima. Il test forte
(range di Pg ampio) resta non eseguito: su questa pipeline non è
raggiungibile senza uno schema contract o un planner/estrattore più
capaci. Questo è il limite onesto dello stato attuale.

## Valutazione 0-10 (conservativa)

| Dimensione | Voto | Nota |
|---|---|---|
| Evidenza sul claim | **4.5** | Ancora nessun punto ad alto Pg; compatibilità sì, discriminazione no |
| Rigore | **9** | Due predizioni falsificate e registrate; violazione formale agli atti; bug di harness dichiarato |
| Valore diagnostico | **8** | Il problema è stato localizzato all'interfaccia planner-estrattore, con esempi meccanici |
| Progresso di prodotto | **5** | Nasce l'idea dello schema contract, ma la pipeline resta inutilizzabile |
