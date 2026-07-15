# Rapporto — Primo test con dati esterni e compilatore LLM reale

Data: 2026-07-15 · Harness: `examples/pilot_openrouter/` · Protocollo
pre-registrato: `PROTOCOL.md` (criteri congelati prima di qualsiasi
estrazione) · Dati: HotpotQA validation (contesti Wikipedia reali,
non generati da noi) · Estrattore: `nemotron-3-ultra-550b:free` via
OpenRouter, question-blind · Planner: `hy3:free`, context-blind.

## Esito secondo il criterio pre-registrato

| | |
|---|---|
| campione effettivo | 33/40 domande bridge (7 perse per rate-limit upstream, dichiarato) |
| audit (n=15, simbolico a meno di inverse) | Pg = 0.07 ± 0.13 |
| **contratto pre-query** | **6% ± 14%** |
| **misurato (33 domande end-to-end)** | **6% (2/33)** |
| **errore del contratto** | **0.4% → RISPETTATO** |

## Lettura onesta, nelle due direzioni

**Cosa NON dice questo risultato.** Con Pg=0.07 il CI è ±14%: il test
aveva poca potenza di falsificazione — quasi qualunque esito basso
sarebbe passato. Il campione è piccolo (33), l'audit più piccolo (15),
e 2 hit su 33 non distinguono finemente tra 3% e 12%. Non è "il
contratto funziona sui dati reali", è "il contratto ha superato il suo
primo test esterno, debole ma reale". E l'accuratezza assoluta (6%) è
pessima: la pipeline free, così com'è, non serve a nessuno.

**Cosa dice.** Formulazione corretta (da review): **il claim non è
stato falsificato da questo test** — non "è sopravvissuto". Con CI
±14% anche 0%, 4%, 8% o 11% sarebbero risultati compatibili: il test
non discrimina. Detto ciò:

1. La predizione puntuale è caduta a 0.4% dal misurato, emessa prima
   delle query da un audit simbolico su 15 esempi.
2. **Attribuzione automatica dei fallimenti** (checklist a criteri
   dichiarati, `attribute_failures.py`, categorie mutuamente esclusive
   su TUTTI i 31 miss — sostituisce l'analisi preliminare su 8 casi,
   che usava un criterio substring troppo lasco):

   | categoria | n | % |
   |---|---|---|
   | C — planner/schema (percorso ≤2 hop esiste, il piano non lo esprime) | 18 | 58% |
   | B — percorso assente nell'estrazione | 8 | 26% |
   | A — gold assente dalle triple | 3 | 10% |
   | **D — algebra** (piano eseguibile simbolicamente, esecuzione errata) | **2** | **6%** |

   Il 6% di fallimenti algebrici è compatibile con 1−Pr = 3.7%
   previsto dalla teoria: la separazione dei livelli regge anche qui.
3. Coerenza col negativo storico su HotpotQA — ma con attribuzione
   misurata, non dedotta.

## Limiti dichiarati

- Modelli free (i più deboli del listino); 50 req/giorno hanno
  troncato il campione a 33 e imposto un planner economico.
- Lo schema di piano (1 hop + vincolo) è una scelta nostra ed è oggi
  il limite dominante — NON un limite dell'algebra: le triple per
  piani a 2 hop in gran parte esistono già nelle estrazioni.
- Il matching risposta-gold è token-based (può sia regalare che
  togliere hit marginali).

## Prossimo passo con il miglior rapporto informazione/costo

Piani a 2 hop + allineamento del vocabolario relazioni. Predizione
nella forma corretta (da review): **se** il nuovo planner cattura i
2-hop mancanti **senza** degradare precision, aumentare aliasing o
carico, **allora** Pg dovrebbe salire (la checklist dà il tetto:
eliminare la categoria C porta Pg verso ~0.6); il contratto dovrà
seguire la misura anche lì, dove il CI si stringe e falsificare
diventa facile.

**Esperimento registrato (il discriminante, da review):
multi-compilatore REALE.** Stesso corpus (le 33 domande), estratto da
3-4 famiglie di modelli indipendenti disponibili via OpenRouter
(nvidia/nemotron-ultra, tencent/hy3, google/gemma-4, openai/gpt-oss) →
un audit, un contratto e una misura per compilatore → se l'errore
previsto segue quello osservato per OGNI compilatore, la Resource
Composition Law modella una proprietà dell'architettura, non del
prompt. Vincolo: ~50 req/giorno → un compilatore al giorno, o crediti.

## Valutazione 0-10 (conservativa)

| Dimensione | Voto | Nota |
|---|---|---|
| Valore del risultato | **6.5** | Primo test esterno superato, ma a bassa potenza statistica; da solo non prova il claim |
| Rigore | **9** | Protocollo pre-registrato e rispettato; campione troncato dichiarato; nessun ritocco post-hoc |
| Diagnostica | **8** | Attribuzione al livello C con evidenza diretta (gold presente nelle triple in 6/8 fallimenti) |
| Utilità di prodotto immediata | **3** | 6% end-to-end: la pipeline free non è utilizzabile; il valore è nel metodo, non nel sistema |
