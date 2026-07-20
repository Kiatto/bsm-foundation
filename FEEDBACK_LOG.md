# Feedback Log — fase di osservazione (M2)

Regole (vincolanti, decise PRIMA del primo tester):

1. **Ogni feedback genera al massimo una modifica.** Un problema
   riportato da 1 persona si annota, non si insegue; da 2+ persone
   diventa prioritario.
2. **Il pitch non si cambia** prima di avere più utenti indipendenti:
   3 tester individuano problemi evidenti, non inferiscono un mercato.
3. **Il core non si tocca** (FREEZE, 8 settimane) qualunque cosa dica
   un feedback — si annota per la fine del periodo.
4. Registrare SUBITO, con citazioni testuali, non parafrasi a memoria.

**KPI ufficiale — TTFS (Time To First Success):** tempo tra
`pip install abm-runtime` (o clone) e la prima risposta corretta da
`Memory`. Sintetizza documentazione, esempi, API e UX in un numero.

## Registro

| Tester | Profilo | TTFS | Primo errore/blocco (citazione) | Completato? | "Ti mancherebbe?" | Azione (max 1) |
|---|---|---:|---|---|---|---|
| T1 | | | | | | |
| T2 | | | | | | |
| T3 | | | | | | |

## Annotazioni non azionate (problemi riportati 1 sola volta)

- —

## Idee core emerse durante l'osservazione (VIETATE fino a fine freeze)

- **Sleep-time learning (apprendimento continuo per consolidamento)**
  (20/7/2026, da conversazione con kiatto, non da un tester). Idea:
  un ciclo offline ("sonno") che rivede contesti multipli (conversazioni/
  sessioni passate) ed estrae fatti nuovi mai memorizzati esplicitamente
  — analogo neuroscientifico: consolidamento ippocampo→neocorteccia via
  replay durante il sonno.
  - Cosa esiste GIÀ e non è questo: `Memory.compile_pairs()` (sleep-time
    *compilation*, congelata) consolida coppie di fatti GIÀ NOTI in una
    traccia più efficiente da interrogare (2 cleanup → 1). Non impara
    fatti nuovi, riorganizza quelli esistenti.
  - Cosa esiste GIÀ e copre PARZIALMENTE l'idea: `store(s,r,o,weight=n)`
    + Law VII (N_eff=Σw²) danno già significato quantitativo al
    rinforzo per esposizioni ripetute dello stesso fatto — l'analogo
    più vicino, nella teoria attuale, al consolidamento per ripetizione.
    Utilizzabile SUBITO, non è nuova teoria.
  - Cosa sarebbe GENUINAMENTE NUOVO (e quindi vietato fino a fine
    freeze): un meccanismo interno ad ABM che rivisita periodicamente
    la propria traccia, rileva ridondanze/contraddizioni tra fatti
    accumulati nel tempo, e si ripesa da solo. Implicherebbe un nuovo
    operatore/legge — non un'estrazione (Livello A: resta compito
    dell'LLM decidere cosa vale la pena memorizzare da un contesto).
  - Da riprendere a fine freeze, PRIMA come esperimento con predizione
    scritta, non come feature.
