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

**KPI ufficiale #2 — TTFV (Time To First Value):** non "quanto ci
mette a installare", ma quanto tempo passa prima che l'utente dica
"ah, questo mi serve" — può essere 30s, 5 min, o mai. Si osserva, non
si misura con un timer: è il momento in cui, guardando lo schermo
condiviso, il tester smette di "provare il tool" e inizia a "usarlo".

**Metrica candidata (da 👍/👎 sulle risposte in knowledge.html):**
risposte utili / risposte totali. A n=5 tester si conta a occhio dai
video, nessuna telemetria centralizzata finché non serve a scala.

## Disciplina aggiuntiva: origine delle modifiche

Ogni modifica al comportamento del sistema (soglie, euristiche,
messaggi) va etichettata:
- **[UTENTE]** — nata da un tester reale che si è bloccato/confuso.
- **[INTERNA]** — trovata da me/kiatto testando, senza un utente.

Se si accumulano 10 modifiche INTERNE e zero da utenti, è il segnale
di ottimizzazione autoreferenziale — fermarsi e aspettare dati veri.

### Modifiche registrate

- **[INTERNA]** (23/7/2026) soglia di confidence in knowledge.html
  alzata da 0.52 a 0.65 + guardia anti-self-loop. Trovata testando io
  stesso un piano di query sbagliato generato dal planner, non da un
  tester. Motivazione tecnica valida (evitava una risposta
  silenziosamente errata), ma l'origine resta interna — da verificare
  se un utente reale l'avrebbe mai notata.

### Ipotesi da osservare nel test crudele (5 persone, schermo condiviso, zero aiuto)

- Termini come "pressure", "Memory Contract", "confidence" potrebbero
  confondere un utente al primo contatto. NON li ho riscritti in
  anticipo — sarebbe una modifica interna basata su una mia ipotesi.
  Si osserva se un tester chiede "cosa vuol dire pressure?" — solo
  allora si corregge, ed entra come [UTENTE].
- "Che PDF devo caricare?" — se emerge, è un segnale di onboarding
  mancante (esempio pre-caricato?), non di UI.

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
