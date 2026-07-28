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
5. **Non modificare il kernel/UX per un singolo feedback [UTENTE].**
   Serve ricorrenza — almeno 3 utenti indipendenti sullo stesso punto.
   Eccezione: bug evidenti o problemi di sicurezza si correggono
   subito, a prescindere dalla ricorrenza.
6. **Ogni modifica elimina un problema OSSERVATO, non una sensazione.**
   ❌ "questa pagina potrebbe essere più chiara" (impressione, non si
   agisce). ✔ "quattro utenti hanno cercato il pulsante Upload per più
   di 20 secondi" (osservazione concreta, comportamento specifico —
   si agisce). Se non riesci a citare un comportamento o una frase
   precisa, non è ancora evidenza.
7. **Non innamorarti dei feedback positivi.** "Molto bello" = quasi
   zero informazione. Cerca: "pensavo facesse quest'altra cosa", "qui
   mi sono bloccato", "non avevo capito che potevo fare una domanda" —
   meno piacevoli da leggere, molto più utili.

**Modello mentale (correzione 23/7/2026):** non "aspetto 5 tester
insieme". La prossima persona che può realisticamente provarlo — un
collega, un ex collega, uno sviluppatore amico, un contatto LinkedIn,
qualcuno incontrato a una conferenza. Si logga a partire dalla prima,
incrementalmente; il primo test serve anche a verificare che il
protocollo di osservazione stesso funzioni. Trovare quella persona
richiede una rete nel mondo reale — è il pezzo che non posso fare al
posto di kiatto; il resto (progettare il test, registrare le sessioni,
interpretare i risultati, applicare la disciplina sopra) lo faccio con
lui, non "glielo consegno e basta".

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
- **(osservato in T2, 1/1 finora)** L'utente pone una domanda
  "topica"/titolo ("Assunzione a tempo indeterminato?") invece di una
  domanda su una relazione specifica fra entità ("chi è il datore di
  lavoro?", "qual è la retribuzione lorda?"). Il sistema non comunica
  da nessuna parte che tipo di domande sa gestire (relazioni fra
  entità nominate, non domande sull'intero documento o sì/no). Se
  ricorre con un secondo utente indipendente → azione candidata:
  esempi di domande nel placeholder o sotto la casella di input
  (specifico per il documento caricato, se possibile). Non ancora
  implementato — 1 solo caso osservato.

## Registro

| Tester | Profilo | TTFS | TTFV | Primo errore/blocco (citazione) | Completato? | "Ti mancherebbe?" | Origine | Tipo | Azione (max 1) |
|---|---|---:|---|---|---|---|---|---|---|
| T1 | kiatto stesso, PDF reale (listini Shopify, italiano) | ~15s import (45 facts, 46 entities) | TBD | 3/3 domande fallite: "sei d'accordo con quanto stabilito nel documento allegato?" → no answer (couldn't ground "documento allegato"); "come vengono gestiti su shopify i listini multipli?" → low confidence; "Come si costruisce un listino?" → no answer (couldn't ground "listino") | No — 0/3 risposte utili | TBD | UTENTE | TBD — in diagnosi, serve vedere i fatti estratti prima di classificare | Nessuna ancora — regola 6: serve evidenza, non ipotesi |
| T2 | kiatto stesso, PDF reale (contratto di assunzione, italiano, dati personali/sensibili) | TBD | TBD | Domanda testuale: "Assunzione a tempo indeterminato?" (= il titolo del documento + "?", non una domanda su una relazione specifica) → "no answer couldn't ground..." | No | TBD | UTENTE | **Comprensione** — diagnosi confermata dalla citazione stessa, non serve altro dato: la domanda non è una query fattuale su un'entità (tipo "chi è il datore di lavoro?"), è il titolo del documento. Il sistema si è comportato correttamente: nessuna entità reale da agganciare → ha detto "non lo so" invece di inventare una risposta. Il gap è che l'utente non sapeva che tipo di domande porre | **Nessuna azione (regola 5: 1 sessione, non 3; e non è un bug — il sistema ha fallito onestamente, non silenziosamente)**. Ipotesi aggiunta sotto: serve guidare l'utente sul tipo di domande accettate |

(Righe aggiunte una alla volta, non in blocco da 5 — per-tester,
appena disponibile. **Tipo**: Comprensione / Feature / Bug / Ipotesi —
serve a scoprire dopo un mese se es. il 70% dei feedback riguarda il
linguaggio e solo il 10% riguarda davvero ABM, prima di investire mesi
nel posto sbagliato.)

## Decisioni prese (separata dal registro feedback)

Non "cosa ha detto l'utente" — cosa **abbiamo deciso** e perché,
incluse le non-modifiche. Fra tre mesi non ricorderai perché hai
scelto di non cambiare qualcosa; questa tabella lo ricorda al posto
tuo, evitando di rilitigare la stessa discussione.

| Data | Evidenza | Decisione | Motivazione |
|---|---|---|---|
| 2026-07-23 | Nessun tester ancora osservato; ipotesi interna che "pressure"/"Memory Contract"/"confidence" possano confondere | Nessuna modifica alla terminologia | Ipotesi non validata da un utente reale — vedi "Ipotesi da osservare" sotto; si aspetta l'osservazione |
| 2026-07-28 | Modello locale: 11/14 triple attribuite all'azienda invece che al dipendente (contratto T2); dopo il fix 11/11 corrette e 5/5 domande risposte | Prompt di estrazione corretto (risoluzione soggetti impliciti) — APPLICATO | Bug di correttezza, non preferenza: rientra nell'eccezione della regola 5. Origine INTERNA, dichiarata. Costo accettato: perdita delle triple corrette sull'azienda (14→11) |
| 2026-07-23 | Trovato testando internamente: piano del planner sbagliato produceva un self-loop mostrato come risposta valida al 57.9% di confidence | Soglia alzata 0.52→0.65 + guardia anti-self-loop (implementato) | Bug di correttezza (risposta silenziosamente sbagliata), non estetica — rientra nell'eccezione della regola 5, corretto subito nonostante origine INTERNA |

## CAUSA RADICE di T1/T2 — trovata (28/7/2026, modello locale)

Con il modello locale (Qwen3-4B-Instruct-2507-GGUF UD-Q4_K_XL su
Unsloth Studio) ho rieseguito l'estrazione sul contratto di assunzione
di T2. L'estrazione **tecnicamente funziona**: 14 triple, JSON valido,
snake_case, italiano gestito correttamente. Ma contiene un errore
semantico sistematico e riproducibile:

    ['kore77_srl', 'has_monthly_salary', '1781.68']      ← SOGGETTO SBAGLIATO
    ['kore77_srl', 'has_qualification', 'software_integration_developer']
    ['kore77_srl', 'has_trial_period', '15_days']
    ['luigi_piccirillo', 'receives_letter', 'assunzione…']  ← unico fatto sul dipendente

**11 triple su 14 attribuite all'AZIENDA invece che al DIPENDENTE.**
Retribuzione, qualifica, sede, orario, periodo di prova sono attributi
di Luigi Piccirillo, non di KORE77.

**Meccanismo:** l'italiano formale dei contratti usa il possessivo con
soggetto implicito ("**La Sua** retribuzione", "**Lei** sarà assunto").
Senza risoluzione delle coreferenze, il modello aggancia i fatti
all'entità nominata più prominente — l'azienda.

**Questo spiega esattamente il fallimento di T1 e T2:** chiedendo
"qual è la retribuzione di Luigi Piccirillo?" la memoria non ha nulla
sotto quell'ancora, perché i fatti sono sotto `kore77_srl`. Non è
l'algebra (Livello B), non è il planner: è **grounding, Livello A**,
con una firma precisa e misurabile — coreferenze/soggetti impliciti in
testo formale italiano.

### CORREZIONE APPLICATA E VERIFICATA (28/7/2026)

Prompt di estrazione: aggiunta istruzione esplicita "RESOLVE IMPLICIT
SUBJECTS BEFORE EMITTING A TRIPLE" con esempi in inglese e italiano
("La Sua retribuzione" → soggetto = persona, non azienda) e il divieto
di agganciare i fatti all'entità più prominente per default.

Misura, stesso modello e stesso testo, prima/dopo:

| | soggetti corretti | triple |
|---|---|---|
| prima | 3/14 (11 attribuite all'azienda) | 14 |
| dopo | **11/11** | 11 |

End-to-end su ABM (D=4096): le 5 domande realistiche sul contratto
(retribuzione, qualifica, periodo di prova, orario, sede) rispondono
**tutte correttamente**, confidence 87-90%, accuratezza misurata 100%
= contratto previsto 100%.

**Effetto collaterale onesto:** le triple sull'azienda che erano
corrette (`kore77_srl has_legal_address`, `has_vat_number`) sono
scomparse — il modello ora sovra-corregge attribuendo tutto alla
persona. 14 → 11 triple. Netto positivo per il caso d'uso (domande sul
contratto del dipendente), ma è una perdita reale di copertura, da
registrare e non nascondere.

Origine: **[INTERNA]** (diagnosi mia, non segnalazione di un tester),
applicata come eccezione "bug evidente" alla regola 5: attribuire i
fatti al soggetto sbagliato è scorrettezza, non preferenza di UX.

## Test end-to-end col modello locale (28/7/2026) — isolamento dei tre livelli

Rieseguito T2 **attraverso gli endpoint reali del prodotto** (non uno
script), con modello locale. Risultato per livello:

| livello | esito | evidenza |
|---|---|---|
| A — estrazione | **14/14 soggetti corretti** | tutte le triple su `luigi_piccirillo`, incluse data d'inizio, settore, tipo di contratto |
| B — ABM | **100%**, contratto 100% = misurato 100% | 5/5 domande corrette con piano giusto, confidence 82-87% |
| C — planner | **0/3 piani corretti** | vedi sotto |

**Il planner è ora l'UNICO collo di bottiglia.** Con piano corretto il
prodotto risponde a tutto; con i piani che il planner genera davvero,
sbaglia. Esempio misurato: piano a 2 hop spuri → risposta
`trav_d_ambrosio_22` invece di `1781.68` (per fortuna a 56% di
confidence, sotto la soglia 0.65 → il prodotto mostra "low confidence"
anziché una risposta sbagliata: la guardia funziona).

### Tre ipotesi mie, tutte FALSIFICATE (registrate per non riprovarle)

1. **"Il planner non conosce le entità"** → passargliele **peggiora**:
   0/3 contro 1/3. Inizia a confondere entità e relazioni.
2. **"È non-determinismo/sampling"** → falso: 3 ripetizioni identiche
   sia a temperatura default sia a 0. Il planner è deterministico.
3. **"Colpa della frase 'Use 2 hops when…' nel prompt"** → rimuoverla
   non aiuta: 0/3 in entrambe le varianti di prompt.

Conclusione: **il problema non è risolvibile con prompt engineering** a
questa taglia di modello. Firma costante del fallimento: il modello usa
NOMI DI RELAZIONE come ancora (`anchor='job_title'`,
`anchor='trial_period'`, `anchor='work_location'`) e sceglie relazioni
quasi-giuste (`base_salary` invece di `gross_monthly_salary`).

### Il test che cambia la diagnosi

Stesso modello, stesse domande, stesse relazioni — ma **compito
riformulato** da "genera un piano JSON" a "scegli quale campo risponde
alla domanda (rispondi col numero)":

    generazione piano JSON:      0/3   (0%)
    classificazione relazione:   6/7   (86%, chance 7%)

Il modello **sa** mappare una domanda italiana sulla relazione inglese
giusta. Non sa produrre il piano JSON. **Il difetto è nella
formulazione del compito, non nella capacità del modello.**

Implicazione (candidata, NON implementata): l'ancora non serve chiederla
a un LLM — in un documento con un soggetto dominante (qui 14/14 triple)
è deterministica; la relazione si ottiene per classificazione (86%);
il multi-hop si affronta solo dove serve davvero.

## Blocchi infrastrutturali (non feedback di prodotto)

- **(23/7/2026)** Durante T1+T2 il tetto giornaliero free-tier di
  OpenRouter è stato esaurito (non contesa tra modelli — verificato:
  il messaggio reale è "Rate limit exceeded: free-models-per-day. Add
  10 credits to unlock 1000 free model requests per day", un limite
  per-account, non per-modello). Correzione di rotta: avevo ipotizzato
  contesa di quota fra extract.js/plan.js sugli stessi modelli e stavo
  per "risolverla" diversificando i modelli — ipotesi sbagliata,
  scartata prima di implementarla dopo aver testato altri modelli e
  trovato lo stesso errore ovunque. Decisione in sospeso (costo reale,
  spetta a kiatto): aggiungere ~10 crediti OpenRouter (sblocca
  1000 richieste/giorno) oppure aspettare il reset giornaliero.

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
