# PRODUCT_HYPOTHESES.md — ipotesi di prodotto, non roadmap

Questo file **non contiene funzionalità pianificate.** Contiene ipotesi su cosa
gli utenti vorranno, con lo stato dell'evidenza. Nessuna voce qui è un impegno a
costruire; nessuna voce qui va costruita finché non scatta il suo trigger.

Serve a evitare due errori simmetrici:

- **dimenticare un'intuizione** perché non era il momento;
- **costruirla subito** perché sembrava buona.

È l'equivalente, sul prodotto, di quello che `docs/falsification_report.md` è per
la teoria e `PERFORMANCE.md` per l'ingegneria.

## Formato

Ogni ipotesi ha cinque campi. Il quinto è quello che rende il file utile:

| campo | cosa significa |
|---|---|
| **Ipotesi** | l'affermazione, formulata in modo che possa essere falsa |
| **Stato** | non verificata / in verifica / confermata / **smentita** |
| **Evidenza** | osservazioni concrete, con data e fonte. "Sembra sensato" non è evidenza |
| **Si costruisce quando** | il trigger, quantificato prima di vedere i dati |
| **Si abbandona quando** | la condizione che la uccide |

L'ultimo campo è obbligatorio. Un'ipotesi che non si può abbandonare non è
un'ipotesi: è una decisione già presa che aspetta una scusa. Vale la stessa regola
del `PERFORMANCE.md`: i trigger vanno fissati **prima** di raccogliere i dati, non
adattati dopo per far scattare quello che si voleva costruire comunque.

## Cosa conta come evidenza

Una richiesta indipendente = una persona che, **senza esserne sollecitata**, ha
descritto il problema che la feature risolverebbe. Non conta:

- la stessa persona che ne parla due volte;
- una richiesta indotta da una domanda tipo "ti servirebbe una dashboard?";
- un'intuizione di chi sviluppa (la fonte di tutte le voci qui sotto);
- un concorrente che ce l'ha.

Registrare la fonte, la data e le **parole testuali**, come impone
`FEEDBACK_LOG.md`. Le due discipline sono la stessa.

---

## Stato generale: nessuna evidenza, su nessuna ipotesi

Al 2026-08-07 `FEEDBACK_LOG.md` non registra **nessuna sessione con un tester
esterno**. Tutte le ipotesi qui sotto hanno evidenza zero e provengono dalla stessa
fonte (discussione interna del 2026-08-06). Sono correlate, non indipendenti: se
l'intuizione di partenza è sbagliata, sono probabilmente sbagliate tutte insieme.

Questo è il vincolo che conta oggi. Il limite del progetto non è tecnico né
scientifico: **manca conoscenza sul comportamento di utenti reali**, e non si
produce scrivendo altro codice.

---

# ASSUNZIONE FONDANTE

> ## H0
>
> **Esiste almeno un tipo di utente per cui conoscere in anticipo l'affidabilità
> della memoria cambia una decisione operativa.**
>
> Se H0 viene falsificata, tutte le ipotesi successive perdono significato.

H0 non è una funzionalità in attesa di trigger: è la ragione per cui il progetto
esiste. Sta sopra le altre perché non è in coda con loro — le altre presuppongono
che sia vera.

- **Stato:** non verificata.
- **Evidenza:** nessuna.
- **Si verifica quando:** un tester esterno, senza che gli venga spiegata la
  teoria, guarda il Memory Contract e **fa qualcosa di diverso** da quello che
  avrebbe fatto senza — rimanda un deploy, aggiunge documenti, alza D, rifiuta di
  usare la memoria per un certo tipo di domanda. Non basta che gli interessi, né
  che lo trovi elegante: deve **cambiare una decisione**.
- **Si abbandona quando:** tre tester indipendenti guardano il contratto e non ne
  fanno nulla. In quel caso il problema non è la presentazione: è la premessa.

**Precondizione per poterla testare.** H0 non è testabile finché il contratto non
è credibile. Il criterio di accettazione è stato riscritto il 2026-08-06 — prima
rifiutava 30/30 risposte corrette su una memoria la cui accuratezza predetta era
1.00 — e ha alle spalle **un solo carico sintetico**. Serve almeno una prova su
conoscenza reale prima di mettere H0 davanti a qualcuno: presentarla ora
rischierebbe di falsificare la presentazione invece della premessa.

---

## Ipotesi derivate

Tutte presuppongono H0. Nessuna ha senso se H0 cade.

## H1 · Dashboard di leggibilità del contratto

> **Ipotesi:** un responsabile qualità o un manager vuole vedere lo stato della
> conoscenza in forma sintetica (barre, semafori) invece che come numeri.

- **Stato:** non verificata.
- **Evidenza:** nessuna.
- **Si costruisce quando:** 2 richieste indipendenti, **oppure** 1 sola sessione in
  cui un non-tecnico legge l'output dell'Inspector e non riesce a dire cosa
  dovrebbe fare. La soglia è bassa perché il costo è basso ed è verificabile in
  un'ora.
- **Si abbandona quando:** i tester leggono il contratto testuale senza difficoltà,
  o chiedono *più* dettaglio numerico invece che meno.
- **Nota:** la traduzione giusta la determina chi legge. Farla in astratto produce
  la dashboard che immaginiamo noi, non quella che serve.

## H2 · Policy Engine (soglie e regole di escalation dichiarative)

> **Ipotesi:** le organizzazioni vogliono esprimere `SE confidenza < X ALLORA non
> rispondere / escala` come configurazione, invece di leggere un float.

- **Stato:** non verificata.
- **Evidenza:** nessuna.
- **Si costruisce quando:** **3 clienti indipendenti** chiedono soglie diverse,
  regole di escalation o routing differenziato — cioè quando la varietà delle
  richieste, non la loro esistenza, rende insufficiente un parametro `alpha`.
- **Si abbandona quando:** i primi utenti usano `alpha` così com'è e non ne
  cambiano mai il valore. Se nessuno tocca il parametro che c'è già, un linguaggio
  per esprimerlo è teatro.
- **Già disponibile oggi:** `AgentMemory(alpha=...)` e `query_calibrated(alpha=)`
  coprono il caso singolo. Il DSL serve solo per la *varietà*, non per la funzione.

## H3 · Report di audit esportabile (PDF firmabile, versionato)

> **Ipotesi:** esiste un auditor o una funzione compliance che accetterebbe un
> report generato da ABM come evidenza.

- **Stato:** non verificata.
- **Evidenza:** nessuna. **È l'ipotesi con il rischio più alto**, perché richiede
  che un terzo esterno all'utente riconosca il documento come valido — una cosa che
  non dipende da noi.
- **Si costruisce quando:** un utente reale mostra il documento che oggi produce a
  mano per lo stesso scopo, e possiamo confrontarlo con ciò che ABM sa già dire.
- **Si abbandona quando:** un auditor guarda il contratto e dice che una stima
  statistica di accuratezza non è evidenza ammissibile nel suo processo. Va chiesto
  **prima** di costruire, non dopo: costa una conversazione.
- **Precondizione dura:** H0 confermata. Un PDF firmabile costruito su un criterio
  di accettazione con un giorno di vita non aumenta la fiducia, sposta il rischio
  dal codice alla firma.

## H4 · Knowledge lifecycle (diff fra versioni della conoscenza)

> **Ipotesi:** gli utenti vogliono vedere come grounding, pressione e accuratezza
> cambiano fra un import e il successivo.

- **Stato:** non verificata.
- **Evidenza:** nessuna.
- **Si costruisce quando:** 2 utenti indipendenti fanno un **secondo** import sullo
  stesso corpus. Prima di allora il diff non ha nulla da confrontare — e nessuno ha
  ancora fatto un primo import reale.
- **Si abbandona quando:** l'uso reale è un import unico e statico.

## H5 · Ownership / team

> **Ipotesi:** serve sapere chi è responsabile di quale porzione di conoscenza.

- **Stato:** non verificata.
- **Evidenza:** nessuna.
- **Si costruisce quando:** un utente con **più di una persona** che scrive nella
  stessa memoria segnala un conflitto o una domanda di responsabilità.
- **Si abbandona quando:** l'uso reale è mono-utente. Oggi non esiste nemmeno il
  concetto di utente.

---

## Ipotesi già risolte

Registrate per non riproporle.

| # | Ipotesi | Esito | Data |
|---|---|---|---|
| R1 | Serve una matrice di compatibilità pubblica | **Costruita** — non era un'ipotesi di prodotto ma una verifica: ha trovato `requires-python` falso e la penalità 12x su NumPy 1.x. Vedi [`COMPATIBILITY.md`](COMPATIBILITY.md) | 2026-08-06 |
| R2 | Il cleanup può diventare sublineare con un indice | **Smentita**, strutturalmente. Vedi `docs/falsification_report.md` F4 | 2026-08-06 |
| R3 | Serve orchestrazione cloud / cluster / marketplace di connettori | **Fuori discussione** finché H0 non è confermata: architettura senza utenti | 2026-08-06 |

---

## Come si aggiorna questo file

Dopo **ogni** sessione con un tester, non a fine mese. Aggiornare il campo
*Evidenza* anche quando l'osservazione è negativa — soprattutto quando è negativa:
un'ipotesi che nessuno menziona per tre sessioni consecutive è un dato, e va
scritto lì.

Se un'ipotesi passa a **smentita**, non cancellarla: spostarla in *Ipotesi già
risolte* con la data e il motivo. Il valore di questo file sta nelle voci morte
quanto in quelle vive.
