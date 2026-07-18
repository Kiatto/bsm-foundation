# Kit tester (M2) — 3-5 persone che NON conoscono ABM

Regole: non amici del progetto, non spiegare nulla oltre il messaggio,
non difendere il prodotto durante il feedback. Il deliverable non è
"gli è piaciuto": è DOVE SI SONO BLOCCATI.

## Il messaggio da mandare (copia-incolla, non aggiungere altro)

> Ciao — sto raccogliendo feedback su un tool di memoria per sistemi
> AI. Non ti spiego niente apposta: è parte del test.
>
> pip install abm-runtime && abm demo
>
> Repo: https://github.com/Kiatto/bsm-foundation
> Demo web (niente da installare): [URL GitHub Pages]/demo/
>
> Mi servono solo tre cose, anche due righe:
> 1. quanto ci hai messo dal link al primo output;
> 2. dove ti sei bloccato (anche "non ho capito a cosa serve" vale);
> 3. se domani te lo togliessi, ti mancherebbe qualcosa? Se sì, cosa?

## A chi mandarlo (profili, in ordine di valore)

1. Chi mantiene una knowledge base o manualistica interna.
2. Chi costruisce agenti LLM e si lamenta della memoria/RAG.
3. Chi lavora in un dominio regolamentato (audit, medicale, ISO).
4. Uno sviluppatore Python generico (baseline di installabilità).

## Registro feedback (uno per tester — compilare SUBITO, non a memoria)

```
Tester #: ____   profilo: ____________   data: ________
Tempo link→primo output: ______ min     (KPI 2: target <10)
Punto di blocco esatto (citazione, non parafrasi):
  "________________________________________________"
Risposta a "ti mancherebbe qualcosa?":  NO / SÌ → cosa:
  "________________________________________________"
Azione derivata (bug/doc/UX — una sola): ___________________
```

## Disciplina (le regole di sempre, applicate ai feedback)

- Un "no, non mi mancherebbe niente" si registra con la stessa cura
  di una legge falsificata. Niente razionalizzazioni ("non era il
  target giusto") senza un criterio scritto PRIMA di mandare il link.
- Criterio pre-registrato: il tester conta se ha un problema reale di
  gestione della conoscenza (profili 1-3) o è un dev Python (4).
- Ogni feedback produce al massimo UNA azione. Se ne servono tre, la
  prima è quasi sempre "la documentazione non ha spiegato X".
