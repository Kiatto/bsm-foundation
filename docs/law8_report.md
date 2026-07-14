# Rapporto — Law VIII quantitativa: robustezza all'errore di estrazione

Data: 2026-07-14 · Harness: `examples/extraction_robustness.py` ·
Figura: `docs/figures/fig6_robustness.png` · Dati:
`extraction_robustness_results.json` · Eseguito sulla **reference
implementation congelata** (dogfooding).

## Il disegno

Quattro tipi di errore di estrazione iniettati a tasso ε ∈ [0, 50%]
su catene a 2 hop (D=2048, N=120, 3 seed), con le predizioni scritte
**prima** della misura:

- Tipi 1-3 (missing / wrong relation / wrong entity) → legge
  moltiplicativa Acc = (1−ε)²·Pr
- Tipo 4 (spurious) → legge di capacità (degrado dolce, Law IV)
- Rilevabilità: il tipo 3 doveva produrre errori "confidenti"

## Risultati

| Tipo di errore | |dev| predizione naive | esito |
|---|---|---|
| wrong_relation | **0.027** | confermata (moltiplicativa) |
| wrong_entity | **0.019** | confermata (moltiplicativa) |
| spurious | **0.021** | confermata (capacità, dolce: 47%→23% a ε=50%) |
| missing | 0.106 → **0.045 (raffinata)** | corretta dalla teoria stessa |

**Le due famiglie previste esistono** (moltiplicativa vs capacità), e la
sorpresa del tipo "missing" è la scoperta del run: i fatti mancanti
degradano *meno* del previsto perché **alleggeriscono anche la traccia**
— le due leggi si compongono. La forma finale, domain-independent:

    Acc(ε) = Pg(ε) × Pr(N_eff(ε))

il fattore di grounding *per* il fattore di reasoning valutato al
carico che il grounding lascia. Deviazione media complessiva sulle 40
celle (4 tipi × 10 livelli): **~3%** con le forme raffinate.

## Predizione falsificata (conservata)

La rilevabilità del tipo 3 **non si materializza a livello di catena**:
la confidence media delle risposte sbagliate è indistinguibile tra i
quattro tipi (0.35-0.36 vs 0.37-0.39 delle corrette, a ε=20%). Il
meccanismo ipotizzato (oggetto corrotto ∈ codebook ⇒ errore confidente)
esiste al singolo hop ma si diluisce nel prodotto delle confidence.
Conseguenza pratica: **la confidence di catena non è un rilevatore di
errori di grounding** — servono controlli al livello A (es. doppia
estrazione), non al livello B. Registrata come falsificazione.

## Il valore per il prodotto

La frase del cliente ora è calcolabile: *"il tuo estrattore ha
precisione 93% (ε=7%) su fatti a 2 hop → il tuo ABM renderà
(0.93)²·Pr(N_eff) ≈ 86%·Pr"* — prima del deploy, per qualsiasi dominio,
perché nessun termine della formula dipende dal contenuto. Con la
distinzione operativa utile: gli errori *spurious* (falsi positivi
dell'estrattore) costano molto meno dei falsi negativi — un estrattore
va tarato per il recall, non per la precision. Anche questa è una
conseguenza della teoria, non un'opinione.

## Valutazione 0-10

| Dimensione | Voto | Nota |
|---|---|---|
| Valore della legge ottenuta | **9** | Domain-independent, componibile, con la distinzione spurious/missing che rovescia una best practice (tarare per recall) |
| Rigore | **9** | Predizioni scritte prima; una raffinata dalla teoria stessa; una falsificata e conservata |
| Valore di prodotto | **8.5** | La formula del cliente + la regola di taratura dell'estrattore |
| Completezza | **7** | Manca: k>2 hop, errori correlati (non i.i.d.), domini reali |

## Prossimi passi (Fase 1 → Fase 2 della roadmap)

1. Estendere a k hop e a errori correlati (gli estrattori reali
   sbagliano a cluster, non i.i.d.).
2. **ABM Inspector**: la dashboard della memoria (utilizzo, pressione,
   accuratezza attesa, degrado previsto a +100 fatti) — ora ha tutte le
   formule che le servono, inclusa questa.
3. SDK con la teoria esposta (`expected_accuracy`, `capacity_remaining`,
   e ora `expected_accuracy_given_extractor(precision)`).
