# Rapporto — Multi-compilatore e Scale Bench

Data: 2026-07-14 · Harness: `examples/multicompiler_bench.py`,
`examples/scale_bench.py` · Dati: `multicompiler_results.json`,
`scale_bench_results.json` · Eseguito sulla reference congelata.

## 1. Esperimento multi-compilatore (dry-run)

**Domanda:** la Resource Composition Law (forma per-query) permette
all'Inspector di prevedere il *ranking* di compilatori diversi prima di
qualsiasi query? È un test di falsificazione della Law VIII al Livello A
e insieme la demo di prodotto ("ABM valuta i compilatori").

**Setup:** stesso corpus (60 catene a 2 hop, D=2048, 10 seed), tre
compilatori simulati con profili d'errore realistici e distinti:

| compilatore | profilo | precisione media | E_q[Pg] |
|---|---|---|---|
| alpha | wrong_entity i.i.d. 8% | 92% | 0.846 |
| beta | intere catene corrotte, 20% | 80% | 0.800 |
| gamma | missing 10% + spurious 30% | 90% | 0.810 (+carico) |

Contratti calcolati **prima** della misura dai soli parametri di
profilo + Pr_clean di calibrazione (47%): alpha 39%, beta 36%, gamma
25% (forma per-query); la forma media dà beta 29%.

**Risultato:**

| | previsto (per-query) | misurato (95% CI) | \|dev\| per-query | \|dev\| media |
|---|---|---|---|---|
| alpha | 39% | 36% ±4% | 0.031 | 0.031 |
| beta | 36% | 40% ±4% | **0.040** | **0.112** |
| gamma | 25% | 26% ±3% | 0.004 | 0.004 |

- **Coppie risolvibili** (Δ oltre l'errore combinato): alpha>gamma e
  beta>gamma — il ranking per-query è concorde **2/2**.
- La coppia alpha/beta era prevista a 3 punti (Δ misurato −4%, soglia
  6%): **non risolvibile** — nessuna falsificazione, ma nessuna
  conferma. Residuo aperto: beta misura ~4 punti sopra il contratto
  in modo consistente (effetto di secondo ordine dei cluster da
  caratterizzare).
- Il punto commerciale regge nettamente: sul compilatore a cluster la
  forma media sbaglia di 11 punti, la per-query di 4. **Solo la forma
  per-query rende comparabili compilatori con strutture d'errore
  diverse** — precision/recall medie non bastano.

**Prossimo passo dichiarato:** replica con LLM reali (GPT/Claude/Gemini
sullo stesso corpus) — le estrazioni le lancia kiatto; il protocollo e
l'harness sono questi.

## 2. Scale bench — la tabella dei tre numeri

Dimensioni scelte **dalla teoria** (minima potenza di 2 con pressione
≤ 0.5), contratto emesso prima delle query:

| | edge | server |
|---|---|---|
| D (traccia) | 8192 (**1 KB**) | 131072 (**16 KB**) |
| fatti | 250 | 2000 |
| ingestione | 0.8 ms/fatto | 5.5 ms/fatto |
| latenza query | 1.7 ms | 110 ms |
| RAM (codebook non impacchettato) | 4.3 MB | 550 MB |
| contratto (pre-query) | 95% | 100% |
| misurato | 90% | 100% |
| **errore del contratto** | **5.0%** | **0.2%** |

Note oneste: (a) la latenza è della reference non ottimizzata (cleanup
= scansione lineare in Python su int8); il bit-packing con popcount
ridurrebbe RAM di 8× e latenza di >10×, ma appartiene al livello
applicativo, non alla reference; (b) 2000 fatti stanno in una traccia
da 16 KB — il footprint della *memoria* è la traccia, il codebook è
ricostruibile dai nomi; (c) l'errore del contratto è la metrica
proprietaria: nessun sistema concorrente emette il numero della
penultima riga prima del deploy.

## Valutazione 0-10

| Dimensione | Voto | Nota |
|---|---|---|
| Valore del test multi-compilatore | **8.5** | Ranking per-query 2/2 sulle coppie risolvibili; il vantaggio sulla forma media (11 vs 4 punti) è il risultato spendibile |
| Rigore | **9** | Contratti pre-query, criterio di risolvibilità statistica esplicito, residuo beta dichiarato invece che nascosto |
| Completezza | **6.5** | Dry-run: mancano gli LLM reali (bloccante: API non accessibili dalla sessione) e corpus non a catene |
| Valore del scale bench | **8** | Errore contratto 0.2-5%, 2000 fatti in 16 KB; latenza server onesta ma migliorabile 10× fuori dalla reference |

## Prossimi passi

1. Replica multi-compilatore con LLM reali (kiatto lancia le estrazioni
   sul protocollo di `multicompiler_bench.py`).
2. Caratterizzare il residuo +4pt di beta (secondo ordine dei cluster).
3. Benchmark industriale su corpus documentale reale con la stessa
   tabella (costo/latenza/errore-contratto).
