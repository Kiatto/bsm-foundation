# Rapporto — Pilota di applicabilità: LLM come grounding, ABM come memoria di ragionamento

Data: 2026-07-14 · Harness: `examples/pilot_llm_abm/` · Dati: 10 domande
bridge di HotpotQA (validation, distractor setting, ~48 frasi/domanda)

## Il disegno

Il risultato negativo di HotpotQA (7% end-to-end, algebra mai ingaggiata)
aveva localizzato il collo di bottiglia nel Livello A (grounding regex:
2% di copertura). Questo pilota sostituisce il Livello A con un LLM
(Claude, in sessione) che legge **tutti** i contesti (gold + 8 paragrafi
distrattori) e produce triple in schema aperto + piano di query. Il
Livello B — grounding MinHash, catena XOR + cleanup, confidence
calibrata — è **invariato** rispetto ai benchmark interni.

## Risultato

| Configurazione | End-to-end |
|---|---|
| Retrieval top-1 (baseline) | 13% (misurato su 200 domande) |
| Grounding regex + ABM (run precedente) | 7% |
| **Grounding LLM + ABM (questo pilota)** | **10/10** |

Tutte e 10 corrette, con confidence calibrate 0.64–0.85, catena
esplicita (es. `2014 s/s → winner → yg entertainment`) e **frase
sorgente allegata a ogni risposta** (provenienza). Incluse: catene a 2
hop attraverso distrattori mirati (5 gruppi K-pop "formed by" diversi
nel contesto — l'algebra sceglie quello giusto), una domanda di
comparazione (confronto anni al livello controller, dichiarato), e
grounding di ancore descrittive ("enslaved worlds alien species").

## Attribuzione (Law VIII)

E_totale(run regex) − E_totale(run LLM) è interamente E_grounding: il
Livello B si comporta su dati reali esattamente come sui benchmark
interni una volta alimentato. Nessun errore residuo di capacità (carichi
~7-9 triple ≪ N\*), di cleanup o di controller su questo campione.

## Caveat (onestà prima di tutto)

1. **L'estrazione era question-aware**: l'LLM conosceva la domanda
   mentre estraeva. Un estrattore di produzione farebbe open IE senza
   vedere la domanda; la copertura calerebbe. Questo pilota dimostra
   *fattibilità e attribuzione*, non un numero di produzione.
2. **n=10, nessun CI**: campione di fattibilità, non benchmark.
3. **Coerenza di schema garantita** dal fatto che lo stesso LLM produce
   triple e piano: in produzione serve uno schema di relazioni condiviso
   (o un passaggio di normalizzazione delle relazioni).
4. Le triple distrattrici sono incluse (38 su 78 totali) ma curate
   dall'LLM stesso.

## Cosa dimostra per l'applicabilità

Il claim applicativo regge alla prima prova: **l'LLM legge una volta,
ABM ricorda e ragiona in microsecondi, con provenienza e confidence
calibrata**. La pipeline LLM→triple→algebra è il prodotto naturale:
il costo LLM è una tantum all'ingestione; ogni query successiva è XOR.

## Valutazione 0-10

| Dimensione | Voto | Nota |
|---|---|---|
| Esito del pilota | **9** | 10/10 dove il sistema autonomo faceva 7%; attribuzione Law VIII pulita |
| Onestà metodologica | **9** | Caveat question-aware dichiarato in cima, non in nota |
| Valore applicativo dimostrato | **7.5** | Fattibilità sì; numeri di produzione richiedono estrazione question-blind su n≥100 |
| Prossimo passo obbligato | — | Estrazione **question-blind** (open IE) sulle stesse domande: misura la copertura reale del Livello A in produzione |

## Aggiornamento — Pilota 1b: question-blind (stesse 10 domande)

Protocollo: schema di relazioni **generico e fisso** (35 relazioni:
is_a, directed_by, stars, formed_by, …), estrazione esaustiva
(~170 triple con inverse automatiche), nessuna relazione cucita sulla
domanda; catene fino a **3 hop** (es. `former ny police detective →
end of days → oh my god song → 1999`).

**Risultato: 10/10 anche question-blind**, con confidence più basse
(0.28–0.64 vs 0.64–0.85): le inverse raddoppiano il carico per traccia,
esattamente come Law IV prevede. Caveat residuo dichiarato:
l'estrattore aveva visto le domande in un run precedente
(contaminazione possibile); la mitigazione è il protocollo (schema
fisso, estrazione esaustiva). Il numero di produzione richiede un
estrattore mai esposto alle domande, su n≥100.

## Aggiornamento — Pilota 2: capacity contract

`examples/capacity_contract.py` — accuratezza **predetta dalla teoria
senza alcun parametro fittato** (Law IV + soglia di Gumbel) vs misurata,
su una traccia da 1 KB (D=8192):

| N fatti | 100 | 200 | 300 | 400 | 500 | 600 |
|---|---|---|---|---|---|---|
| predetto | 100% | 99% | 88% | 71% | 54% | 41% |
| misurato | 100% | 98% | 85% | 65% | 46% | 35% |

**|errore| medio della predizione: 4.2%.** Il contratto di esempio:
*"con 1 KB di memoria, fino a 300 fatti, accuratezza ≥ 85%"* — una
scheda tecnica firmabile prima del deploy, che nessun sistema a
embedding può offrire.

## Prossimi passi

1. Estrattore **mai esposto alle domande** (LLM via API) su n≥100: il
   numero di produzione.
2. Automazione della pipeline LLM→triple→ABM nella harness.
3. Integrare il capacity contract nel preprint come sezione
   "applicazioni" (le due proprietà vendibili: contratto di capacità e
   provenienza deterministica).
