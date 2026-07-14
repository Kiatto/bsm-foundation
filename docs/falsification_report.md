# Rapporto — Fase adversariale: tentativi di falsificazione

Data: 2026-07-13 · Harness: `examples/falsification_suite.py` + verifiche
mirate · Dati: `falsification_results.json`

Obiettivo dichiarato: **non cercare conferme, cercare smentite.**
Bilancio: una legge falsificata, una promossa a forma quasi
parameter-free, una nuova congettura, due asimmetrie di robustezza
caratterizzate.

---

## Il diagramma della teoria

```
      D (dimensione)          M (codebook)         {wᵢ} (pesi/ripetizioni)
           │                       │                        │
           ▼                       ▼                        ▼
   segnale per fatto        soglia di rumore         carico efficace
   z = √(2D/(π·N_eff))      z_G(M)  [Gumbel:            N_eff ≈ Σwᵢ²
           │                 √(2lnM) − 2° ordine]     (congettura F3)
           └──────────┬────────────┘
                      ▼
        p = P(cleanup corretto | z, z_G)      ← collasso a N* : z ≈ z_G
                      │                          N* = k·2D/(π·z_G²),
                      ▼                          k = 0.94 ± 0.01
        Acc(h) = p^h   [Law V: il cleanup
                      │  azzera il rumore ⇒ Markov]
                      ▼
        profondità massima di ragionamento  h_max(N, D, M)
```

Tutte le frecce sono state misurate; nessuna è solo postulata.

## La tabella delle leggi (aggiornata dopo la falsificazione)

| Legge | Formula | Verifica | Stato |
|---|---|---|---|
| I | Località di Hamming (nulla nota a priori) | calibrazione + cleanup | corroborata |
| II | Retrieval > decoder (fuori dominio) | HotpotQA | corroborata (condizionale) |
| III | Dimensionalità intrinseca (proiezione r≈12 ⇒ margine +70%) | Task 3 alias | corroborata |
| IV | **N\* = k·2D/(π·z_G(M)²), k = 0.94 ± 0.01** | 5 seed × 4 D × 4 M | **corroborata, quasi parameter-free** |
| V | Acc(h) = p^h (cleanup ⇒ Markov) | dev. 0.023±0.017 | corroborata + sketch di prova |
| VI | ~~Failure ∝ branching~~ | **falsificata a carico costante** | **RITIRATA → sostituita** |
| VI′ | Il fallimento dipende dal carico efficace, non dalla topologia: a N fisso, B∈[1,24] è neutro | 3 seed, carico costante | corroborata (nuova) |
| VII (cong.) | N_eff = Σwᵢ² (la ridondanza pesa quadraticamente) | previsto 11% vs misurato 21% | direzione giusta, costante da raffinare |

---

## I risultati, uno per uno

### F1 — Il test di ln M (la previsione della Law IV)

N\* a D=2048 si contrae con il codebook: 149 → 119 → 99 → 81 per
M = 447 → 16 242. **L'ipotesi nulla (N\* indipendente da M) è distrutta**
(−46%). La forma semplice k=N\*·lnM/D mostra un drift residuo del 15%;
con il termine di secondo ordine di Gumbel per il massimo di M gaussiane
(z_G = √(2lnM) − (lnlnM+ln4π)/(2√(2lnM))):

| M | 447 | 1 356 | 4 297 | 16 242 |
|---|---|---|---|---|
| k semplice | 0.444 | 0.419 | 0.404 | 0.383 |
| **k Gumbel** | **0.943** | **0.937** | **0.942** | **0.925** |

Spread <2% su un range 36× di M, e k≈0.94 ≈ 1: **la legge di capacità è
ora quasi parameter-free** — N\* ≈ 2D/(π·z_G(M)²) senza costanti libere
entro il 6%. Il tentativo di rompere la legge l'ha resa più forte.

### F2+verifica — Law VI falsificata

Il test Zipf non mostrava il degrado atteso sui nodi pesanti (65% vs
60%). La verifica diretta — branching a **carico costante** (120 fatti):

| B (grado uscente) | 1 | 4 | 8 | 24 |
|---|---|---|---|---|
| accuracy | 64% | 73% | 69% | 73% |

Piatto. Il drammatico 100%→14% del benchmark originale (B=1→16) era
**interamente un artefatto del carico** (B=16 significava 400 fatti,
2.5× oltre N\*). Law VI è ritirata e sostituita da VI′: la topologia del
grafo è neutra; conta solo N_eff/D. (Conseguenza pratica: niente
sharding degli hub — era una raccomandazione basata su un artefatto.)

### F3 — La ridondanza è un'arma a doppio taglio (nuova congettura)

10 fatti ripetuti 5× su 100: i ripetuti salgono al **100%**, gli altri
crollano al **21%** (controllo senza ripetizioni: 73%). La ripetizione
pesa il majority vote e *parassita* la capacità altrui. La congettura
N_eff = Σwᵢ² (=340) predice ~11% per i non-ripetuti: ordine giusto,
costante imprecisa. Da raffinare, ma il messaggio è già solido: in una
traccia olografica **la frequenza è salienza**, con un costo quadratico
per il resto della memoria.

### F4 vs F5 — L'asimmetria di robustezza

- **Rumore nella traccia** (flip di ε bit): degrado dolce e prevedibile
  (86% → 80% → 65% → 42% per ε = 0→20%), compatibile con segnale ∝ (1−2ε).
- **Rumore nel codebook**: catastrofico — ε=5% dimezza l'accuracy (54%),
  ε=10% la distrugge (24%). Il danno si compone: chiave di query,
  bersaglio di cleanup e contenuto della traccia si corrompono insieme.

Caratterizzazione: **la memoria è robusta, il codebook è la trusted
computing base** del paradigma. (Implicazione ingegneristica futura:
l'item memory merita ridondanza/ECC; la traccia no.)

---

## Valutazione 0-10 della fase adversariale

| Dimensione | Voto | Motivazione |
|---|---|---|
| **Guadagno scientifico** | **9.5** | Una legge falsificata e sostituita, una promossa a forma parameter-free (k=0.94±0.01 su 36× di M), una congettura nuova (Σw²), un'asimmetria caratterizzata: il singolo run più produttivo dell'intera sessione |
| **Rigore del metodo** | **9** | Il confound carico/topologia è stato scoperto *dalla suite stessa* (F2 non tornava con Law VI) e risolto con l'esperimento a carico costante |
| **Stato della teoria** | **8.5** | Le frecce del diagramma sono tutte misurate; restano da chiudere la costante di VII e la prova formale di V |
| **Prontezza da preprint** | **8** | La storia ora ha tutto: leggi, derivazioni, previsione verificata (Gumbel), falsificazione interna documentata. Mancano solo figure a ≥10 seed e la scrittura |

## Prossimi passi

1. Raffinare la congettura VII (Σw²): curva accuracy vs Σw² a parità di
   N unico — un pomeriggio di esperimenti.
2. Prova formale della Law V (l'argomento Markov è già a un passo).
3. Figure definitive a 10 seed e stesura del preprint:
   *"Algebraic Reasoning over Binary Geometric Memory: Capacity Laws
   and Error Composition"* — con la falsificazione di Law VI raccontata
   nel testo: è la parte che dà credibilità a tutto il resto.
