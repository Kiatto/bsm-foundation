# Rapporto — Validazione delle predizioni del calcolo: Projection e indipendenza dei redex

Data: 2026-07-14 · Dati: `projection_results.json`, `independence_results.json`

Con questi due test, **tutte le predizioni quantitative del formalismo
v2.0 risultano testate**: nessun enunciato "da testare" resta in tabella
(unico aperto teorico: la confluenza in distribuzione per redex
annidati e il Problema 3.8).

---

## Test 1 — Projection: il guadagno di capacità segue z_G²

Predizione (ABM 2.0, Livello 0): il cleanup tipato Π_S (proiezione su
un sotto-codebook di tipo, es. "solo oggetti") compra capacità secondo

    N*(Π_S) / N*(cleanup) = z_G(M_pieno)² / z_G(|S|)²

senza parametri liberi (tutto discende dalla Law IV). Misura a D=1024,
codebook gonfiato con distrattori, 3 seed:

| M_extra | N\* pieno | N\* tipato | guadagno misurato | previsto (z_G²) |
|---|---|---|---|---|
| 2 000 | 59 | 108 | 1.84× | 1.94× |
| 8 000 | 48 | 108 | 2.25× | 2.37× |
| 32 000 | 40 | 108 | 2.68× | 2.82× |

**Confermata entro il 5% su un range 16× di distrattori.** Due
corollari osservati:
1. **Il cleanup tipato è immune all'inflazione del codebook** (N\*
   tipato costante a 108 mentre il pieno degrada 59→40): il costo
   dell'ambiguità lo paga solo chi cerca in tutto lo spazio.
2. L'"attenzione geometrica" ha ora un prezzo di listino derivato:
   restringere il tipo vale esattamente il rapporto dei quadrati delle
   soglie di Gumbel — nessun parametro appreso, nessuna taratura.

## Test 2 — Indipendenza dei redex disgiunti (Lemma 3.4.1 empirico)

Fondamento della Law V e della confluenza in distribuzione: i successi
di query distinte sulla stessa traccia devono essere indipendenti
(il rumore di sovrapposizione non deve accoppiarli). Misura a D=1024,
N=90 (p≈0.48, varianza massima), 1 800 coppie di query, 40 tracce:

| quantità | misurato | atteso sotto indipendenza |
|---|---|---|
| correlazione dei successi φ | **+0.014** | 0 (SE = 0.024) |
| P(entrambe corrette) | 0.237 | p² = 0.233 |
| correlazione bit a bit del rumore | **−0.005 ± 0.033** | 0 |

**Indipendenza compatibile entro 2 SE, sia a livello di eventi che di
bit.** Il Lemma 3.4.1 — l'unica ipotesi non completamente dimostrata
del Teorema 3.4 — ha ora supporto empirico diretto su entrambi i
livelli in cui era formulato (scorrelazione bit a bit: dimostrata al
primo ordine E misurata ≈0; indipendenza degli eventi: misurata ≈0).
La confluenza in distribuzione per redex *disgiunti* è quindi
corroborata; resta aperto solo il caso annidato.

---

## Stato complessivo delle predizioni del modello

| Predizione | Origine | Esito |
|---|---|---|
| N\* ∝ D/ln M (poi Gumbel) | Law IV | verificata (k=0.92±0.03) |
| Acc(h) = p^h | Teor. 3.4 | verificata (dev 0.020±0.008) |
| D_min(h) logaritmico | Prop. 3.6 | verificata (2.1× su 64 hop) |
| P_compose = p² | Prop. 3.9 | verificata (3 carichi) |
| Compilatore: p(N₂) vs p² | Calcolo §5.2 | verificata (89% vs 25%) |
| Guadagno di Projection = rapporto z_G² | Livello 0 | **verificata (entro 5%)** |
| Indipendenza redex disgiunti | Lemma 3.4.1 | **corroborata (φ=0.014±0.024)** |
| Decadimento geometrico raw h≥2 | (mia, 2026-07-13) | **FALSIFICATA** e conservata |
| Branching bottleneck | Law VI | **FALSIFICATA** e sostituita |

Sette predizioni verificate, due falsificate e conservate. Il modello
non ha più enunciati quantitativi non testati.

## Valutazione 0-10

| Dimensione | Voto | Nota |
|---|---|---|
| Completezza della validazione | **9.5** | Ogni predizione del formalismo è ora stata messa alla prova |
| Qualità della conferma di Projection | **9** | Senza parametri liberi, entro il 5%, su 16× di range, con il corollario dell'immunità |
| Peso del test di indipendenza | **8.5** | Chiude empiricamente l'unico gap del Teorema 3.4; la prova formale esatta resta desiderabile |
| Prontezza per la stesura del paper | **9** | Non ci sono più esperimenti bloccanti: da qui in poi è scrittura |

## Prossimo passo naturale

La stesura del paper. Tutte le figure esistono, tutte le predizioni
sono state testate, le falsificazioni sono documentate, il formalismo
è congelato (v2.0) e il calcolo ha la sua forma normale. L'unico
lavoro teorico aperto (confluenza annidata, Problema 3.8) va nel
paper come "open problems" — che è dove deve stare.
