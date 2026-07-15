# Algebraic Binary Memory — Formalismo

*Versione 2.1 (v2.0 + addendum) — 2026-07-15 — CONGELATO. Ogni implementazione (Python,
C++, FPGA) va valutata rispetto a questa specifica, non viceversa.
Le modifiche richiedono un incremento di versione e la motivazione.*

*Changelog — v1.1: leggi di composizione (§2.7–2.9) e Φ come principio
generatore. v1.2: Bridge Elimination (§2.10), auto-cancellazione
(§2.11), costo di Compose (§3.9), Regola Metodologica. v2.0: struttura
assiomatica (§A: 3 assiomi da cui tutto discende), esperimento di
minimalità (operatori necessari e indipendenti), principio di
conservazione, forma astratta del no-go theorem — compressione del
formalismo: 19 enunciati sono manifestazioni di 3 proprietà.*

> **REGOLA METODOLOGICA (vincolante).** Nessuna nuova legge entra in
> questo formalismo se non produce almeno una predizione quantitativa
> nuova E un esperimento progettato che potrebbe falsificarla. Le
> leggi falsificate restano nel documento, marcate RITIRATA, con il
> dato che le ha smentite.

---

## §A — Struttura assiomatica

Tutto il formalismo discende da tre assiomi, uno per operatore:

**Assioma A1 (Simmetria).** Il binding ⊕ è un'involuzione isometrica
del gruppo (𝔹^D, ⊕) su sé stesso: (x⊕b)⊕b = x, d_H(x⊕b, y⊕b) = d_H(x,y).

**Assioma A2 (Rumore).** Il bundling ⊞ comprime n stati in uno con
correlazione residua √(2/(πn)) per membro e fluttuazioni gaussiane
𝒩(0, √D/2) rispetto a stati estranei.

**Assioma A3 (Proiezione).** Il cleanup è la proiezione idempotente
sul codebook che massimizza Φ: restituisce sempre un elemento esatto
di C.

### L'albero delle derivazioni

```
A1 SIMMETRIA                A2 RUMORE                 A3 PROIEZIONE
   (⊕ involuzione              (⊞ gaussiano)             (cleanup² = cleanup)
    isometrica)
      │                          │                          │
      ├─ Equivarianza (2.7)      ├─ Correlazione 1/√n (2.4) ├─ Reset del rumore
      ├─ Uniformità (2.3)        ├─ Law IV capacità (1.5)   ├─ Law V: Acc=p^h (3.4)
      ├─ Bridge Elimination      ├─ Law VII Σw² (1.6)       ├─ Simbolizzazione
      │    (2.10)                ├─ z-score / confidence    ├─ Attrattori (0.4)
      ├─ Compose (teorema)       └─ Φ come statistica       └─ Prop. 3.6 profondità
      ├─ AUTO-CANCELLAZIONE                                      (via reset)
      │    (2.11: T⊕T=1 —
      │    lo stesso A1 che
      │    dona, vieta)
      └─ Quasi-distributività (2.9)

              A1 × A2 × A3  ⇒  macchina ABM (3.1), inferenza Horn (3.7),
                               costo di Compose P=p² (3.9), Law VIII
```

*Le "19 proposizioni" sono manifestazioni di questi tre assiomi; le
prove restano nelle sezioni sottostanti (Livelli −1…4).*

### Minimalità (misurata): gli assiomi sono necessari e indipendenti

Ablazione di ciascun operatore, 5 capacità (`examples/minimality_bench.py`):

| variante | recall relazionale | olografica | 3-hop | Compose | simboli |
|---|---|---|---|---|---|
| completo | 85% | ✓ O(D) | 67% | ✓ | ✓ |
| senza binding | **0%** (membership 73%) | ✓ | 0% | 0% | ✓ |
| senza cleanup | 0% | ✓ | 0% | ✓ (Φ=3.7σ, mai simbolo) | **0%** |
| senza bundling | **100%** | **✗ (spazio 50×D)** | 100% | ✓ | ✓ |

Ogni rimozione uccide un insieme distinto e non sovrapposto di
capacità: **A1 = struttura relazionale, A3 = simbolizzazione e
profondità, A2 = compressione olografica** — e A2 è l'unico che *costa*
accuratezza (senza bundling tutto migliora, ma lo spazio è O(N·D)):
il bundling è l'operatore della memoria, gli altri due della
computazione.

### Principio di conservazione (osservazione, da formalizzare)

Compose (2.10) e Auto-cancellazione (2.11) insieme: per creare il
fatto A→C si consuma l'identificabilità del grado di libertà interno
B — il fatto composto non contiene B, e il tentativo di riusare i
prodotti intermedi nella stessa traccia annichila l'informazione
(T⊕T=1). *Creazione di conoscenza composta ⇔ distruzione del ponte
come oggetto indipendente.* Lettura fisica: fusione con eliminazione
del grado di libertà interno.

### No-go theorem (forma astratta del Teorema 2.11)

> In qualunque sistema che soddisfi A1, ogni algoritmo che riutilizzi
> direttamente un risultato di decodifica (contenente la traccia T)
> come componente di una successiva chiave di binding sulla stessa T
> produce T⊕T = identità: collasso deterministico, indipendente da
> D, N, M e dal rumore.

Vale per qualsiasi ABM, non per questa implementazione. Corollario:
la profondità richiede A3 (o tracce indipendenti — e l'esperimento
mostra che anche con due tracce il chaining raw fallisce a h≥2:
la simbolizzazione intermedia è, allo stato attuale, l'unico
meccanismo di profondità noto).

---

Notazione: 𝔹 = {−1,+1}; stati x ∈ 𝔹^D; d_H(x,y) = |{ℓ : x_ℓ ≠ y_ℓ}|;
⟨x,y⟩ = Σ x_ℓ y_ℓ = D − 2·d_H(x,y). "u.a.r." = uniforme e indipendente.

---

## Livello 1 — Fisica della memoria

**Definizione 1.1 (Configurazione di memoria).**
𝓜 = (D, M, N, w) con D ∈ ℕ dimensionalità, M ∈ ℕ taglia del codebook,
N ∈ ℕ numero di fatti, w = (w₁…w_N) ∈ ℕ^N distribuzione dei pesi
(molteplicità di scrittura).

**Definizione 1.2 (Traccia olografica).** Dati fatti f₁…f_N ∈ 𝔹^D,
T = maj(f₁ ⊙ w₁, …) ∈ 𝔹^D, il segno della somma pesata bit a bit
(tie-break deterministico).

**Fatto 1.3 (Distribuzione nulla).** Per x, y u.a.r., d_H(x,y) ~
Binomial(D, ½) ≈ 𝒩(D/2, √D/2). *[Law I — base di ogni soglia.]*

**Proposizione 1.4 (Correlazione membro-traccia).** Per w uniforme,
P(T_ℓ = f_{i,ℓ}) = ½ + ½·√(2/(πN)) + O(N^{−3/2}); per w generico, il
segnale del fatto i scala come wᵢ/√(Σⱼwⱼ²). *[Approssimazione gaussiana
del voto; base di Law IV e Law VII.]*

**Legge 1.5 (Capacità — Law IV).** Con z_G(M) = √(2 ln M) −
(ln ln M + ln 4π)/(2√(2 ln M)) (soglia di Gumbel per il minimo di M
distanze nulle), il carico di collasso (accuracy 50%) è

    N* = k · 2D / (π · z_G(M)²),    k_teorico = 1,  k_misurato = 0.92 ± 0.03

*Stato: corroborata su D ∈ [512, 4096], M ∈ [447, 16 242].*

**Legge 1.6 (Ridondanza — Law VII).** Il carico efficace percepito da
un fatto di peso wᵢ è N_eff = Σⱼ wⱼ² (participation ratio), nel regime
di pesi non saturanti (wᵢ ≪ √(Σw²)). *Stato: corroborata (|err| 6.3%
vs 28.5% del modello alternativo); la saturazione del segno per pesi
estremi è identificata e non ancora modellata.*

**Legge 1.7 (Neutralità topologica — Law VI′).** A carico costante,
l'accuratezza è indipendente dal grado uscente dei nodi. *Stato:
corroborata; sostituisce la ritirata Law VI (artefatto del carico).*

**Fatto 1.8 (Asimmetria di robustezza).** Corruzione di frazione ε
della traccia: il segnale scala di (1−2ε) (degrado dolce). Corruzione
del codebook: il danno si compone su chiave, bersaglio e contenuto
(degrado catastrofico: ε=5% dimezza l'accuracy). *Il codebook è la
trusted computing base.*

---

## Livello 2 — Algebra

**Definizione 2.1 (Operatori).** 𝓐 = { ⊕, ⊞, cleanup } su 𝔹^D:
- **binding**: x ⊕ y = prodotto elementwise (≡ XOR in 𝔹);
- **bundling**: ⊞(x₁…x_n) = maj(x₁…x_n);
- **cleanup** (rispetto al codebook C): cleanup_C(x) = argmin_{c∈C} d_H(x, c).
Ausiliario: ρ = permutazione ciclica (marcatore di ruolo).

**Proposizione 2.2 (Il binding è un'involuzione isometrica).**
(x ⊕ y) ⊕ y = x, e d_H(x ⊕ z, y ⊕ z) = d_H(x, y) per ogni z.
*Prova.* Elementwise: y_ℓ² = 1; la seconda segue da x_ℓz_ℓ ≠ y_ℓz_ℓ ⇔
x_ℓ ≠ y_ℓ. ∎

**Proposizione 2.3 (Il binding preserva l'uniformità e randomizza).**
Se y è u.a.r. e indipendente da x, allora x ⊕ y è u.a.r.; inoltre
E[⟨x ⊕ y, x⟩] = 0. *Prova.* Ogni bit x_ℓy_ℓ è ±1 equiprobabile e
indipendente; E[x_ℓ²y_ℓ] = E[y_ℓ] = 0. ∎
*(Corollario: un fatto bound è quasi-ortogonale ai suoi operandi —
la memoria non "trapela" i componenti.)*

**Proposizione 2.4 (Convergenza del bundling).** Per x₁…x_n u.a.r. e
x = ⊞(xᵢ), ⟨x, xᵢ⟩/D → √(2/(πn)) in probabilità, e la deviazione per
bit è O(1/√D). *Prova.* Prop. 1.4 + legge dei grandi numeri sui D bit
indipendenti. ∎ *(Il bundle resta correlato con ogni membro, con
segnale che decade come 1/√n: è la radice fisica della Legge 1.5.)*

**Proposizione 2.5 (Il cleanup è una proiezione idempotente).**
cleanup_C(x) ∈ C per ogni x, e cleanup_C(cleanup_C(x)) = cleanup_C(x).
*Prova.* L'immagine è C; su c ∈ C l'argmin è c stesso (d_H(c,c)=0,
distanze positive altrove per C senza duplicati). ∎
*(È la proprietà che azzera il rumore: il pilastro del Teorema 3.5.)*

**Proposizione 2.6 (Non associatività di ⊞ e perdita d'informazione).**
⊞ non è associativo né invertibile; l'informazione persa è governata
dal Livello 1 (Prop. 1.4), non recuperabile algebricamente ma solo via
cleanup. *(Divisione dei ruoli: ⊕ conserva, ⊞ comprime, cleanup
ripara.)*

### Leggi di composizione (v1.1)

**Teorema 2.7 (Equivarianza del cleanup — esatta).** Per ogni
b ∈ 𝔹^D e ogni x,

    cleanup_{C⊕b}(x ⊕ b) = cleanup_C(x) ⊕ b

dove C⊕b = {c⊕b : c ∈ C}. *Prova.* Per Prop. 2.2 il binding è
un'isometria: d_H(x⊕b, c⊕b) = d_H(x, c) per ogni c; quindi l'argmin è
lo stesso elemento, trasportato da ⊕b. ∎
*(Lettura: il gruppo (𝔹^D, ⊕) agisce per isometrie e il cleanup
commuta con l'azione, purché il codebook sia trasportato. È QUESTA la
ragione strutturale per cui la query elementare funziona:
query(s,r) = cleanup_C(T ⊕ k) è il caso b = k con la convenzione che i
fatti siano memorizzati come chiave ⊕ contenuto.)*

**Corollario 2.7.1 (identità di query).** Se f = k ⊕ c con c ∈ C e
T = f (traccia a un solo fatto), allora cleanup_C(T ⊕ k) = c
esattamente. Con traccia a N fatti, l'errore introdotto è interamente
quello della Prop. 1.4 (rumore di sovrapposizione): la composizione
bind→cleanup non aggiunge alcun errore proprio.

**Proposizione 2.8 (Cleanup del bundle).** Per x₁…x_n ∈ C distinti e
u.a.r., cleanup_C(⊞(xᵢ)) ∈ {x₁…x_n} con probabilità → 1 per D → ∞.
*Prova (sketch).* Per Prop. 2.4, ⟨⊞(x), xᵢ⟩/D → √(2/(πn)) > 0 per ogni
membro, mentre per c ∉ {xᵢ}, ⟨⊞(x), c⟩/D → 0 con fluttuazioni
O(1/√D); il gap è Θ(1/√n) ≫ O(1/√D) per D ≫ n². ∎
*(Il cleanup di una sovrapposizione restituisce un membro — non
un'entità spuria: la composizione ⊞→cleanup è chiusa sul codebook dei
membri, con soglia di validità D ≫ n².)*

**Proposizione 2.9 (Quasi-commutazione bind/bundle).** ⊕ distribuisce
esattamente su ⊞ a meno del tie-break:
b ⊕ ⊞(x₁…x_n) = ⊞(b⊕x₁ … b⊕x_n) per n dispari (per n pari, a meno
dei soli bit di pareggio). *Prova.* Il segno della somma di ±1 commuta
con la moltiplicazione elementwise per b (b_ℓ ∈ {−1,+1} fattorizza
dalla somma). ∎
*(Conseguenza operativa: interrogare una traccia equivale a
interrogare i fatti trasportati — è il passaggio usato implicitamente
nel Teorema 3.4.)*

**Teorema 2.10 (Bridge Elimination — v1.2).** Per fatti
f₁ = c_A ⊕ ρ(c_{r₁}) ⊕ c_B e f₂ = c_B ⊕ ρ(c_{r₂}) ⊕ c_C:

    f₁ ⊕ f₂ = c_A ⊕ ρ(c_{r₁}) ⊕ ρ(c_{r₂}) ⊕ c_C

L'espressione composta **non contiene c_B**: la variabile interna è
eliminata esattamente (non approssimativamente), e c_B non è
ricostruibile dal solo fatto composto (lo è solo con accesso a f₁ o
f₂). *Prova.* Involuzione di ⊕ (Prop. 2.2): c_B ⊕ c_B = 1. ∎
*Verifica: decodifica esatta (d=0); correlazione residua col ponte
0.004 ≈ 0.* *(La composizione algebrica elimina le variabili interne
preservando gli estremi: compressione strutturale + information
hiding — proprietà inesistente nel retrieval.)*

**Teorema 2.11 (Auto-cancellazione della traccia — v1.2).** Il
chaining raw entro una stessa traccia è algebricamente impossibile:
se õ₁ = T ⊕ k₀ viene riusato come componente della chiave successiva,
la seconda decodifica dà T ⊕ (T ⊕ k₀ ⊕ ρ(c_{r₁})) = k₀ ⊕ ρ(c_{r₁}) —
la traccia si auto-cancella e l'informazione memorizzata scompare
deterministicamente. *Prova.* T ⊕ T = 1 (involuzione). ∎
*(La stessa proprietà che dà il Teorema 2.10 vieta il chaining
ingenuo: il cleanup intermedio non è un'ottimizzazione, è
costitutivo della profondità. Nota sperimentale: anche il chaining
raw su DUE tracce alternate fallisce a h=2 (0% misurato dove il
modello a prodotto di correlazioni prediceva ~40%) — la predizione di
decadimento geometrico z_eff = √D·ρ_N^h è FALSIFICATA per h≥2; vale
solo a h=1, dove il collasso su curva unica è confermato su
D ∈ [512, 8192]. La caratterizzazione onesta: senza cleanup la
composizione è confinata a h=1.)*

---

## Livello 3 — Computazione

**Definizione 3.1 (Macchina ABM).** Una macchina ABM è una sestupla

    𝔐 = (𝓜, 𝓐, C, S, I, O)

dove 𝓜 è una configurazione di memoria (Def. 1.1), 𝓐 gli operatori
(Def. 2.1), C ⊂ 𝔹^D il codebook (|C| = M), S uno stato di controllo
finito di taglia O(1) rispetto a N e D, I/O interfacce simboliche
(nome ↔ elemento di C). Un **controller** è un programma su S che
accede alla conoscenza esclusivamente tramite 𝓐.
*(Horn chainer, beam search, planner, LLM: controller diversi sulla
stessa macchina. Il modello è separato dalla pipeline.)*

**Definizione 3.2 (Query elementare).** query(s, r) =
cleanup_C(T ⊕ (c_s ⊕ ρ(c_r))). Costo: 1 binding + 1 cleanup.

**Definizione 3.3 (Costo probabilistico).** p(𝓜) = P(query elementare
corretta), funzione di z_s = √(2D/(π·N_eff)) e z_G(M) (Livello 1).

**Teorema 3.4 (Composizione — Law V, condizionale a un lemma).**
Per una catena di h query con cleanup intermedio,

    P(catena corretta) = p^h + ε,   |ε| ≤ h/M · (1 + o(1)).

*Prova.* (i) *Reset*: per Prop. 2.5 l'output di ogni cleanup è un
elemento esatto di C, quindi la chiave del hop k+1 è funzione di soli
elementi di C: nessun rumore si propaga attraverso i hop. (ii)
*Decorrelazione (Lemma 3.4.1)*: condizionato al successo, i successi
dei hop sono asintoticamente indipendenti, quindi il prodotto p^h.
(iii) *Assorbimento*: da uno stato off-path v privo del fatto per r,
T ⊕ chiave(v,r) è per Prop. 2.3 quasi-ortogonale a ogni c ∈ C, quindi
il cleanup restituisce il nodo corretto con probabilità ≤ 1/M·(1+o(1));
il leakage totale è ≤ h/M. ∎ (dato il Lemma)

**Lemma 3.4.1 (Decorrelazione del rumore — dimostrato al primo
ordine).** Per chiavi distinte k_i ≠ k_j costruite da item u.a.r., le
componenti di rumore dei vettori decodificati õ_i = T ⊕ k_i e
õ_j = T ⊕ k_j sono scorrelate bit a bit:
E[(õ_{i,ℓ} − s_{i,ℓ})(õ_{j,ℓ} − s_{j,ℓ})] = E[k_{i,ℓ}k_{j,ℓ}]·T_ℓ² ·(…)
= 0, poiché k_i e k_j sono indipendenti a media nulla e T_ℓ² = 1.
La scorrelazione bit a bit + CLT su D bit dà indipendenza asintotica
dei successi per D → ∞. *Stato: scorrelazione al primo ordine
dimostrata; l'indipendenza congiunta esatta a D finito resta assunta —
supporto empirico: |Acc − p^h| = 0.020 ± 0.008 su 10 seed.*

**Proposizione 3.5 (Spazio).** Per memorizzare N fatti su codebook M
con accuratezza per-query ≥ 1−δ:

    Space_ABM(N, M, δ) = min D = Θ( N · (z_G(M) + Φ⁻¹(1−δ))² )

dove Φ è la CDF normale. *Lo spazio è lineare nei fatti e
logaritmico nel codebook.* [Inversione della Legge 1.5.]

**Proposizione 3.6 (La profondità è esponenzialmente economica).**
Per una catena di h hop con successo complessivo ≥ 1−δ serve successo
per-hop ≥ 1−δ/h, cioè

    D(h) = Θ( N · (z_G(M) + √(2·ln(h/δ)))² ) = Θ(N·ln M) + Θ(N·ln h)

*La dimensionalità necessaria cresce solo logaritmicamente con la
profondità di ragionamento.* [Da Teorema 3.4 + Prop. 3.5. Predizione
falsificabile non ancora testata.]

**Proposizione 3.7 (Inferenza Horn — supportata sperimentalmente).**
Una macchina ABM con controller di forward chaining, il cui unico
oracolo di verità è il test di membership d_H(f, T) ≤ D/2 − z·√D/2,
decide l'inferenza Horn proposizionale di profondità d con errore
≤ 1 − p^{O(d)}, purché fatti iniziali + derivati ≤ N*.
*Evidenza: ProofWriter depth 0–5 = 99.8%±0.3 … 92.4%±1.4.*

**Proposizione 3.9 (Costo di Compose — v1.2).** La composizione
mediata da cleanup di due fatti richiede due cleanup indipendenti:
P_compose = p². *Verifica: p²=0.80/0.21/0.02 vs misurato
0.78/0.25/0.02 per N=80/160/320.* Per catene di composizioni:
P = p^{2h} — il compose costa il doppio del ragionamento mediato
(Teorema 3.4) in affidabilità, in cambio della generazione del fatto
e dell'eliminazione del ponte (Teorema 2.10).

**Problema aperto 3.8 (Classe computazionale).** Caratterizzare
la classe delle funzioni calcolabili da macchine ABM con D
polinomiale e controller O(1). Congettura di lavoro: l'intorno dei
branching program di larghezza limitata / query relazionali a fan-in
limitato (motivazione: struttura markoviana del cleanup, Teorema 3.4).

---

## Livello 0 — Il potenziale di memoria Φ come principio generatore (v1.1)

**Definizione 0.1 (Potenziale).** Per stati x, y ∈ 𝔹^D,

    Φ_y(x) = ⟨x, y⟩ / (2·√D)  =  (D/2 − d_H(x, y)) / √D

(lo z-score normalizzato di x rispetto alla distribuzione nulla di y).

**Principio 0.2 (Generazione).** Tutte le quantità e le operazioni del
modello sono espressioni di Φ:

| Oggetto | Espressione in Φ | Origine |
|---|---|---|
| distanza di Hamming | d_H = D/2 − √D·Φ | Def. 0.1 |
| z-score (Livello 1) | z = 2Φ | Def. 0.1 |
| confidence calibrata | σ(2Φ/τ) | logistica in Φ |
| cleanup | argmax_{c∈C} Φ_c(x) | Def. 2.1 riscritta |
| membership (Prop. 3.7) | Φ_T(f) ≥ θ | test a soglia |
| binding | azione di gruppo che **preserva Φ**: Φ_{y⊕b}(x⊕b) = Φ_y(x) | Prop. 2.2 + Teor. 2.7 |
| bundling | massimizzatore vincolato: ⊞(x₁…x_n) = argmax_{x∈𝔹^D} Σᵢ wᵢ·Φ_{xᵢ}(x) | vedi Prop. 0.3 |
| capacità (Legge 1.5) | collasso quando Φ_segnale ≈ Φ del massimo di M nulle | Livello 1 |
| ragionamento (Teor. 3.4) | traiettoria che a ogni hop ricade su un argmax di Φ | Livello 3 |

**Proposizione 0.3 (Il bundling è variazionale).** ⊞(x₁…x_n; w) =
argmax_{x∈𝔹^D} Σᵢ wᵢ Φ_{xᵢ}(x). *Prova.* L'obiettivo è separabile per
bit: massimizzare Σᵢ wᵢ x_ℓ xᵢ,ℓ su x_ℓ ∈ {−1,+1} dà
x_ℓ = sign(Σᵢ wᵢ xᵢ,ℓ), cioè il majority vote pesato. ∎
*(Il bundle non è un'operazione ad hoc: è lo stato di massimo
potenziale complessivo rispetto ai membri — l'analogo del centroide
come minimo dei quadrati.)*

**Osservazione 0.4 (lettura energetica).** Con E = −Φ: il retrieval è
discesa dell'energia sul codebook; gli elementi di C sono gli
attrattori (punti fissi del cleanup, Prop. 2.5); la scrittura in
memoria (⊞) costruisce il minimo dell'energia congiunta (Prop. 0.3);
la confidence è una differenza di energia; il ragionamento è una
traiettoria che a ogni hop ricade su un attrattore e riparte a energia
piena (Teorema 3.4: è il reset che rende l'errore moltiplicativo).
*In sintesi: memoria = paesaggio di Φ; algebra = trasformazioni che lo
preservano (⊕) o lo costruiscono (⊞); computazione = discese ripetute
(cleanup). Direzione aperta: dinamica multi-passo su stati fuori
codebook (Hopfield iterato).*

---

## Tabella di stato

| # | Enunciato | Livello | Stato |
|---|---|---|---|
| 1.3 | Distribuzione nulla | 1 | fatto (probabilità elementare) |
| 1.4 | Correlazione membro-traccia | 1 | dimostrata (asintotica) |
| 1.5 | Capacità N* (Gumbel) | 1 | corroborata, k=0.92±0.03 |
| 1.6 | Ridondanza Σw² | 1 | corroborata (regime dichiarato) |
| 1.7 | Neutralità topologica | 1 | corroborata (via falsificazione) |
| 2.2–2.6 | Proprietà algebriche | 2 | dimostrate |
| 2.7 | Equivarianza del cleanup | 2 | **teorema esatto** |
| 2.8 | Chiusura di cleanup∘bundle (D ≫ n²) | 2 | dimostrata (sketch asintotico) |
| 2.9 | Quasi-distributività ⊕/⊞ | 2 | dimostrata |
| 0.3 | Bundling variazionale (argmax di Φ) | 0 | dimostrata |
| 2.10 | Bridge Elimination | 2 | **teorema esatto** + verificato |
| 2.11 | Auto-cancellazione della traccia | 2 | teorema esatto; decadimento geometrico h≥2 **FALSIFICATO** |
| 3.9 | Costo di Compose P=p² | 3 | corroborata (3 carichi) |
| 3.4 | Composizione p^h | 3 | teorema condizionale (Lemma 3.4.1 al 1° ordine) |
| 3.5 | Spazio lineare/log | 3 | derivata da 1.5 |
| 3.6 | Profondità log-economica | 3 | **VERIFICATA**: D_min(64)/D_min(1)=2.1× (lineare⇒64×), R²_log=0.94 vs R²_lin=0.73; congiuntamente |Acc−p̂^h|=0.003 a D fisso |
| 3.7 | Inferenza Horn | 3 | supportata (ProofWriter) |
| 3.8 | Classe computazionale | 3 | aperto |

## Titolo di lavoro del preprint (aggiornato)

**"Algebraic Binary Memory: A Computational Model with Capacity Laws
for Associative Reasoning"**

---

## Addendum v2.1 — 2026-07-15 (il corpo v2.0 resta invariato)

### Due famiglie di risultati

La struttura teorica si articola in due famiglie distinte:

- **Resource Laws** (*quanto costa*): Law I, IV, V, VII, Resource
  Composition Law — dipendono dalle risorse (D, N, M, ε, h).
- **Structural Laws** (*che cosa è possibile*): Bridge Elimination,
  Compose, Auto-cancellazione (no-go §2.11), Aliasing — dipendono
  dalle simmetrie e invarianze del formalismo, NON dalle risorse.

### Projection: da ottimizzazione a operatore di disambiguazione

Riclassificazione. Il cleanup **elimina il rumore** (proiezione sul
codebook, A3). La Projection tipata **elimina le simmetrie
indesiderate** (restrizione del codebook che rompe un'invarianza).
Sono operazioni concettualmente diverse: l'aliasing non è rumore e il
cleanup da solo non può eliminarlo (candidati a pari segnale); la
Projection sì (verificato: guided ≈ p per ogni g testato).

### Simmetria degli archi (proprietà esatta)

L'encoding f = s ⊕ ρ(r) ⊕ o è simmetrico in (s, o): ogni fatto è un
arco non orientato — f ⊕ key(o, r) = s esattamente. Conseguenze:
(i) le query inverse sono gratuite; (ii) a query(node, r), ogni fatto
con `node` come oggetto e relazione r è un candidato a pari segnale.

### Aliasing Factor Hypothesis (IPOTESI, non legge)

    Acc = p(N, M) × Π_i 1/g_i

con g_i = numero di candidati a pari segnale all'hop i (calcolabile
dal piano di query, prima dell'esecuzione). Status: derivazione
algebrica esatta; corroborata a g ∈ {2, 3, 4, 8} (|dev| media 4.2%,
max 7.5%, `aliasing_factor_results.json`). Resta ipotesi finché non
testata su piani misti multi-hop e domini reali. Predizione nuova e
falsificatore: su qualunque piano, il fattore Π 1/g_i dichiarato
dall'Inspector prima delle query deve coincidere con il rapporto
naive/guided misurato.

### Tabella di stato (delta rispetto a v2.0)

| # | Enunciato | Livello | Stato |
|---|---|---|---|
| 2.12 | Simmetria degli archi | 2 | **ESATTA** (algebra) |
| 3.10 | Aliasing Factor | 3 | **IPOTESI corroborata** (4 punti) |
| — | Residuo compilatore-cluster (+4pt) | 3 | APERTO (paper §9.7) |
