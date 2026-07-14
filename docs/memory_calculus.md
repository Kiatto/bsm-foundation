# Memory Calculus — v0.1

*2026-07-13 — Livello 4 del formalismo (FORMALISM.md v2.0). Regola
metodologica rispettata: ogni teorema del calcolo ha una verifica
meccanica in `examples/` (property test su vettori reali).*

Il ruolo: quello che il λ-calcolo è per la computazione funzionale —
una sintassi, regole di riduzione, forme normali, e una nozione di
equivalenza di programmi. La domanda a cui risponde: **ABM è una
collezione di operatori o una struttura computazionale?**

---

## 1. Sintassi

Atomi a ∈ C (codebook), unità **1** (vettore di tutti +1).

    e ::= a  |  1  |  e ⊕ e  |  ρ(e)  |  ⊞(e, …, e)  |  cleanup_C(e)

Il calcolo è **a due sorte**:
- il **frammento esatto** E: termini con solo {atomi, 1, ⊕, ρ};
- il **frammento probabilistico** P: termini che contengono ⊞ o cleanup.

La distinzione non è stilistica: nel frammento esatto le riduzioni
sono uguaglianze; nel frammento probabilistico ogni riduzione porta
una probabilità di successo, determinata dalle leggi del Livello 1.

## 2. Regole di riduzione

### Frammento esatto (uguaglianze, da A1)

    (E1)  x ⊕ x            →  1                 [involuzione]
    (E2)  x ⊕ 1            →  x                 [unità]
    (E3)  x ⊕ y            =  y ⊕ x             [commutatività, AC]
    (E4)  (x ⊕ y) ⊕ z      =  x ⊕ (y ⊕ z)       [associatività, AC]
    (E5)  ρ(x ⊕ y)         →  ρ(x) ⊕ ρ(y)       [ρ è un omomorfismo]
    (E6)  ρ(1)             →  1

*(E5–E6: la permutazione ciclica distribuisce sul prodotto elementwise
— verifica meccanica in `examples/`.)*

### Frammento probabilistico (riduzioni con costo, da A2+A3)

    (P1)  cleanup(cleanup(e))       →  cleanup(e)                    [p = 1, idempotenza]
    (P2)  cleanup_{C⊕k}(e ⊕ k)      →  cleanup_C(e) ⊕ k              [p = 1, equivarianza]
    (P3)  cleanup_C(⊞(a₁…a_n) ⊕ k') →  aᵢ ⊕ k'   per qualche i       [p = p(z(n, D, M))]
    (P4)  ⊞(x₁…x_n) ⊕ b            →  ⊞(x₁⊕b … x_n⊕b)               [esatta per n dispari]

**Nota critica su P2**: la forma "ingenua" cleanup(e⊕k) → cleanup(e)⊕k
è FALSA senza trasportare il codebook (Teorema 2.7 del formalismo):
l'annotazione C⊕k non è pedanteria, è ciò che rende la regola corretta.

## 3. Normal Form Theorem (frammento esatto) — dimostrato

**Osservazione chiave.** (E, ⊕, 1) con E1–E4 è il **gruppo abeliano
libero di esponente 2** sugli atomi ρ-stratificati: ogni elemento è il
proprio inverso, l'ordine non conta, le coppie si cancellano.

**Teorema NF.** Ogni termine del frammento esatto ha un'unica forma
normale:

    nf(e)  =  ⊕ { ρ^j(a)  :  l'atomo a compare a profondità-ρ j
                             con molteplicità DISPARI in e }

(l'insieme, ordinato canonicamente, degli atomi stratificati a
molteplicità dispari; **1** se l'insieme è vuoto).

*Prova.* (i) E5–E6 spingono ρ sugli atomi (terminante: la taglia sotto
ρ decresce). (ii) Nel gruppo abeliano di esponente 2 ogni parola è
determinata dal vettore delle molteplicità mod 2 degli generatori
(qui: ρ^j(a)); E1–E2 realizzano la riduzione mod 2, E3–E4 permettono
il riordino. Unicità: due parole con lo stesso vettore di molteplicità
mod 2 denotano lo stesso elemento; con vettori diversi, elementi
diversi (libertà del gruppo). ∎

**Corollario NF.1 (Compose non è una regola — è la forma normale).**

    nf( (a ⊕ ρr₁ ⊕ b) ⊕ (b ⊕ ρr₂ ⊕ c) )  =  a ⊕ ρr₁ ⊕ ρr₂ ⊕ c

Il Teorema Compose e il Bridge Elimination (2.10) sono istanze della
normalizzazione: b compare con molteplicità 2 → pari → eliminato.
*(La regola R4 proposta è quindi derivata, non primitiva: il calcolo
è più piccolo del previsto.)*

**Corollario NF.2 (Auto-cancellazione come forma normale).**
nf(T ⊕ (T ⊕ k₀ ⊕ ρr₁)) = k₀ ⊕ ρr₁: il Teorema 2.11 è la
normalizzazione che elimina T. Il no-go theorem è sintattico.

## 4. Confluenza

**Teorema (confluenza del frammento esatto).** Il sistema E1–E6
modulo AC (E3–E4) è terminante e localmente confluente, quindi
confluente (Newman).
*Prova (sketch).* Terminazione: E5–E6 decrescono la profondità di ρ
sui composti; E1–E2 decrescono la lunghezza della parola. Confluenza
locale: le coppie critiche sono tra E1 e E2 (x⊕x⊕1: entrambe le
strade danno 1) e tra E5 e E1 (ρ(x⊕x): → ρ(1) → 1 oppure
→ ρx⊕ρx → 1) — tutte convergenti; l'AC-matching è standard
(molteplicità mod 2). ∎ *(Verifica meccanica: riduzioni in ordini
casuali convergono alla stessa forma normale su termini random.)*

**Frammento probabilistico: confluenza in distribuzione (congettura).**
P1–P2 sono deterministiche e commutano col frammento esatto (P2 È
l'equivarianza). P3 è probabilistica: due strategie di riduzione
diverse dello stesso termine devono dare la stessa *distribuzione* sul
risultato — vero per P3 applicata a redex disgiunti (indipendenza,
Lemma 3.4.1 al primo ordine); per redex annidati è aperto.
**Il costo di una strategia è il prodotto delle p dei suoi passi P3:
Law V è esattamente il teorema di soundness di questa semantica** —
una catena di h riduzioni P3 riesce con p^h (Teorema 3.4), e la
strategia ottima è quella che minimizza i passi P3 (non i passi
esatti, che sono gratuiti e sicuri).

## 5. Conseguenze

1. **Equivalenza di programmi**: due programmi ABM sono equivalenti
   sse hanno la stessa forma normale esatta e la stessa distribuzione
   sulle riduzioni P3. L'ottimizzatore canonico: normalizza il
   frammento esatto (gratis), poi minimizza i cleanup.
2. **Il compilatore esiste già in nuce**: normalizzare prima di
   eseguire = comporre algebricamente tutto il componibile (NF.1) e
   pagare il cleanup solo dove serve. Predizione falsificabile: un
   ragionatore che normalizza prima di decodificare deve costare
   p^(#P3 minimi), meno del chaining naïf.
3. **La gerarchia dei costi**: passi esatti = 0 errore, 0 consumo;
   passi ⊞ = consumo di capacità (Law IV/VII); passi cleanup =
   consumo di affidabilità (Law V). Tre valute, una contabilità.

## 6. Stato e prossimi passi

| Risultato | Stato |
|---|---|
| Normal Form Theorem (esatto) | dimostrato |
| Compose/Bridge/Auto-canc. come istanze di NF | dimostrato (NF.1, NF.2) |
| Confluenza frammento esatto | dimostrata (sketch Newman + verifica meccanica) |
| Confluenza in distribuzione (P3 annidati) | **aperta** |
| Soundness dei costi (= Law V) | ereditata dal Teorema 3.4 |
| Predizione del compilatore (§5.2) | **VERIFICATA** (vedi sotto) |

### Verifica della predizione §5.2 (sleep-time compilation)

Le composizioni esatte F = f₁⊕f₂ (NF.1: gratuite, probabilità 1)
consolidate a sleep-time in una seconda traccia T₂; a query time le
domande a 2 hop costano UN cleanup su T₂ invece di due su T:

| catene | naïf (2 cleanup su T) | p² previsto | compilato (1 cleanup su T₂) |
|---|---|---|---|
| 40 | 82% | 87% | **99%** |
| 80 | 25% | 26% | **89%** |
| 160 | 3% | 3% | **44%** |

Doppio guadagno, entrambi previsti dal calcolo: (i) un solo passo P3
(p invece di p²), (ii) T₂ è metà carica di T (Law IV). Il naïf segue
p² (Law V, di nuovo); il compilato segue p(N₂, D, M). *La
normalizzazione non è un'ottimizzazione stilistica: a carichi alti è
la differenza tra un sistema che ragiona (89%) e uno che non ragiona
(25%). Il "sonno che compila" ha ora una giustificazione derivata dal
calcolo, non ispirata alla biologia.*

Prossimo lavoro: (i) la confluenza in distribuzione; (ii) il test del
compilatore (normalizza-poi-decodifica vs chaining naïf — esperimento
già progettato dalla predizione §5.2); (iii) tipi: un sistema di sorte
che distingua entità/relazioni/tracce renderebbe P2 e P3 verificabili
staticamente.
