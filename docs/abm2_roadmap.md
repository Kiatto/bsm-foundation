# ABM 2.0 — Da framework a teoria della memoria

*2026-07-13 — documento di direzione. Regola invariata: per ogni nuovo
oggetto, PRIMA il test che potrebbe falsificarlo.*

La domanda del progetto è cambiata quattro volte ed è arrivata alla
forma stabile: **esiste una teoria quantitativa della memoria
associativa algebrica?** Questo documento organizza i sei livelli di
ABM 2.0, con lo stato onesto di ciascuno: *dimostrato / dimostrabile /
speculativo*, e il test di falsificazione associato.

---

## Livello −1 — Geometria dell'informazione (speculativo, promettente)

Tesi: Φ non è una statistica derivata — è la struttura primitiva;
Hamming è una coordinata. Il passo formale: definire il paesaggio
𝔹^D → ℝ, x ↦ Φ_T(x) e studiarne gradiente discreto (flip di bit che
massimizza ΔΦ), attrattori (già noti: il codebook, Prop. 2.5),
curvatura (da definire — candidato: spettro locale delle ΔΦ sui D flip).

**Stato:** la riscrittura è già completa (FORMALISM §0); la geometria
*dinamica* (discese multi-passo fuori codebook, alla Hopfield) è
aperta. **Falsificazione:** se la discesa greedy di Φ da uno stato
rumoroso non converge agli stessi attrattori del cleanup (o converge a
minimi spuri in misura non predetta dalla teoria), il livello −1 non
aggiunge nulla al cleanup e va ritirato.

## Livello 0 — Operatori: + Projection (dimostrabile subito)

**Definizione.** Π_S(x) = cleanup_{C_S}(x), con C_S ⊆ C il
sotto-codebook del tipo S (aziende, persone, luoghi).

Il punto non è l'implementazione (banale) — è che **la teoria delle
risorse ne quantifica il beneficio senza esperimenti nuovi**: tutte le
leggi dipendono da M solo tramite z_G(M), quindi proiettare su |S| ≪ M
compra capacità esattamente prevedibile:

    N*(Π_S) / N*(cleanup) = z_G(M)² / z_G(|S|)²

**Falsificazione:** misurare N* con cleanup ristretto a |S| = M/10:
se il guadagno non segue il rapporto dei z_G², la Law IV è incompleta.
*(È l'equivalente geometrico dell'attenzione, ma con un prezzo
dichiarato in anticipo — nessun softmax appreso.)*

## Livello 1 — + Law VIII: Conservazione dell'errore

    E_totale = E_grounding + E_capacità + E_cleanup + E_controller

Ogni fallimento appartiene a un livello; non esiste errore misterioso.
Non è un auspicio: è già stata *usata* due volte senza nome —
HotpotQA (errore di grounding, non di reasoning) e Law VI (errore di
attribuzione: carico, non topologia). Formalizzazione: dato un
fallimento, esiste una procedura di attribuzione univoca (ablazione
per livello: triple oracolari → isola grounding; codebook pulito →
isola cleanup; ecc.).

**Falsificazione:** un fallimento riproducibile che sopravvive a tutte
le ablazioni di livello falsificherebbe la decomposizione (o
rivelerebbe un livello mancante — entrambi progressi).

## Livello 2 — + Compose: la memoria genera fatti (DIMOSTRATO OGGI)

**Teorema (Composizione transitiva).** Per fatti
f₁ = c_A ⊕ ρ(c_r₁) ⊕ c_B e f₂ = c_B ⊕ ρ(c_r₂) ⊕ c_C:

    f₁ ⊕ f₂ = c_A ⊕ ρ(c_r₁) ⊕ ρ(c_r₂) ⊕ c_C

Il ponte c_B **si cancella algebricamente** (⊕ è un'involuzione,
Prop. 2.2). Il fatto composto A→C esiste dopo UN XOR, e:
cleanup(f₁⊕f₂ ⊕ c_A⊕ρ(c_r₁)⊕ρ(c_r₂)) = c_C esattamente. ∎

*Verifica empirica: dist = 0; correlazione residua col ponte 0.004 ≈ 0.*

Due conseguenze non banali:
1. **La memoria non recupera il fatto a 2 hop: lo genera.** Il ruolo
   del controller si riduce a scegliere *quali* coppie comporre (la
   ricerca), non a eseguire la composizione (algebra).
2. **Il fatto composto non rivela il ponte** (correlazione ~0):
   proprietà di astrazione — e potenzialmente di privacy — gratuita.

**Falsificazione/limite da misurare:** composizione di fatti decodificati
da tracce rumorose (senza cleanup intermedio): il rumore si somma — la
teoria del Livello 1 deve predire il degrado; e composizioni a catena
(f₁⊕f₂⊕f₃…): quante prima del collasso?

## Livello 3 — Controller come funzione C: Φ → Action

Riformulazione accolta: il controller non è uno stato, è una politica
sul potenziale — legge i valori di Φ (confidence, membership, margini)
e sceglie l'operatore successivo. Rende confrontabili controller
diversi (Horn, beam, LLM, planner) *sulla stessa interfaccia
osservabile*, e definisce cosa un controller non può fare: accedere ai
bit, solo a Φ. **Test:** riscrivere il forward-chainer di ProofWriter
in questa interfaccia senza perdita di accuratezza (refactoring
concettuale, non feature).

## Livello 4 — Memory Calculus (il traguardo, speculativo)

Primitive: bind, bundle, cleanup, project, compose. Regole di
riduzione candidate (le prime tre già dimostrate!):

    (x ⊕ b) ⊕ b            →  x                    [Prop. 2.2]
    cleanup(cleanup(x))     →  cleanup(x)           [Prop. 2.5]
    cleanup_{C⊕b}(x ⊕ b)    →  cleanup_C(x) ⊕ b     [Teor. 2.7]
    f₁ ⊕ f₂  (ponte comune) →  fatto composto       [Compose]
    b ⊕ ⊞(x…)              →  ⊞(b⊕x…)              [Prop. 2.9]

Manca: una nozione di forma normale, la confluenza, e il costo di ogni
riduzione in risorse (D, N_eff, p). Se esiste, si dimostrano
*programmi*, non benchmark. È il contenuto del Problema aperto 3.8
visto dal lato giusto.

---

## Paper e reference implementation

- **Titolo adottato:** *"Algebraic Binary Memory: A Resource Theory
  for Associative Computation"* — il retrieval è un corollario.
- **Reference implementation < 500 righe:** bind, bundle, cleanup,
  project, compose + ItemMemory + calibrazione Φ. Tutto il resto
  (GraphCache, ReasoningEngine, ensemble, benchmark) migra in un repo
  di esempi. Operazione di sottrazione, da fare *dopo* che il
  formalismo v2 si stabilizza (ordine: teoria → poi congelamento).

## Il rischio, accettato come vincolo

La disciplina anti-eleganza resta la regola: ogni livello sopra porta
il suo test di falsificazione scritto *prima* della sua adozione. Un
oggetto senza test di smentita progettabile non entra nel formalismo.

## Priorità operative

1. **Compose sotto rumore** (l'esperimento che il teorema di oggi
   rende obbligatorio): composizione senza cleanup intermedio e a
   catena — dove collassa?
2. **Projection**: il test del rapporto z_G² (falsifica o raffina Law IV).
3. Controller Φ→Action come refactoring concettuale di ProofWriter.
4. Memory Calculus: forma normale e confluenza (lavoro carta e penna).
5. Reference implementation, per ultima.
