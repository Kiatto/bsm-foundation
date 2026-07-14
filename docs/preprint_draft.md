# Algebraic Binary Memory: A Computational Model with Capacity Laws for Associative Reasoning

*Bozza di preprint — v0.2, 2026-07-13 — il formalismo di riferimento è
[FORMALISM.md](FORMALISM.md) (congelato, v1.0); questo documento ne è
la presentazione narrativa.*

## Abstract

Studiamo una classe di memorie di ragionamento in cui la conoscenza vive
in una singola traccia olografica binaria (majority vote di fatti
XOR-bound in {−1,+1}^D) e l'inferenza multi-hop è una sequenza di
unbinding (XOR) e cleanup. Deriviamo e verifichiamo sperimentalmente due
leggi quantitative: (i) una **legge di capacità quasi parameter-free**,
N\* = k·2D/(π·z_G(M)²) con k = 0.92 ± 0.03, dove z_G è la soglia di
valore estremo (Gumbel) del codebook di M item — verificata su un range
8× di D e 36× di M; (ii) una **legge di composizione dell'errore**,
Acc(h) = p^h, che dimostriamo essere conseguenza dell'azzeramento del
rumore operato dal cleanup (struttura markoviana), con deviazione
empirica 0.020 ± 0.008. Il sistema raggiunge 99.8–92.4% (±1.4) su
ProofWriter (OWA, depth 0–5) usando come unico oracolo di verità una
distanza di Hamming da un singolo vettore. Documentiamo inoltre una
falsificazione interna (l'apparente collo di bottiglia topologico era un
artefatto del carico) e una legge di ridondanza (N_eff = Σw², con
saturazione), e mostriamo con un risultato negativo su HotpotQA che il
collo di bottiglia del paradigma su testo libero è il grounding, non
l'algebra.

## 1. Introduzione

La domanda: **quali classi di inferenza sono chiuse rispetto alle
operazioni di binding e unbinding su stati discreti?** A differenza dei
sistemi RAG/LLM, il paradigma studiato è deterministico, ha una
distribuzione nulla nota a priori, e — come mostriamo — proprietà
quantitative *predicibili prima di eseguire il sistema*.

Contributi: (1) leggi di capacità e composizione con derivazioni e CI al
95%; (2) validazione esterna su ProofWriter; (3) metodologia di
separazione dei livelli (grounding / algebra / verbalizzazione) motivata
da un risultato negativo; (4) una falsificazione interna documentata.

## 2. Modello

Item memory C = {c_1…c_M} ⊂ {−1,+1}^D (hypervector deterministici).
Fatto (s, r, o) → f = (c_s ⊙ ρ(c_r)) ⊙ c_o, con ⊙ prodotto elementwise
(≡ XOR) e ρ permutazione ciclica. Traccia T = maj(f_1…f_N) (tie-break
deterministico). Query(s, r): decodifica õ = T ⊙ (c_s ⊙ ρ(c_r)), cleanup
= argmin_{c∈C} d_H(õ, c). Inferenza a catena: h query con cleanup
intermedio. Forward chaining logico (ProofWriter): l'oracolo di
membership è d_H(f_candidato, T) sotto soglia z ≥ 3.

## 3. Le leggi

### Law IV — Capacità (derivazione)

Per un fatto membro, ogni bit di T concorda con f_i con probabilità
q = ½ + ½√(2/(πN)); lo XOR è un'isometria, quindi il segnale decodificato
ha z-score z_s = √(2D/(πN)) contro la nulla N(D/2, √D/2). Il cleanup
fallisce quando il minimo di M−1 distanze indipendenti scende sotto il
segnale; con la correzione di secondo ordine di Gumbel,
z_G(M) = √(2lnM) − (ln ln M + ln 4π)/(2√(2lnM)). Il collasso (accuracy
50%) avviene a z_s ≈ z_G:

    N* = k · 2D / (π · z_G(M)²)

Teoria: k = 1. **Misurato: k = 0.92 ± 0.03** (D ∈ {512…4096},
M ∈ {447…16 242}; forma semplice N\*·lnM/D deriva del 15%, forma Gumbel
stabile entro il 2% sul sweep di M). Il modello lineare N\* = cD è
respinto: c deriva sistematicamente (0.098 → 0.068, CI disgiunti).

### Law V — Composizione dell'errore (proposizione e prova)

**Proposizione.** Con cleanup dopo ogni hop, P(catena corretta dopo h
hop) = p^h + ε con |ε| ≤ h/M·(1+o(1)).

**Prova (sketch formale).** Sia X_k lo stato (on-path/off-path) dopo il
hop k. (i) *Reset del rumore*: il cleanup restituisce un elemento esatto
di C, quindi la chiave del hop k+1 è funzione di soli vettori di
codebook: condizionato a X_k = on-path, il successo del hop k+1 ha
probabilità p, indipendente dalla storia, purché le componenti di rumore
di chiavi distinte siano asintoticamente scorrelate (vero per costruzione:
chiavi = XOR-shift per vettori quasi-ortogonali; supporto empirico:
deviazione 0.020 ± 0.008 compatibile con zero). (ii) *Assorbimento*:
condizionato a X_k = off-path su un nodo v senza fatto per r_{k+1}, la
decodifica è quasi-uniforme rispetto a C e la probabilità di rientrare
sul cammino corretto è ≤ 1/M·(1+o(1)). Il processo è quindi una catena
di Markov a due stati con leakage ≤ 1/M per hop; srotolando,
P = p^h + O(h/M). ∎

**Corollario (predicibilità a priori).** p = p(z_s(N,D), z_G(M)), quindi
Acc(h; N, D, M) è calcolabile senza eseguire il sistema. Verifica:
R²(log Acc vs h·log p) = 0.914 ± 0.065 su 10 seed.

### Law VI′ — Neutralità topologica (via falsificazione)

L'ipotesi iniziale "il fallimento scala con il grado uscente" (accuracy
100%→14% per B=1→16) è **falsificata**: a carico costante (120 fatti),
B ∈ {1,4,8,24} dà accuracy piatta (64–73%). L'effetto era interamente
spiegato dal carico. Enunciato corretto: *il degrado dipende solo dal
carico efficace, non dalla topologia del grafo dei fatti.*

### Law VII — Ridondanza (participation ratio, con saturazione)

Fatti con molteplicità w_i pesano il majority vote: il carico efficace
per i fatti non ripetuti è N_eff ≈ Σw_i². Verifica su 6 configurazioni a
N unico costante: |err| medio 6.3% per il modello Σw² contro 28.5% per
il modello N_tot. Residuo sistematico nei pesi estremi (w ≳ 10): il
segno satura i pesi grandi (w̃ < w), direzione consistente col misurato.
*Conseguenza: in una traccia olografica la frequenza è salienza, con
costo quadratico per il resto della memoria.*

### Robustezza asimmetrica (caratterizzazione)

Rumore nella traccia: degrado dolce, segnale ∝ (1−2ε) (86%→42% per
ε=0→20%). Rumore nel codebook: catastrofico (ε=5% → 54%). L'item memory
è la trusted computing base del paradigma.

## 4. Validazione esterna

**ProofWriter** (OWA, sottoinsieme attributivo senza negazione, 100
domande/depth, 10 seed): depth 0 = 99.8% ± 0.3, depth 2 = 99.1% ± 0.3,
depth 5 = 92.4% ± 1.4 (baseline di maggioranza 42%). Il degrado a
depth 5 è quantitativamente coerente con Law IV (le derivazioni alzano
N). Copertura grammaticale dichiarata: ~35–55% delle domande (4 pattern);
il forward chaining è simbolico, l'oracolo di verità è puramente
algebrico.

**HotpotQA** (distractor, 200 domande bridge — risultato negativo
informativo): l'algebra non viene mai ingaggiata (estrazione regex: 1.9
triple/42 frasi; planner 0.5%); le euristiche multi-hop demo-tuned fanno
7% contro il 13% del top-1 single-hop. Falsifica il *grounding layer*,
non l'algebra; motiva la separazione dei livelli.

## 5. Lavori correlati (bozza)

VSA/HDC (Kanerva; Plate HRR; Gallant & Okaywe MBAT); capacità delle
memorie a sovrapposizione (forma D/lnM); Hopfield (0.138·N, stesso
meccanismo di interferenza gaussiana + evento di coda); ProofWriter e
soft theorem proving (RuleTaker); differenza chiave: qui l'inferenza non
è appresa — è un'operazione algebrica con leggi derivabili.

## 6. Limiti e lavoro futuro

Negazione e quantificatori non testati; copertura grammaticale parziale
su ProofWriter; costante k misurata sotto la teorica (soglia soft vs
max-exceedance, da chiudere analiticamente); saturazione dei pesi in
Law VII da modellare; RuleTaker (scala) e 2Wiki gold-triples (conoscenza
reale a grounding perfetto) come prossimi banchi; grounding su testo
libero aperto (livello A).

## Appendice — riproducibilità

Tutti gli esperimenti: `examples/synthetic_algebra_bench.py`,
`seed_robustness.py`, `falsification_suite.py`, `proofwriter_eval.py`,
`hotpotqa_eval.py`; risultati JSON inclusi nel repo; nessuna dipendenza
oltre numpy (+pyarrow per i dataset); tutti gli hypervector sono
deterministici (seed nei nomi).
