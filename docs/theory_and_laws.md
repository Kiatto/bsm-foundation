# Le Leggi della Memoria Algebrica Discreta — teoria e intervalli di confidenza

Data: 2026-07-13 · Harness: `examples/seed_robustness.py` (5 seed
indipendenti, CI 95% t-Student) · Dati grezzi: `seed_robustness_results.json`

Questo documento promuove le osservazioni sperimentali della sessione a
**leggi numerate con incertezze**, e ne deriva due dai principi primi.

---

## Le leggi

**Law I — Località semantica.** Contenuti simili occupano regioni vicine
dello spazio di Hamming; la distribuzione nulla tra stati indipendenti è
Binomial(D, ½) ≈ N(D/2, √D/2), nota a priori e indipendente dai dati
(base della confidence calibrata).

**Law II — Dominanza del retrieval.** Su dati fuori dominio il reasoning
euristico sottrae valore al retrieval puro (HotpotQA: 7% vs 13%);
qualunque layer sopra il substrato deve dimostrare di battere il top-1.

**Law III — Dimensionalità intrinseca.** La qualità dell'addressing
dipende dal sottospazio informativo, non dalla dimensione nominale: la
proiezione lossy fittata a rango r≈12 eguaglia l'accuratezza del binding
esatto con margine +70% (esperimento Task 3, vsa_report).

**Law IV — Capacità algebrica.**

    N* = k · D / ln M        k = 0.445 ± 0.013   (R² = 0.9988)

dove N* è il carico al 50% di accuracy e M la dimensione del codebook.
Il modello lineare ingenuo N*=cD dà c che *deriva* con D
(0.095→0.069 tra D=512 e 4096, R²=0.979): la correzione logaritmica è
reale e misurabile.

**Law V — Composizione degli hop.**

    Acc(h) = p^h             |Acc − p^h| = 0.023 ± 0.017
                             R²(log-log) = 0.862 ± 0.128

**Law VI — Collo di bottiglia topologico.** Il fallimento scala con il
grado uscente del nodo (100%→40%→14% per B=1→8→16 a carico costante
ben sotto N*): il limite è la topologia del grafo, non il numero di fatti.

Validazione esterna (ProofWriter OWA AttNoneg, 5 seed):
depth 0 = 99.8% ± 0.6, depth 2 = 99.1% ± 0.6, depth 5 = 91.6% ± 3.0
(baseline di maggioranza 42%).

---

## Derivazione della Law IV (sketch)

Traccia olografica T = maj(f₁…f_N) con fᵢ hypervector ±1 indipendenti.

1. **Correlazione membro-traccia.** Per ogni bit, la probabilità che la
   maggioranza di N segni casuali concordi con un membro fissato è
   q = ½ + ½·√(2/(πN)) (approssimazione gaussiana della binomiale).
2. **Segnale decodificato.** o′ = T ⊕ chiaveᵢ conserva la correlazione
   (lo XOR è un'isometria): dist(o′, oᵢ) ≈ D(1−q) = D/2·(1−√(2/(πN))).
3. **Rumore.** Gli altri M−1 item del codebook distano ~N(D/2, √D/2);
   lo z-score del segnale è  z = √(2D/(πN)).
4. **Soglia di recupero.** Il cleanup fallisce quando il minimo di M−1
   distanze di rumore scende sotto il segnale; il minimo tipico sta a
   z_max ≈ √(2·ln M). Imponendo z = z_max:

       N* = (1/π) · D / ln M  ≈  0.318 · D / ln M

Misurato: k = 0.445 ± 0.013 — **stessa forma funzionale**, costante ~1.4×
maggiore perché il criterio sperimentale (50% di accuracy media) è più
morbido dell'eccedenza stretta del massimo (che dà accuracy→0), e per le
correzioni di taglia finita della coda binomiale. La previsione
falsificabile: raddoppiare M a parità di D e N deve ridurre l'accuracy
secondo ln M — testabile in un pomeriggio.

Collocazione in letteratura: la forma D/ln M è quella classica delle
memorie a sovrapposizione (superposition capacity, Plate/Kanerva/Gallant);
l'ordine di grandezza c≈0.07–0.09 a M≈2N ricorda il limite di Hopfield
(0.138·N), con cui condivide il meccanismo (interferenza gaussiana +
evento di coda).

---

## Dimostrazione della Law V (sketch)

**Claim.** Con cleanup dopo ogni hop, Acc(h) = p^h a meno di termini
O(1/M) di correzione accidentale.

**Argomento.** Il punto essenziale è che **il cleanup azzera il rumore**:
restituisce un vettore *esatto* del codebook, quindi la query del hop
k+1 è costruita da uno stato privo del rumore accumulato al hop k. Lo
stato del processo è quindi una catena di Markov a due classi:

- *on-path*: la query corrente usa l'entità corretta; il successo del
  hop ha probabilità p, identica per ogni hop (ogni query affronta la
  stessa statistica di traccia — stesso N, D, M);
- *off-path*: dopo un errore, la query usa un'entità sbagliata; il
  ritorno accidentale sul cammino corretto richiede che una decodifica
  ~uniforme sul codebook colpisca il nodo giusto: probabilità O(1/M),
  trascurabile.

Da cui P(successo dopo h hop) = p^h + O(h/M). L'osservazione
sperimentale — deviazione media 0.023 ± 0.017, entro il rumore
binomiale di 30 catene — è consistente con il termine di correzione
nullo. ∎ (sketch)

**Corollario (unificazione IV+V).** p è a sua volta funzione di
(N, D, M) tramite lo z-score della Law IV, quindi l'accuratezza
end-to-end di una catena è predicibile *a priori*:

    Acc(h; N, D, M) ≈ [P_cleanup(z(N, D, M))]^h

Le due leggi non sono indipendenti: sono la stessa fisica (interferenza
gaussiana in spazio di Hamming) vista su un fatto e su una catena. È
questa la caratterizzazione che nessun benchmark pubblico può dare da
solo.

---

## Sul nome e sulla narrativa

La proposta di invertire la gerarchia narrativa è accolta nei rapporti:

    Algebra discreta  →  Memoria (supporto fisico)  →  Grounding  →  Linguaggio

Titolo di lavoro per il preprint:
**"Algebraic Reasoning over Binary Geometric Memory: Capacity Laws and
Error Composition"** — le figure centrali sono già prodotte: (1) N* vs
D/ln M con CI, (2) log Acc vs h·log p, (3) accuracy vs branching,
(4) ProofWriter per depth con CI. (Il rinominare il package `bsm` è
rimandato: è un breaking change che ha senso fare insieme al preprint.)

## Valutazione 0-10 di questo passo

| Dimensione | Voto | Motivazione |
|---|---|---|
| **Maturità teorica** | **8.5** | Era 7.5: due leggi derivate dai principi primi, una previsione falsificabile enunciata (dipendenza da ln M), unificazione IV+V |
| **Rigore statistico** | **8.5** | CI 95% su 5 seed per tutte le grandezze; il processo ha *corretto* una legge (lineare → log-corretta, R² 0.979→0.9988): il metodo funziona |
| **Onestà del risultato** | **9** | La costante misurata (0.445) non coincide con quella teorica (0.318): discrepanza dichiarata e spiegata, non nascosta |
| **Prontezza da preprint** | **7.5** | Mancano: seed ≥10 per le figure finali, copertura grammaticale completa di ProofWriter, il test ln M, e la scrittura |

**Nota di processo**: l'esercizio dei CI non ha solo aggiunto barre
d'errore — ha *cambiato una legge*. Il valore c=0.154 del primo run
(griglia grossolana, singolo seed) era sbagliato del 2×; la stima
robusta ha rivelato la deriva con D e da lì la forma D/ln M. È
l'argomento definitivo per cui questa fase andava fatta prima di
qualsiasi altro benchmark.
