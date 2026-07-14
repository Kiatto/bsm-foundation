# Algebraic Binary Memory: A Resource Theory for Associative Computation

**Preprint v1.0 — July 2026**
*Normative specification: [FORMALISM.md](FORMALISM.md) (frozen, v2.0).
Reference implementation: [`reference/abm.py`](../reference/abm.py).
All experiments reproducible from `examples/`; raw results in the
repository as JSON.*

## Abstract

We study Algebraic Binary Memory (ABM): a computational model in which
knowledge lives in a single holographic binary trace — the bitwise
majority vote of XOR-bound facts in {−1,+1}^D — and reasoning is a
sequence of unbinding (XOR) and cleanup operations. We give a minimal
axiomatization (A1: binding is an isometric involution; A2: bundling has
Gaussian superposition statistics; A3: cleanup is an idempotent
projection onto a codebook) and show experimentally that the three
operators are *necessary and independent*: ablating each destroys a
disjoint capability set. From the axioms we derive two quantitative
laws and verify them with confidence intervals: a **capacity law**
N\* = k·2D/(π·z_G(M)²) with k = 0.92 ± 0.03, stable within 2% over a
36× codebook range once the second-order Gumbel term is included; and
an **error-composition law** Acc(h) = p^h (mean deviation
0.020 ± 0.008), which we prove is a structural consequence of cleanup
idempotence rather than an empirical regularity. The model makes
predictions in advance of measurement, two of which we verify —
reasoning depth is exponentially cheap (D_min grows 2.1× from 1 to 64
hops, versus 64× under a linear model), and sleep-time algebraic
compilation converts p² chains into single-cleanup queries (89% vs 25%
at high load) — and two of which we falsify and retain (a topological
bottleneck law and a geometric-decay model of raw chaining). A Memory
Calculus with a Normal Form Theorem shows that fact composition and
bridge elimination are instances of normalization in the free Boolean
group, not primitive rules. Externally, a forward chainer whose only
truth oracle is a Hamming distance to a single vector reaches
99.8–92.4% (±1.4) on ProofWriter at depths 0–5, and a negative result
on HotpotQA cleanly attributes end-to-end failure to the grounding
layer rather than the algebra — an attribution confirmed by a pilot in
which an LLM grounding layer restores 10/10 on the same task. We
propose ABM as a memory model whose accuracy, capacity and reasoning
depth can be contracted *before deployment* (theory-predicted accuracy
within 4.2% of measurement at a fixed 1 KB budget, with no fitted
parameters).

## 1. Introduction

Modern retrieval-augmented systems treat memory as an index: a store
whose only operation is similarity search, with all composition
delegated to a language model. We investigate the opposite division of
labour: a memory whose *operations themselves* compose knowledge, and
ask a model-theoretic question:

> **Which classes of inference are closed under binding and unbinding
> over discrete states?**

This question is independent of any language model, benchmark, or
implementation. Our contribution is a quantitative treatment: a minimal
axiom set, laws with confidence intervals, theorems for what was
previously folklore, predictions made before measurement — and
falsifications retained on record. The model is deterministic, requires
no training, uses no floating-point tensors in the reasoning path, and
its null distribution is known a priori — which is what makes a
*resource theory* (space D, codebook M, depth h, reliability p)
possible at all.

**Methodological rule.** No law enters the formalism without a new
quantitative prediction and a designed falsification experiment.
Retired laws stay in the record with the data that killed them. This
rule is load-bearing: it corrected our capacity law (linear → Gumbel
form), retired one law entirely (§6), and produced the two verified
predictions of §5.

## 2. The model

**States.** x ∈ 𝔹^D with 𝔹 = {−1,+1}; d_H the Hamming distance;
the null distribution between independent states is Binomial(D, ½).

**Operators** (with their axioms):

- **Binding** x ⊕ y: elementwise product (≡ XOR).
  *A1: an isometric involution* — (x⊕y)⊕y = x, distances preserved.
- **Bundling** ⊞(x₁…x_n): bitwise majority vote.
  *A2: Gaussian superposition* — member correlation √(2/(πn)),
  fluctuations 𝒩(0, √D/2) against outsiders.
- **Cleanup** over codebook C: argmin_{c∈C} d_H(x, c).
  *A3: idempotent projection* onto C.

**Facts and traces.** A fact (s, r, o) is encoded as
f = (c_s ⊕ ρ(c_r)) ⊕ c_o with ρ a cyclic shift; a memory is the single
trace T = maj(f₁…f_N). Query(s,r) = cleanup(T ⊕ c_s ⊕ ρ(c_r)).

**The ABM machine** is (𝓜, 𝓐, C, S, I, O): memory configuration,
operators, codebook, an O(1) symbolic controller, and name↔codeword
interfaces. Controllers (Horn chainer, planner, LLM) are programs over
the same machine; the model is separated from the pipeline.

**The potential Φ.** All quantities in the model are expressions of one
scalar field Φ_y(x) = (D/2 − d_H(x,y))/√D: distance, z-scores,
calibrated confidence σ(2Φ/τ), cleanup (argmax Φ), membership tests,
and — non-trivially — bundling itself, which is the variational
maximizer of Σwᵢ·Φ_{xᵢ}(x) (provable bitwise). Binding is the group
action that preserves Φ. Reasoning is a trajectory of Φ-descents with
resets.

### 2.1 The operators are necessary and independent

Ablating each operator kills a disjoint capability set (measured,
D=1024, N=50, 3 seeds):

| variant | relational recall | holographic O(D) | 3-hop | compose | symbols |
|---|---|---|---|---|---|
| full | 85% | ✓ | 67% | ✓ | ✓ |
| −binding | **0%** (content membership survives at 73%) | ✓ | 0% | 0% | ✓ |
| −cleanup | 0% | ✓ | 0% | ✓ (signal 3.7σ, never a symbol) | **0%** |
| −bundling | **100%** | **✗ (50× space)** | 100% | ✓ | ✓ |

Binding carries relational structure; cleanup carries symbolization and
depth; bundling carries compression — and is the only operator that
*costs* accuracy: it is the memory operator, the other two are the
computation operators.

## 3. Capacity laws

**Law IV (capacity).** The load at which query accuracy crosses 50% is

  N\* = k · 2D / (π · z_G(M)²),  k = 0.92 ± 0.03

where z_G(M) = √(2 ln M) − (ln ln M + ln 4π)/(2√(2 ln M)) is the
second-order Gumbel threshold for the minimum of M null distances.
Derivation: member-trace correlation (A2) gives a signal z-score
√(2D/(πN)); retrieval fails when the extreme of M−1 null distances
crosses it. The naive linear model N\* = cD is *rejected by our own
data*: c drifts systematically with D (0.098→0.068, disjoint CIs over
D ∈ [512, 4096]); the Gumbel form is stable within 2% over
M ∈ [447, 16 242] (R² = 0.9988 vs 0.979 linear). The ln M dependence
was posed as a prediction and confirmed before the constant was
refined.

**Law VII (redundancy).** A fact written with multiplicity wᵢ weights
the majority vote; the effective load felt by singletons is
N_eff = Σwⱼ² (participation ratio). Model comparison over six weighted
configurations at constant unique N: mean |error| 6.3% for the Σw²
model versus 28.5% for total-count; the residual at extreme weights
(w ≳ 10) is sign saturation, identified and unmodelled. Frequency is
salience, at quadratic cost to the rest of memory.

**Law VI′ (topological neutrality) — via internal falsification.** An
earlier law ("failure scales with node out-degree", 100%→14% for
B=1→16) was **falsified at constant load**: accuracy is flat (64–73%)
for B ∈ [1, 24] when total facts are fixed. The original effect was
entirely a load artifact. Only N_eff/D matters; graph topology does
not.

**Robustness asymmetry.** Flipping a fraction ε of trace bits degrades
gracefully (signal ∝ 1−2ε; 86%→42% at ε=20%). Corrupting the codebook
is catastrophic (ε=5% halves accuracy): key construction, cleanup
targets and stored content corrupt jointly. The item memory is the
trusted computing base of the paradigm.

## 4. Error composition and the Memory Calculus

**Theorem (hop composition).** With cleanup between hops,

  P(h-hop chain correct) = p^h + ε, |ε| ≤ h/M·(1+o(1)).

*Proof structure.* (i) *Noise reset*: cleanup returns an exact codebook
element (A3), so each hop's key is built from noise-free vectors; (ii)
*decorrelation*: noise components of distinct keys are uncorrelated
bitwise (E[k_i k_j] = 0; proven to first order, and measured:
success-event correlation φ = 0.014 ± 0.024, bit-level noise
correlation −0.005 across 1 800 query pairs); (iii) *absorption*:
off-path decodes are near-uniform over C, so return probability is
O(1/M). Empirically |Acc − p^h| = 0.020 ± 0.008 over 10 seeds. The law
is a structural property of the composition cleanup→bind→cleanup, not
of any benchmark.

**Memory Calculus.** Terms e ::= a | 1 | e⊕e | ρ(e) | ⊞(e…) |
cleanup(e), in two sorts: an *exact fragment* (⊕, ρ) and a
*probabilistic fragment* (⊞, cleanup).

- **Normal Form Theorem (exact fragment, proved).** (E, ⊕, 1) is the
  free Boolean group on ρ-stratified atoms: every exact term has the
  unique normal form "atoms of odd multiplicity". Mechanically
  verified: 200/200 random reduction orders converge.
- **Compose is not a rule.** For facts sharing a bridge,
  nf(f₁ ⊕ f₂) = c_A ⊕ ρr₁ ⊕ ρr₂ ⊕ c_C: the two-hop fact is *generated*
  by normalization, and the bridge is *exactly eliminated*
  (**Bridge Elimination**: the composed fact contains no information
  about c_B — structural compression and information hiding,
  unavailable to any retrieval system).
- **No-go theorem.** T ⊕ T = 1: any algorithm that reuses a raw decode
  (containing T) inside a subsequent key on the same trace collapses
  deterministically. Depth *requires* symbolization (A3). We also
  falsified our own softer model here: geometric signal decay
  z_eff = √D·ρ^h predicts measurable accuracy for two-trace raw
  chaining at h=2; measurement gives 0%. Raw composition is confined
  to h=1; the prediction is retained as falsified.
- **Confluence.** The exact fragment is terminating and confluent
  (Newman); distributional confluence for disjoint probabilistic
  redexes follows from the measured independence; nested redexes are
  open.
- **Cost semantics.** Exact steps are free and certain; ⊞ spends
  capacity (Law IV/VII); cleanup spends reliability (Law V). The hop
  composition theorem is precisely the soundness of this semantics.

## 5. Predictions verified in advance

**P1 — Depth is exponentially cheap.** From Laws IV+V:
D_min(h) = Θ(N ln M) + Θ(N ln h). Measured at constant load (N=120),
target 95% chain accuracy:

| h | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| D_min | 4035 | 4656 | 6305 | 6305 | 6790 | 7275 | 8488 |

Growth 1→64 hops: **2.1×** (a linear model predicts ~64×); log fit
R² = 0.94 vs 0.73 linear. The limiting resource of associative
reasoning is load, not depth.

**P2 — The compiler.** The calculus predicts that normalizing before
decoding (composing facts exactly at "sleep time" into a second trace)
converts p² two-hop queries into single-cleanup queries at the
compiled trace's load. Measured: 99% vs 82% (40 chains), **89% vs 25%**
(80), **44% vs 3%** (160); the naive path tracks p² throughout.
Offline consolidation is not biologically-inspired heuristics; it is
the optimal strategy prescribed by the cost semantics.

**P3 — Typed projection.** Restricting cleanup to a typed sub-codebook
S buys capacity by exactly z_G(M)²/z_G(|S|)² (no free parameters).
Measured gains 1.84/2.25/2.68× vs predicted 1.94/2.37/2.82× over a 16×
distractor range — within 5% — and typed cleanup is immune to codebook
inflation. This is geometric attention with a price list.

**Compose cost.** P_compose = p² (two independent cleanups): measured
0.78/0.25/0.02 vs predicted 0.80/0.21/0.02 across three loads.

## 6. Falsifications retained

| retired claim | killed by |
|---|---|
| Law VI: failure ∝ out-degree | constant-load ablation: flat 64–73% for B ∈ [1,24]; original effect was load |
| geometric decay of raw chaining (z_eff = √D·ρ^h for h≥2) | two-trace raw chaining: 0% measured where the model predicts ~40%; T⊕T no-go is the sharp form |

Both remain in the formalism, marked RETIRED, with the data.

## 7. External validation and error attribution

**ProofWriter** (OWA, attribute fragment, 100 questions/depth, 10
seeds). Facts live only in the holographic trace; forward chaining uses
a single truth oracle: d_H(fact-hv, T) under a z ≥ 3 threshold. Derived
facts are written back (noise grows with depth — part of the test).

| depth | 0 | 2 | 5 | majority baseline |
|---|---|---|---|---|
| accuracy | 99.8% ± 0.3 | 99.1% ± 0.3 | 92.4% ± 1.4 | 42% |

The depth-5 degradation is quantitatively the load increase Law IV
predicts. Declared limits: ~35–55% grammatical coverage (4 patterns);
negation subset untested; chaining control is symbolic, the truth
oracle is purely algebraic.

**HotpotQA (negative result).** With a regex grounding layer, the
algebra is never engaged (1.9 triples per 42 sentences; 0.5% of
queries planned) and demo-tuned multi-hop heuristics *subtract* value
(7% vs a 13% single-hop baseline; oracle 100%, chance 4.9%). This
falsifies the grounding layer, not the algebra — and motivates the
error-conservation principle (Law VIII): every failure is attributable
to exactly one level (grounding / capacity / cleanup / controller) by
level ablation.

**Attribution confirmed (pilot).** Replacing the regex layer with an
LLM extractor (all 10 contexts read, gold + distractors; generic
35-relation schema; question-blind protocol with auto-inverses;
chains up to 3 hops) restores **10/10** on the same task with
calibrated confidences and per-answer provenance — and the lower
confidences of the blind protocol (0.28–0.64 vs 0.64–0.85) are the
Law IV effect of the doubled load. Declared caveats: n=10 feasibility
sample; extractor previously exposed to the questions; production
numbers require a blind extractor at n ≥ 100.

**Capacity contract.** At a fixed 1 KB budget (D=8192), theory-predicted
accuracy (Laws IV + Gumbel, zero fitted parameters) matches measurement
with 4.2% mean error over N ∈ [100, 600]: *"up to 300 facts at ≥ 85%
accuracy in 1 KB"* is a spec sheet signable before deployment.

**Resource Composition Law (empirically corroborated, per-query form).**
Injecting four extraction-error types at rates ε ∈ [0, 0.5] (2-hop
chains, D=2048), end-to-end accuracy follows Acc = E_q[Pg(q)] ·
Pr(N_eff(ε)): grounding and reasoning compose multiplicatively, with
the reasoning factor evaluated at the load the grounding actually
leaves (missing facts also *lighten* the trace). Mean deviation ~3%
over 40 cells. Stress-tested against non-i.i.d. structures: the
per-query form survives (|dev| 2.3–4.8%) while the mean-precision form
Acc = p̄^k · Pr breaks exactly where predicted — cluster-correlated
errors outperform i.i.d. errors of equal mean rate (52% vs 30% at
ε = 0.4) because the damage concentrates on fewer queries. One
prediction was falsified and is retained: chain confidence does *not*
detect grounding errors (the confident-wrong signal exists per hop but
dilutes in the product), so grounding must be audited at its own level.

**Compiler ranking (dry run).** Three simulated extractors with equal
apparent quality but different error structure (uniform 8%, whole-chain
clusters at 20%, recall-tuned with 30% spurious facts) were compiled
into memories and ranked by the per-query contract *before any query*.
On the statistically resolvable pairs the predicted ranking matches the
observed one (2/2 at 95% CI, 10 seeds); on the cluster extractor the
mean-precision form errs by 11 points where the per-query form errs
by 4. Corollary: at equal mean precision, an extractor that fails in
clusters is preferable to one that fails uniformly — a selection
criterion no precision/recall metric expresses. Real-LLM replication
is the natural next step.

## 8. Related work

Vector Symbolic Architectures and hyperdimensional computing (Kanerva;
Plate's HRR; Gallant & Okaywe's MBAT; Rachkovskij & Kussul's
context-dependent thinning) supply the operator vocabulary; superposition
capacity of the D/ln M form is classical. Hopfield networks share the
interference-plus-extreme-value mechanism (our measured c ≈ 0.07–0.09
at M ≈ 2N recalls the 0.138·N regime). ProofWriter/RuleTaker study soft
theorem proving with transformers. Our contribution relative to this
literature is the *resource theory*: laws with confidence intervals and
derivations, a calculus with normal forms whose cost semantics is
proved sound, predictions issued before measurement, and an
axiomatization justified by operator-ablation.

## 9. Open problems

1. Distributional confluence for nested probabilistic redexes.
2. The ABM complexity class (polynomial D, O(1) controller); working
   conjecture: the neighbourhood of bounded-width branching programs.
3. Exact finite-D independence in the hop-composition lemma (first-order
   decorrelation proved; joint independence measured but assumed).
4. Sign-saturation correction to Law VII at extreme weights.
5. Negation and quantifiers in the algebraic truth oracle.
6. Φ-descent dynamics off-codebook (Hopfield-style multi-step cleanup).

## 10. Conclusion

ABM is not proposed as a better retriever — on generic semantic search
it is not competitive, and we say so with measurements. It is proposed
as a *computational model of associative memory with predictable
resources*: three necessary and independent operators, laws that
survived deliberate falsification attempts, a calculus in which
composition and abstraction are normal-form phenomena, and contracts
(capacity, depth, reliability) that can be signed before the system
runs. The theory's strongest evidence is that it corrected and retired
its own laws — and that its predictions, from depth-scaling to
compilation to typed projection, were confirmed *after* being derived.

---

*Reproducibility: every number in this paper is produced by a script in
`examples/` with results committed as JSON; the formal specification is
frozen as FORMALISM.md v2.0; the reference implementation
(`reference/abm.py`, < 500 lines, numpy-only, deterministic) passes the
property tests derived from the axioms.*
