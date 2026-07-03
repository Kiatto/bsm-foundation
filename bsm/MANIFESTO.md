# BSM Foundation — Technical Manifesto

**Version 1.0 — July 2026**

---

## Preamble

This manifesto codifies the empirical laws discovered through 30+ experiments
on binary state spaces for language modeling. It is not a research proposal.
It is a record of what has been demonstrated.

---

## Principle 1: Knowledge lives in geometry.

A feedforward encoder maps input to a discrete binary state space.
Semantic relationships between inputs are encoded as Hamming distances
in this space. **Nearest neighbors in state space share meaning.**

*Evidence:* P(same token | d=0) ≈ 0.75 across all experiments. The geometry
is not random; it is semantically structured.

---

## Principle 2: Retrieval is a primitive operation.

Nearest-neighbor retrieval in state space consistently outperforms parametric
decoding for next-token prediction. The ratio GAM/Decoder ≈ 1.6× is invariant
under dimension scaling (D=128..1024).

*Condition:* This holds for linguistic domains (vocabulary ≥ 4000 tokens).
Fails for extremely constrained domains (DNA with 256 tokens) where the decoder
catches up.

*Implication:* The decoder is not the bottleneck — the state space is.
Information is accessible through locality, not through learned weights.

---

## Principle 3: Memory precedes the decoder.

Every attempt to compress the local distribution into a global representation
has failed: dynamics training, prototypes, centroids, distillation, concept
vectors. The information lives in the local geometry of stored states.
There is no shortcut through parametric compression.

---

## Principle 4: Topology matters more than weights.

The model's weights define the encoder, but the encoder merely projects into
a space where topological relationships (Hamming neighborhoods) are
informative. Increasing weight capacity (more parameters, more dimensions)
does not improve retrieval beyond a fixed ceiling (~15.5% for 4-token
TinyStories). The limit is in the data, not the model.

---

## Principle 5: Intrinsic dimensionality is domain-limited.

The state manifold has intrinsic dimension PR ≈ 33 ± 4 for natural language
(TinyStories), invariant of nominal dimension D=128..1024. This PR varies
by domain: code (PR≈14), DNA (PR≈16). The model cannot be forced to use
more dimensions than the input structure provides.

---

## Principle 6: Efficiency is Knowledge Density.

The relevant metric is not accuracy alone, but accuracy per unit of memory,
latency, or energy. Knowledge Density (KD) = bits of uncertainty reduced
per MB of stored memory. DSG achieves KD ≈ 1.23 bits/MB in less than 1 MB
total memory for 4096-class prediction.

---

## Architectural Invariants (core BSM)

```
Input (48 bits)
  → Encoder (linear + tanh)
    → Raw state (continuous, ID≈12)
      → sign()
        → Binary state (±1, ID≈33)
          → Memory (content-addressable)
            → Retrieval (Hamming + LSH)
              → Vote → Prediction
```

This architecture is frozen as **BSM 1.0**. No further modifications to the
core stack without a new major version.

---

## Definitions

| Term | Definition |
|------|------------|
| State | ±1 vector of dimension D |
| GAM | Geometric Associative Memory — nearest-neighbor retrieval |
| PR | Participation Ratio — effective number of dimensions |
| KD | Knowledge Density — bits reduced per MB of memory |
| LSH | Locality-Sensitive Hashing — fast candidate selection |
| Decoder | Linear projection D → 12 bits (token prediction) |
| ID | Intrinsic Dimension (estimated via PR) |

---

## What BSM Is Not

- Not a Transformer replacement
- Not a language model
- Not a neural network architecture
- Not a compression algorithm

## What BSM Is

- A **memory paradigm**: organize information as discrete geometry
- A **retrieval primitive**: nearest-neighbor as core operation
- A **knowledge density system**: maximize information per byte
- A **platform**: composable, measurable, persistent
