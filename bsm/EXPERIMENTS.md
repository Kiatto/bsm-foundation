# BSM Foundation 1.0 — Experiment Index

## Core Results

| Experiment | File | Key Finding | Status |
|---|---|---|---|
| Scaling (D=128..1024) | `experiments/scaling.py` | All 7 invariants confirmed; GAM≈15.5%, Decoder≈9.4% | **VALIDATED** |
| Intrinsic Dimension | `experiments/intrinsic_dim.py` | PR≈33±4, invariant with D | **VALIDATED** |
| PR Verification | `experiments/pr_verify.py` | PR_bin≈30, PR_raw≈12, sign doubles ID | **VALIDATED** |
| Cross-Domain | `experiments/cross_domain.py` | PR varies: lang≈35, code≈14, DNA≈16 | **VALIDATED** |

## BSM-X Experiments

| Experiment | File | Key Finding | Status |
|---|---|---|---|
| Weighted Hamming | `experiments/phases_1_4.py` | Reducible to MI weighting | **FALSIFIED** |
| State→State Retrieval | `experiments/phases_1_4.py` | 0.39% accuracy (random) | **FALSIFIED** |
| Bank Ablation | `experiments/phases_1_4.py` | Bank ratio≈1.2×, decorrelated | **VALIDATED** |
| Bank Probing | `experiments/bank_probe.py` | Small cross-bank differences | **VALIDATED** |

## MI Analysis

| Experiment | File | Key Finding | Status |
|---|---|---|---|
| MI Distribution | `experiments/mi_analysis.py` | Gini=0.34, top25%=47% MI | **VALIDATED** |
| Pairwise MI | `experiments/pairwise_mi.py` | Working bank NOT from pairwise | **FALSIFIED** |
| Shuffle Test | `experiments/shuffle_test.py` | Working bank not ordering artifact | **VALIDATED** |

## Falsifications

| Experiment | File | Hypothesis | Result |
|---|---|---|---|
| Dynamics | `experiments/dynamics.py` | Training dynamics helps | −0.64% GAM |
| Working bank = higher MI | `experiments/mi_analysis.py` | Ablation by MI | ratio 1.22× vs 1.75× |
| Working bank = ordering | `experiments/shuffle_test.py` | Permuted bits | persists |
| Working bank = pairwise | `experiments/pairwise_mi.py` | Coordinated interactions | lowest intra-MI |
| More D → better | `experiments/scaling.py` | Scaling accuracy | CV=0.017 |

## Applications

| Application | File | Key Finding | Status |
|---|---|---|---|
| Knowledge Density | `experiments/kd_analysis.py` | KD=1.23 bits/MB, <1MB total | **DEMO** |
| Augmentation (LSTM+BSM) | `experiments/augmentation.py` | +1.65pp over LSTM alone | **DEMO** |

## Running an Experiment

```bash
cd /var/www/html/BitKore
pip install --user --break-system-packages scikit-learn 2>/dev/null  # if needed
python3 bsm/experiments/scaling.py        # 36 min for all D
python3 bsm/experiments/intrinsic_dim.py  # 3 min
python3 bsm/experiments/cross_domain.py   # 4 min
python3 bsm/experiments/augmentation.py   # 3 min
```
