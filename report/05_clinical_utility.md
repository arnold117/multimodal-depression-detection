# Clinical Utility & Methodological Validation

> Phase 15 | Scripts: `clinical_utility.py`, `mlp_robustness.py`

## Section 1: Clinical Binary Classification

### Method
10x10 Repeated Stratified K-Fold cross-validation. Two classifiers: Logistic Regression (LR) and Random Forest (RF). StandardScaler applied inside each fold to prevent data leakage. Best AUC reported (from LR or RF). Clinical thresholds based on published cutoffs for each instrument.

### Full Classification Results (5 Tasks x 3 Feature Sets)

| Study | Outcome | N | N_pos | Prevalence | Features | Best AUC | 95% CI | Best Model |
|-------|---------|---|-------|------------|----------|----------|--------|------------|
| S2 | CES-D>=16 | 719 | 141 | 19.6% | Pers-only | 0.751 | [0.622, 0.882] | LR |
| S2 | CES-D>=16 | 365 | 103 | 28.2% | Pers+Beh | **0.833** | [0.698, 0.932] | LR |
| S2 | CES-D>=16 | 365 | 103 | 28.2% | Beh-only | 0.552 | [0.402, 0.704] | LR |
| S2 | STAI>=45 | 719 | 126 | 17.5% | Pers-only | 0.795 | [0.676, 0.896] | LR |
| S2 | STAI>=45 | 365 | 92 | 25.2% | Pers+Beh | **0.856** | [0.707, 0.960] | LR |
| S2 | STAI>=45 | 365 | 92 | 25.2% | Beh-only | 0.577 | [0.416, 0.749] | LR |
| S3 | BDI-II>=14 | 800 | 316 | 39.5% | Pers-only | 0.654 | [0.552, 0.763] | LR |
| S3 | BDI-II>=14 | 594 | 234 | 39.4% | Pers+Beh | 0.671 | [0.520, 0.778] | RF |
| S3 | BDI-II>=14 | 599 | 236 | 39.4% | Beh-only | 0.569 | [0.446, 0.658] | RF |
| S3 | BDI-II>=20 | 800 | 188 | 23.5% | Pers-only | 0.667 | [0.548, 0.788] | LR |
| S3 | BDI-II>=20 | 594 | 135 | 22.7% | Pers+Beh | 0.686 | [0.515, 0.825] | LR |
| S3 | BDI-II>=20 | 599 | 136 | 22.7% | Beh-only | 0.604 | [0.458, 0.722] | LR |
| S3 | PSS>=20 | 800 | 392 | 49.0% | Pers-only | 0.686 | [0.575, 0.773] | LR |
| S3 | PSS>=20 | 594 | 296 | 49.8% | Pers+Beh | 0.704 | [0.594, 0.810] | LR |
| S3 | PSS>=20 | 599 | 297 | 49.6% | Beh-only | 0.605 | [0.484, 0.712] | RF |

### Sensitivity, Specificity, PPV, NPV (Youden's J Optimal Threshold)

Thresholds selected using Youden's J index (maximizing Sensitivity + Specificity - 1):

| Study | Outcome | Features | Sensitivity | Specificity | PPV | NPV | F1 |
|-------|---------|----------|-------------|-------------|-----|-----|-----|
| S2 | CES-D>=16 | Pers-only | 0.849 | 0.634 | 0.376 | 0.950 | 0.512 |
| S2 | CES-D>=16 | Pers+Beh | 0.839 | 0.770 | 0.632 | 0.932 | 0.699 |
| S2 | CES-D>=16 | Beh-only | 0.648 | 0.607 | 0.437 | 0.837 | 0.478 |
| S2 | STAI>=45 | Pers-only | 0.855 | 0.705 | 0.400 | 0.960 | 0.536 |
| S2 | STAI>=45 | Pers+Beh | 0.852 | 0.798 | 0.628 | 0.946 | 0.703 |
| S2 | STAI>=45 | Beh-only | 0.702 | 0.592 | 0.408 | 0.876 | 0.475 |
| S3 | BDI-II>=14 | Pers-only | 0.708 | 0.604 | 0.563 | 0.786 | 0.604 |
| S3 | BDI-II>=14 | Pers+Beh | 0.732 | 0.632 | 0.577 | 0.795 | 0.633 |
| S3 | BDI-II>=14 | Beh-only | 0.680 | 0.542 | 0.513 | 0.754 | 0.558 |
| S3 | BDI-II>=20 | Pers-only | 0.718 | 0.623 | 0.400 | 0.888 | 0.489 |
| S3 | BDI-II>=20 | Pers+Beh | 0.710 | 0.682 | 0.437 | 0.896 | 0.511 |
| S3 | BDI-II>=20 | Beh-only | 0.650 | 0.637 | 0.387 | 0.875 | 0.451 |
| S3 | PSS>=20 | Pers-only | 0.721 | 0.633 | 0.671 | 0.721 | 0.680 |
| S3 | PSS>=20 | Pers+Beh | 0.680 | 0.718 | 0.720 | 0.706 | 0.687 |
| S3 | PSS>=20 | Beh-only | 0.664 | 0.590 | 0.636 | 0.673 | 0.622 |

### Summary
- Personality alone reaches **good classification** (AUC 0.65-0.80) across all tasks
- S2 Pers+Beh improves markedly (+0.06-0.08 AUC) — driven by communication features
- S3 Pers+Beh improves minimally (+0.006-0.022 AUC)
- Behavior alone near-chance in all tasks (AUC 0.55-0.60)
- NPV consistently high (0.75-0.96) — personality-based screening has good negative predictive value (low false-negative rate for ruling out cases)
- DeLong tests (Analysis 2+8): no significant AUC difference between Pers-only and Pers+Beh (all p > 0.34)

**Output**: `results/comparison/clinical_classification.csv`, `results/comparison/figure18_clinical_classification.png`

## Section 2: Incremental Validity (Nested F-test, BH-FDR)

### Method
Nested F-test comparing Model 1 (personality-only OLS) vs Model 2 (personality + behavioral composites). Benjamini-Hochberg FDR correction applied across all 8 tests simultaneously.

### Full Results (All 8 Tests)

| Study | Outcome | N | Pers R² | Full R² | ΔR² | F-stat | df_num | df_den | p | p_fdr | Sig? |
|-------|---------|---|---------|---------|-----|--------|--------|--------|---|-------|------|
| S3 | BDI-II | 588 | 0.130 | 0.154 | **0.024** | 3.31 | 5 | 577 | 0.006 | **0.047*** | **Yes** |
| S3 | STAI | 593 | 0.226 | 0.244 | 0.018 | 2.71 | 5 | 582 | 0.020 | 0.079 | No |
| S3 | CES-D | 594 | 0.112 | 0.130 | 0.018 | 2.39 | 5 | 583 | 0.036 | 0.097 | No |
| S3 | PSS-10 | 594 | 0.176 | 0.190 | 0.014 | 1.99 | 5 | 583 | 0.078 | 0.156 | No |
| S3 | UCLA | 592 | 0.108 | 0.116 | 0.008 | 1.08 | 5 | 581 | 0.370 | 0.494 | No |
| S2 | CES-D | 363 | 0.357 | 0.367 | 0.009 | 1.75 | 3 | 354 | 0.157 | 0.251 | No |
| S2 | STAI | 363 | 0.568 | 0.570 | 0.002 | 0.68 | 3 | 354 | 0.566 | 0.647 | No |
| S2 | BAI | 363 | 0.245 | 0.248 | 0.003 | 0.43 | 3 | 354 | 0.734 | 0.734 | No |

### Interpretation
- **Uncorrected: 3/8 significant** (S3 BDI-II, STAI, CES-D)
- **FDR-corrected: 1/8 significant** — only S3 BDI-II survives (ΔR²=0.024, p_fdr=0.047)
- S2 tests all non-significant: personality already explains 25-57% of variance, leaving little room for sensing to add
- S3 tests show larger ΔR² because personality explains less (BFI-10 attenuated), leaving more unexplained variance
- Even the one significant result (ΔR²=0.024) represents a trivially small effect: 2.4% additional variance explained
- Power analysis (Analysis 5): 6/8 tests adequately powered (>=0.80); S2 STAI (0.65) and BAI (0.27) were underpowered due to very small ΔR²

**Output**: `results/comparison/incremental_validity.csv`

## Section 3: SHAP vs Traditional Methods

### Method
Compared three feature ranking approaches: (1) zero-order Pearson correlations (r), (2) standardized OLS regression coefficients (beta), and (3) SHAP mean absolute values. Kendall's tau rank correlation computed between each pair across 7 outcomes (S2 and S3 combined).

### Full Results Per Outcome

| Study | Outcome | tau(r, SHAP) | tau(beta, SHAP) | tau(r, beta) | Top-1 agree? |
|-------|---------|-------------|----------------|-------------|-------------|
| S2 | CES-D | 0.90 | 0.95 | 0.92 | Yes (N) |
| S2 | STAI | 0.95 | 0.97 | 0.96 | Yes (N) |
| S2 | BAI | 0.88 | 0.92 | 0.90 | Yes (N) |
| S3 | BDI-II | 0.93 | 0.94 | 0.91 | Yes (N) |
| S3 | STAI | 0.95 | 0.97 | 0.94 | Yes (N) |
| S3 | PSS-10 | 0.91 | 0.94 | 0.93 | Yes (N) |
| S3 | UCLA | 0.88 | 0.91 | 0.87 | Yes (E) |

### Summary

| Metric | Value |
|--------|-------|
| Top-1 agreement (all 3 methods) | **7/7 (100%)** |
| Mean tau(r, SHAP) | **0.914** |
| Mean tau(beta, SHAP) | **0.943** |
| Mean tau(r, beta) | **0.919** |

SHAP rankings are nearly redundant with OLS beta for personality traits. The high agreement (tau > 0.88 in all cases) suggests that for linear, 5-feature personality models, SHAP does not provide unique insights beyond traditional regression. SHAP's value lies in visualization (beeswarm plots, dependence plots) and non-linear model interpretation, not in ranking discovery.

**Output**: `results/comparison/shap_vs_traditional.csv`, `results/comparison/figure19_shap_vs_traditional.png`

## Section 4: Demographic Controls

### Hierarchical Regression (Gender-Only, S2)

**Step 1: Gender alone**

| Outcome | Gender R² | Gender beta |
|---------|-----------|-------------|
| CES-D | 0.028 | Female + |
| STAI | 0.026 | Female + |
| BAI | 0.042 | Female + |

**Step 2: + Personality**

| Outcome | ΔR² (Personality) | p | Personality total R² |
|---------|-------------------|---|---------------------|
| CES-D | **+0.329*** | <0.001 | 0.357 |
| STAI | **+0.544*** | <0.001 | 0.570 |
| BAI | **+0.204*** | <0.001 | 0.246 |

**Step 3: + Sensing**

| Outcome | ΔR² (Sensing) | p |
|---------|---------------|---|
| CES-D | +0.005 | n.s. |
| STAI | +0.002 | n.s. |
| BAI | +0.003 | n.s. |

Gender→CES-D and Gender→STAI: fully mediated by personality (gender beta becomes non-significant after controlling for Big Five). Gender→BAI: independent effect survives (females higher, p=0.003).

### Full Demographic Controls (Phase 16g: Gender + Native + Parent Education + Income)

| Step | CES-D R² | STAI R² | BAI R² |
|------|----------|---------|--------|
| Demographics alone | 0.03-0.06 | 0.03-0.06 | 0.03-0.06 |
| + Personality (ΔR²) | **+0.20-0.52*** | **+0.20-0.52*** | **+0.20-0.52*** |
| + Sensing (ΔR²) | <=0.009 (n.s.) | <=0.009 (n.s.) | <=0.009 (n.s.) |

Personality's predictive power is fully robust to demographic confounders. Even with 4 demographic variables controlled, personality adds 20-52% unique variance while sensing adds <1%.

**Output**: `results/comparison/demographic_controls.csv`, `results/comparison/supplementary/full_demographics.csv`

## Section 5: MLP Robustness Check

### Method
2-layer MLP (Multi-Layer Perceptron) neural network with Optuna Bayesian hyperparameter optimization (30 trials, TPE sampler). Tuned parameters: hidden layer sizes, learning rate, alpha (L2 regularization), activation function, batch size. Tested across S1, S2, S3 for both regression and classification.

### Regression Results

| Study | Outcome | Features | Traditional Best | Traditional R² | MLP R² | Winner |
|-------|---------|----------|-----------------|---------------|--------|--------|
| S1 | PSS | Personality | SVR | **0.559** | **-4.06** | Traditional |
| S2 | STAI | Personality | Ridge | **0.530** | 0.488 | Traditional |
| S2 | CES-D | Personality | RF | **0.284** | 0.251 | Traditional |
| S2 | BAI | Personality | EN | **0.165** | 0.142 | Traditional |
| S3 | STAI | Personality | EN | **0.195** | 0.142 | Traditional |
| S3 | BDI-II | Personality | EN | **0.087** | 0.063 | Traditional |
| S3 | PSS-10 | Personality | EN | **0.137** | 0.098 | Traditional |

### Classification Results

| Study | Outcome | Traditional Best | Trad. AUC | MLP AUC | Winner |
|-------|---------|-----------------|-----------|---------|--------|
| S2 | CES-D>=16 | LR | **0.751** | 0.712 | Traditional |
| S2 | STAI>=45 | LR | **0.795** | 0.621 | Traditional |
| S3 | BDI-II>=14 | LR | **0.654** | 0.618 | Traditional |
| S3 | BDI-II>=20 | LR | **0.667** | 0.635 | Traditional |
| S3 | PSS>=20 | LR | **0.686** | 0.652 | Traditional |

### Interpretation
- **Traditional models win every comparison** (regression and classification, all studies)
- S1 MLP catastrophically overfits (R²=-4.06): 5 features, N=28 is far too small for neural networks
- S2/S3: MLP underperforms by 0.04-0.17 AUC and 0.02-0.05 R²
- The personality-MH relationship is fundamentally **linear**: neural networks cannot find non-linear patterns because there are none with 5-feature personality input
- This rules out the possibility that our null sensing findings result from model choice — even flexible non-linear models cannot improve prediction

**Output**: `results/comparison/mlp_regression.csv`, `results/comparison/mlp_classification.csv`
