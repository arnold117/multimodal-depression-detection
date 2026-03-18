# Clinical Utility & Methodological Validation

> Phase 15 | Scripts: `clinical_utility.py`, `mlp_robustness.py`

## Section 1: Clinical Binary Classification

10×10 Repeated Stratified K-Fold, Logistic Regression vs Random Forest, StandardScaler inside fold.

| Study | Outcome | Pers-only AUC | Pers+Beh AUC | Beh-only AUC |
|-------|---------|---------------|-------------|-------------|
| S2 | CES-D≥16 | 0.751 | **0.833** | 0.552 |
| S2 | STAI≥45 | 0.795 | **0.856** | 0.577 |
| S3 | BDI-II≥14 | 0.654 | 0.671 | 0.569 |
| S3 | BDI-II≥20 | 0.667 | 0.686 | 0.604 |
| S3 | PSS≥20 | 0.686 | 0.704 | 0.605 |

Personality alone reaches good classification (AUC 0.65–0.80). S2 Pers+Beh improves markedly (+0.06–0.08) but S3 minimally. Behavior alone near-chance.

**Output**: `results/comparison/clinical_classification.csv`, `results/comparison/figure18_clinical_classification.png`

## Section 2: Incremental Validity (Nested F-test, BH-FDR)

| Study | Outcome | ΔR² | p | p_fdr | Significant? |
|-------|---------|-----|---|-------|-------------|
| S3 | BDI-II | 0.024 | 0.006 | **0.047*** | Yes |
| S3 | STAI | 0.018 | 0.020 | 0.079 | No |
| S3 | CES-D | 0.018 | 0.037 | 0.097 | No |
| S2 | CES-D | 0.009 | 0.157 | 0.251 | No |
| S2 | STAI | 0.002 | 0.566 | 0.647 | No |
| S2 | BAI | 0.003 | 0.734 | 0.734 | No |

**FDR-corrected: 1/8 significant** (S3 BDI-II only, ΔR²=0.024).

**Output**: `results/comparison/incremental_validity.csv`

## Section 3: SHAP vs Traditional Methods

| Metric | Value |
|--------|-------|
| Top-1 agreement (all 3 methods) | **7/7 (100%)** |
| Mean τ(r, SHAP) | 0.914 |
| Mean τ(β, SHAP) | 0.943 |

SHAP rankings are redundant with OLS β for personality traits.

**Output**: `results/comparison/shap_vs_traditional.csv`, `results/comparison/figure19_shap_vs_traditional.png`

## Section 4: Demographic Controls (Gender, S2)

Personality ΔR² after gender: CES-D 0.329***, STAI 0.544***, BAI 0.204***. Gender→CES-D/STAI fully mediated by personality.

**Output**: `results/comparison/demographic_controls.csv`

## MLP Robustness Check

2-layer MLP with Optuna Bayesian optimization (30 trials):

| Metric | Traditional Best | MLP (Optuna) | Winner |
|--------|-----------------|-------------|--------|
| Regression R² (S2 STAI) | 0.530 (Ridge) | 0.488 | Traditional |
| Regression R² (S3 STAI) | 0.195 (EN) | 0.142 | Traditional |
| Classification AUC (S2 STAI≥45) | 0.795 (LR) | 0.621 | Traditional |
| S1 Regression (N=28) | 0.559 (SVR) | -4.06 | Traditional (catastrophic MLP overfit) |

Personality–MH relationship is linear; neural networks add no value with 5-feature input.

**Output**: `results/comparison/mlp_regression.csv`, `results/comparison/mlp_classification.csv`
