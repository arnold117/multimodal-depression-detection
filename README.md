# Big Five Personality, Smartphone Behavior, and Academic Performance

A three-study investigation of Big Five personality traits, passively-sensed behavioral patterns, psychological wellbeing, and academic performance (GPA) across 3 universities and 1,559 participants. Study 1 discovers effects; Study 2 validates GPA prediction; Study 3 validates mental health prediction with gold-standard instruments.

**Core findings**:
- **Conscientiousness → GPA**: Dominant predictor replicated across 2 universities, 8/8 SHAP models (100% consistency)
- **Neuroticism → Mental Health**: #1 predictor for depression, anxiety, and stress across 3 universities, 24/24 SHAP models (100% consistency)
- **Trait-specific dissociation**: C predicts academic outcomes; N predicts clinical outcomes — never reversed

## Study 1: StudentLife (Discovery)

[StudentLife](https://studentlife.cs.dartmouth.edu/) (Dartmouth, 2013): 10-week longitudinal study of 48 college students with continuous smartphone sensing, surveys, and academic records.

- **N = 28** participants with complete Big Five + GPA data
- **87 behavioral features** from 13 sensing modalities → 8 PCA composites
- **6 outcome measures**: GPA, PHQ-9, PSS, Loneliness, Flourishing, PANAS-NA
- **Big Five Inventory** (BFI-44, Cronbach's α: 0.67–0.86)

### Methods (8 Complementary Analyses)

| # | Method | Script | Research Question |
|---|---|---|---|
| 1 | Elastic Net Prediction | `scripts/elastic_net.py` | Which data source best predicts GPA? |
| 2 | Bootstrap Mediation | `scripts/mediation_analysis.py` | Does behavior mediate personality–GPA links? |
| 3 | PLS-SEM | `scripts/plssem_model.py` | Structural relationships between all constructs? |
| 4 | Latent Profile Analysis | `scripts/latent_profiles.py` | Are there distinct student behavioral types? |
| 5 | Moderation Analysis | `scripts/moderation_analysis.py` | Does personality moderate behavior effects? |
| 6 | Temporal Trends | `scripts/temporal_features.py` | Do behavioral changes predict outcomes? |
| 7 | Multi-Model Prediction | `scripts/multi_outcome_prediction.py` | 4 models × 6 outcomes × 3 feature sets |
| 8 | ML Interpretability | `scripts/ml_interpretability.py` | SHAP + cross-model feature importance |

### Key Results

| Finding | Evidence |
|---|---|
| **Conscientiousness → GPA** | RF R²=0.212 (p=0.008), EN R²=0.126 (p=0.010); SHAP #1 across all 4 models |
| **Behavior → Wellbeing (not GPA)** | Behavior predicts PHQ-9 (R²=0.468); PLS-SEM: Digital→Wellbeing β=-0.49\*, Mobility→Wellbeing β=0.38\* |
| **Non-linear Behavior→GPA** | SVR captures it (R²=0.116, p=0.032); linear models fail |
| **Personality moderates behavior** | E×Activity→GPA ΔR²=0.221, C×Activity→Loneliness ΔR²=0.213 |
| **Behavioral subgroups** | 4 LPA profiles distinguish stress (p=.024) and loneliness (p=.023) |
| **Cross-model consistency** | Kendall's τ=0.460 (4/6 pairs significant) |

## Study 2: NetHealth (External Validation)

[NetHealth](https://nethealth.nd.edu/) (Notre Dame, 2015–2019): Large-scale longitudinal study with Fitbit wearables, communication logs, surveys, and registrar GPA.

- **N = 722** total participants (220 with BFI + GPA, 498 with BFI + CES-D/STAI/BAI)
- **28 behavioral features**: Fitbit activity (12), Fitbit sleep (8), communication (8) → 3 PCA composites
- **Same instrument**: BFI-44 (Cronbach's α: 0.69–0.87)
- **Mental health**: CES-D (depression), STAI (trait anxiety), BAI (Beck anxiety)

### Replication Results

| Finding | Study 1 | Study 2 | Replicated |
|---|---|---|---|
| **C → GPA correlation** | r=+0.552 (p=.003) | r=+0.263 (p=.0001) | **Yes** (Fisher z p=.103) |
| **SHAP: C = #1 predictor** | 4/4 models | 4/4 models | **Yes** (8/8 = 100%) |
| **SVR Personality→GPA** | R²=0.059 | R²=0.027 (p=.005) | **Yes** |
| **Behavior → Depression** | R²=0.468 (PHQ-9) | R²=0.313 (CES-D) | Partial (different instruments) |
| **Personality → Anxiety** | PSS R²=0.559 | STAI R²=0.516 | **Yes** (near-identical) |
| **C × Comm → GPA moderation** | — | ΔR²=0.038 (p=.018) | New finding |

**Overall: 8/13 GPA findings replicated; all non-replications attributable to dataset differences (ceiling effect, range restriction), not contradictory evidence.**

### Anxiety Extension

| Outcome | Best Model | R² | SHAP #1 | Cross-Study Analog |
|---|---|---|---|---|
| **STAI (Trait Anxiety)** | Pers+Beh × Ridge | **0.530** | Neuroticism (4/4) | PSS R²=0.559 (replicated) |
| **BAI (Beck Anxiety)** | Pers+Beh × EN | 0.182 | Neuroticism (4/4) | PANAS-NA R²<0 (diverges) |
| CES-D (Depression) | Pers+Beh × RF | 0.313 | — | PHQ-9 R²=0.468 (partial) |

**Key insight**: Conscientiousness is #1 for GPA; Neuroticism is #1 for anxiety/depression. Personality predicts trait-level mental health (stress, anxiety, depression) but not transient affect — consistent across both studies.

### Non-Replicated Findings and Explanation

| Finding | Study 1 | Study 2 | Explanation |
|---|---|---|---|
| E → GPA | r=+0.185 (n.s.) | r=-0.078 (n.s.) | Neither significant; noise around zero |
| N → GPA | r=-0.444\* | r=-0.051 (n.s.) | GPA range restriction (SD=0.26 vs 0.39) attenuates weaker associations |
| EN/Ridge/RF R² | R²=0.13–0.21 | R²≈-0.03 | GPA ceiling effect: Notre Dame M=3.65, 75% above 3.5; ML models cannot split variance. Permutation tests still significant (EN p=.045, SVR p=.005) |

**ML Models**: Elastic Net, Ridge, Random Forest, SVR — LOO-CV (Study 1), 10×10-fold CV (Study 2), Optuna Bayesian hyperparameter optimization, and permutation tests.

## Study 3: GLOBEM (Independent Replication)

[GLOBEM](https://orsonxu.com/research/globem-dataset) (University of Washington, 2018–2021): 4 annual cohorts with smartphone sensing, Fitbit wearables, and comprehensive mental health assessments. Includes a natural COVID cohort (Spring 2020).

- **N = 809** participants across 4 cohorts (INS-W_1 through INS-W_4)
- **BFI-10** short-form personality (2 items/dimension, normalized to 1–5 scale)
- **Gold-standard instruments**: BDI-II (depression), STAI State (anxiety), PSS-10 (stress), CESD-10, UCLA Loneliness
- **Behavioral features**: Fitbit steps/sleep, phone call/screen, GPS location → 5 PCA composites
- **No GPA available** — validates mental health prediction only

### Mental Health Prediction Results

| Outcome | Best Model | R² | SHAP #1 | Cross-Study Analog |
|---|---|---|---|---|
| **STAI (State Anxiety)** | Pers × Ridge | **0.195** | Neuroticism (4/4) | STAI R²=0.530 (S2) |
| **PSS-10 (Stress)** | Pers × Ridge | **0.137** | Neuroticism (4/4) | PSS R²=0.559 (S1) |
| **CESD-10 (Depression)** | Pers × Ridge | 0.091 | Neuroticism (4/4) | CES-D R²=0.313 (S2) |
| **BDI-II (Depression)** | Pers × Ridge | 0.087 | Neuroticism (4/4) | PHQ-9 R²=0.468 (S1) |
| **UCLA Loneliness** | Pers × Ridge | 0.085 | Extraversion (3/4) | Loneliness R²=0.116 (S1) |

### Three-Study Replication Summary

| Finding | Study 1 | Study 2 | Study 3 | Status |
|---|---|---|---|---|
| N → Depression | r=+0.534 | r=+0.305 | r=+0.332 | **Consistent** |
| N → Anxiety/Stress | r=+0.757 | r=+0.432 | r=+0.446 | **Consistent** |
| SHAP: N = #1 (MH) | 4/4 models | 8/8 models | 16/16 models | **Consistent** |
| Behavior → MH (alone) | R²=0.468 | R²≤0 | R²≤0 | S1 likely overfit |
| LPA distinguishes MH | PSS p=.024 | — | UCLA p=.035 | **Consistent** |
| Pers+Beh > Pers alone | PHQ-9 yes | STAI yes | Marginal | **Partial** |

### COVID Sensitivity Analysis

Excluding the COVID cohort (INS-W_3, Spring 2020) changes all R² values by < 0.015 — results are robust to pandemic-era disruption.

## Cross-Study Meta-Analysis & Longitudinal Extensions

### Random-Effects Meta-Analysis

| Effect | k | N | Pooled r | 95% CI | I² |
|---|---|---|---|---|---|
| **N → Anxiety/Stress** | 3 | 1,324 | **0.632** | [0.384, 0.795] | 96.2% |
| **N → Depression** | 3 | 1,304 | **0.444** | [0.235, 0.614] | 91.8% |
| **C → GPA** | 2 | 248 | **0.376** | [0.064, 0.620] | 64.0% |

All pooled effects significant (p < .05). High I² reflects expected construct-level heterogeneity (different instruments across studies).

### Weekly PHQ-4 Trajectory (N=540, ~10 weeks)

- **N → PHQ-4 mean: r=+0.339\*\*\*** — high-N students have consistently higher distress
- **N → PHQ-4 slope: r=-0.012, n.s.** — personality predicts *level*, not *trajectory*
- Interpretation: Neuroticism marks stable between-person differences, consistent with trait theory

### Pre→Post Change (N=748)

All personality × Δ(Post-Pre) correlations < 0.10 (n.s.) for STAI, PSS, CESD, UCLA — personality does not predict within-semester fluctuation (supplementary material).

## Phase 15: Clinical Utility & Methodological Validation

### Section 1 — Clinical Binary Classification (10×10-fold CV)

| Study | Outcome | Pers-only AUC | Pers+Beh AUC | Beh-only AUC |
|-------|---------|--------------|-------------|-------------|
| S2 | CES-D≥16 | 0.751 | **0.833** | 0.552 |
| S2 | STAI≥45 | 0.795 | **0.856** | 0.577 |
| S3 | BDI-II≥14 | 0.654 | 0.671 | 0.569 |
| S3 | BDI-II≥20 | 0.667 | 0.686 | 0.604 |
| S3 | PSS≥20 | 0.686 | 0.704 | 0.605 |

**Key finding**: Personality alone achieves good-to-strong AUC (0.65–0.79); adding behavior improves S2 substantially (+0.06–0.08) but not S3 (+0.006–0.022). Behavior alone is near-chance (0.53–0.59), confirming personality as the primary signal.

### Section 2 — Incremental Validity (Nested F-test, BH-FDR corrected)

Does adding behavioral PCA features significantly improve personality-only prediction?

- **Uncorrected: 3/8 significant**; **FDR-corrected: 1/8** — only S3 BDI-II survives (ΔR²=0.024, p_fdr=0.047\*)
- **S2 all n.s.**: personality already explains 36–57% of variance; behavior adds nothing beyond ceiling
- **Practical interpretation**: behavioral sensing has minimal incremental validity beyond personality, primarily for depression in GLOBEM

### Section 3 — SHAP vs Traditional Methods

Kendall's τ agreement between zero-order r rankings, standardized β rankings, and SHAP |mean| rankings across 7 outcomes:

- **Top-1 agreement (all 3 methods): 7/7** (100%) — the #1 trait is always identified identically
- **Mean τ(r, SHAP) = 0.914** — near-perfect rank agreement
- **Mean τ(β, SHAP) = 0.943** — β and SHAP rankings virtually identical
- Neuroticism = #1 for all depression/anxiety/stress outcomes; Extraversion = #1 for loneliness

**Methodological implication**: SHAP provides no ranking information beyond simpler OLS β for personality traits. Its value lies in non-linear detection and visualization, not in identifying different predictors.

### Section 4 — Demographic Controls (S2 Hierarchical Regression)

Controlling for gender (S2 only; GLOBEM has no demographics):

| Outcome | R²(Gender) | R²(+Pers) | ΔR²(Pers) | p | Gender β survives? |
|---------|-----------|-----------|-----------|---|-------------------|
| CES-D | 0.030 | 0.359 | **0.329** | <0.001\*\*\* | No (p=0.28) |
| STAI | 0.024 | 0.568 | **0.544** | <0.001\*\*\* | No (p=0.96) |
| BAI | 0.061 | 0.264 | **0.204** | <0.001\*\*\* | **Yes** (p=0.003) |

**Key finding**: Personality effects are robust after controlling for gender. Gender→CES-D and gender→STAI associations are fully mediated by personality (β becomes non-significant). Gender→BAI retains an independent effect (females score higher regardless of personality).

### MLP Robustness Check

A 2-layer MLP (Optuna-tuned, 30 trials) was compared against traditional models across all 3 studies:

| Metric | Traditional Best | MLP (Optuna) | Verdict |
|--------|-----------------|-------------|---------|
| **Regression R²** (S2 STAI) | 0.530 (Ridge) | 0.488 | Traditional wins |
| **Regression R²** (S2 CES-D) | 0.313 (RF) | 0.169 | Traditional wins |
| **Regression R²** (S3 STAI) | 0.195 (Ridge) | 0.142 | Traditional wins |
| **Classification AUC** (S2 STAI≥45) | 0.795 (LR) | 0.621 | Traditional wins |
| **Classification AUC** (S2 CES-D≥16) | 0.751 (LR) | 0.653 | Traditional wins |
| **S1 Regression** (N=28) | R²=0.559 (SVR) | R²=-4.06 | Catastrophic overfit |

MLP consistently underperforms simpler models, confirming the personality–mental health relationship is predominantly linear and does not benefit from neural network flexibility. With only 5 personality features, the additional parameters of MLP are a liability rather than an asset.

## Phase 16: Supplementary Analyses (Reviewer-Proofing)

Eight supplementary analyses preemptively address reviewer concerns from clinical, UbiComp, psychometric, epidemiological, and methodological perspectives.

### Analysis 1: Raw RAPIDS Features vs PCA (S3, 1258 features)

Does PCA dimensionality reduction kill sensing signal? **No — raw features are worse.**

| Approach | BDI-II R² | STAI R² | PSS-10 R² |
|----------|-----------|---------|-----------|
| 5 PCA composites | -0.001 | -0.007 | -0.009 |
| PCA 90% variance | -0.516 | -0.534 | -0.524 |
| Raw 1258 features (Ridge) | -2.995 | -16.720 | -5.702 |
| **Personality only** | **0.095** | **0.198** | **0.154** |

More features = more overfitting. The signal is simply not in passive sensing data.

### Analysis 2+8: Expanded Clinical Metrics & Calibration

| Outcome | Features | Brier | ECE | Sens@Spec=0.80 |
|---------|----------|-------|-----|----------------|
| S2 CES-D≥16 | Pers-only | 0.142 | 0.034 | 0.487 |
| S2 STAI≥45 | Pers-only | 0.125 | 0.030 | 0.585 |
| S3 BDI-II≥14 | Pers-only | 0.224 | 0.034 | 0.363 |

DeLong tests: **no significant AUC difference** between Pers-only and Pers+Beh (all p > 0.34).

### Analysis 3: Sensing Feature Reliability (Split-Half)

All 19 sensing features show **high temporal stability** (Spearman-Brown r = 0.94–0.999). Sensing's poor prediction is a signal problem, not a measurement problem.

### Analysis 4: Modality Ablation

No single sensing modality adds meaningful variance (all ΔR² < 0.03). Sleep contributes most to STAI (+0.024), but no hidden gem modality exists.

### Analysis 5: Power Analysis

6/8 incremental validity tests adequately powered (power ≥ 0.80 at α=.05). Non-significant results are mostly true nulls, not underpowered misses.

### Analysis 6: Disattenuation Correction

After correcting for BFI-10 measurement unreliability (α≈0.65): S3 STAI R² = 0.195 → 0.333 (corrected). Even corrected, personality R² far exceeds any sensing approach.

### Analysis 7: Subgroup Analysis

Sensing R² remains ≤ 0 across all subgroups (clinical vs subclinical, high-N vs low-N). No evidence of differential sensing utility for any population.

## Project Structure

```
scripts/
  # Study 1: StudentLife pipeline
  extract_features.py              # Raw sensor data → 87 behavioral features
  score_surveys.py                 # Survey instruments → scored measures
  merge_dataset.py                 # Merge + PCA composites + interaction features
  elastic_net.py                   # Elastic Net with LOO-CV + bootstrap
  mediation_analysis.py            # Bootstrap mediation (10,000 resamples)
  plssem_model.py                  # PLS-SEM structural model
  latent_profiles.py               # Gaussian mixture LPA
  moderation_analysis.py           # Personality × behavior moderation effects
  temporal_features.py             # Weekly slope, CV, delta features
  multi_outcome_prediction.py      # 4 models × 6 outcomes × 3 feature sets
  ml_interpretability.py           # SHAP analysis + cross-model importance
  paper_materials.py               # Study 1 publication figures and report

  # Study 2: NetHealth pipeline
  nethealth_score_surveys.py       # BFI-44, CES-D, SELSA-S, GPA scoring
  nethealth_extract_features.py    # Fitbit + communication → 28 features
  nethealth_merge_dataset.py       # Merge + PCA composites
  nethealth_validation.py          # Core validation (4 models, SHAP, LPA, moderation)
  nethealth_comparison.py          # Study 1 vs Study 2 formal comparison
  nethealth_paper_materials.py     # Study 2 figures, tables, replication report

  # Study 3: GLOBEM pipeline
  globem_score_surveys.py          # BFI-10, BDI-II, STAI, PSS, CESD, UCLA scoring
  globem_extract_features.py       # Fitbit + phone + GPS → behavioral features
  globem_merge_dataset.py          # Merge 4 cohorts + PCA composites
  globem_validation.py             # MH prediction, SHAP, LPA, COVID sensitivity
  globem_comparison.py             # Three-study formal comparison
  globem_paper_materials.py        # Study 3 figures, tables, report
  globem_longitudinal.py           # Weekly trajectory + pre-post change

  # Cross-study
  meta_analysis.py                 # Random-effects meta-analysis (pooled r)
  clinical_utility.py              # Phase 15: clinical classification, incremental validity, SHAP vs traditional
  mlp_robustness.py               # MLP + Optuna robustness check vs traditional models
  supplementary_extended.py       # Phase 16: power, disattenuation, calibration, subgroup
  supplementary_core.py           # Phase 16: RAPIDS raw vs PCA, reliability, ablation
  supplementary_rapids_fast.py    # Phase 16: fast RAPIDS comparison (Ridge-only)

src/features/                     # Feature extraction modules (13 modalities)

data/
  raw/dataset/                    # StudentLife raw data (not tracked)
  raw/nethealth/                  # NetHealth raw data (not tracked)
  raw/globem/                     # GLOBEM raw data (not tracked, 4 cohorts)
  processed/
    analysis_dataset.parquet       # Study 1 final dataset (28 × 156)
    nethealth/
      nethealth_analysis_dataset.parquet  # Study 2 final dataset (722 × 47)
    globem/
      globem_analysis_dataset.parquet     # Study 3 final dataset (809 × ~50)

results/
  figures/                        # Study 1: 7 publication figures (300 dpi)
  tables/                         # Study 1: 17 CSV result tables
  reports/                        # Study 1: summary report
  nethealth/
    figures/                      # Study 2: Figures 8–12
    tables/                       # Study 2: Tables 5–9
    reports/                      # Study 2: summary + non-replication analysis
  globem/
    figures/                      # Study 3: Figures 13–15
    tables/                       # Study 3: Tables 10–12
  comparison/                     # Cross-study: forest plots, replication summaries
    supplementary/               # Phase 16: 8 supplementary analyses
```

## Reproducing Results

```bash
pip install -r requirements.txt

# Study 1: StudentLife (raw data required in data/raw/dataset/)
python scripts/extract_features.py
python scripts/score_surveys.py
python scripts/merge_dataset.py
python scripts/elastic_net.py
python scripts/mediation_analysis.py
python scripts/plssem_model.py
python scripts/latent_profiles.py
python scripts/moderation_analysis.py
python scripts/temporal_features.py
python scripts/multi_outcome_prediction.py
python scripts/ml_interpretability.py
python scripts/paper_materials.py

# Study 2: NetHealth (raw data required in data/raw/nethealth/)
python scripts/nethealth_score_surveys.py
python scripts/nethealth_extract_features.py
python scripts/nethealth_merge_dataset.py
python scripts/nethealth_validation.py         # ~5 min (joblib parallel)
python scripts/nethealth_comparison.py
python scripts/nethealth_paper_materials.py

# Study 3: GLOBEM (raw data required in data/raw/globem/)
python scripts/globem_score_surveys.py
python scripts/globem_extract_features.py
python scripts/globem_merge_dataset.py
python scripts/globem_validation.py            # ~20-30 min (5 outcomes × SHAP)
python scripts/globem_comparison.py
python scripts/globem_paper_materials.py
python scripts/globem_longitudinal.py        # Weekly trajectory + pre-post change

# Cross-study meta-analysis
python scripts/meta_analysis.py

# Phase 15: Clinical utility & methodology
python scripts/clinical_utility.py   # ~5 min (10×10-fold CV)
python scripts/mlp_robustness.py     # ~5 min (MLP + Optuna robustness)

# Phase 16: Supplementary analyses (reviewer-proofing)
python scripts/supplementary_extended.py    # ~5 min (power, disattenuation, calibration, subgroup)
python scripts/supplementary_core.py        # ~15 min (reliability, ablation; RAPIDS via slow version)
python scripts/supplementary_rapids_fast.py # ~10 min (fast RAPIDS comparison, Ridge-only)
```

## Requirements

- Python 3.10+
- See [requirements.txt](requirements.txt) for dependencies
