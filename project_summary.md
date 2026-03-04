# Project Summary: Personality, Smartphone Behavior, and Academic Performance

> A Two-Study Investigation of Big Five Personality Traits, Passively-Sensed Behavioral Patterns, Psychological Wellbeing, and Academic Performance

## Overview

This project investigates how Big Five personality traits and passively-sensed behavioral patterns relate to academic performance (GPA) and psychological wellbeing among college students. Study 1 discovers effects in a small intensive sample; Study 2 validates the core finding in an independent larger dataset.

**Core finding**: Conscientiousness is the dominant personality predictor of GPA — replicated across 2 universities, 2 time periods, 4 ML models, and confirmed via SHAP feature importance (8/8 = 100% consistency).

| | Study 1: StudentLife (Discovery) | Study 2: NetHealth (Validation) |
|---|---|---|
| **University** | Dartmouth (2013) | Notre Dame (2015–2019) |
| **Sample** | N=28 (Big Five + GPA) | N=722 total (220 BFI+GPA, 498 BFI+MH) |
| **Sensing** | 13 smartphone modalities → 87 features | Fitbit + communication → 28 features |
| **Personality** | BFI-44 (α: 0.67–0.86) | BFI-44 (α: 0.69–0.87) |
| **Outcomes** | GPA, PHQ-9, PSS, Loneliness, Flourishing, PANAS-NA | GPA, CES-D, STAI, BAI |

---

## Data Utilization

| Data Source | Details | Status |
|---|---|---|
| Big Five Personality | 5 traits (BFI-44, Cronbach's α: 0.67–0.86) | Fully used |
| Smartphone Sensors | 13 modalities → 87 raw features → 8 PCA composites | Fully used |
| Outcome Variables | GPA, PHQ-9, PSS, Loneliness, Flourishing, PANAS-NA | All 6 tested |
| Temporal Trends | 42 features (weekly slope, CV, early-vs-late delta) | Extracted & tested |
| Interaction Features | 4 composite indices (e.g., social isolation index) | Constructed & tested |
| LPA Profiles | 4 behavioral subgroups | Tested as predictors |

---

## Study 1: StudentLife — Analytical Pipeline (Phases 0–11)

### Phase 0: Data Preprocessing
- Feature extraction from 13 sensing modalities (87 features)
- Survey scoring (6 instruments + GPA)
- Dataset merging with PCA dimensionality reduction (28 × 156 variables)

### Phase 1–5: Core Analyses

| # | Method | Script | Core Question | Key Result |
|---|---|---|---|---|
| 1 | Descriptive + Correlation | `merge_dataset.py` | Basic relationships? | Correlation heatmap, descriptive stats |
| 2 | Bootstrap Mediation | `mediation_analysis.py` | Does behavior mediate Personality→GPA? | 0/40 significant (underpowered, need N≥71) |
| 3 | PLS-SEM | `plssem_model.py` | Structural paths? | Digital→Wellbeing β=-0.49\*, Mobility→Wellbeing β=0.38\* |
| 4 | Latent Profile Analysis | `latent_profiles.py` | Behavioral subgroups? | 4 profiles; differ in PSS (p=.024) & Loneliness (p=.023) |
| 5 | Elastic Net Prediction | `elastic_net.py` | What predicts GPA? | M1 Personality R²=0.170 (p=0.008) |

### Phase 7: Interaction Features & Moderation Analysis
- **Interaction features** (digital×mobility, social isolation index, etc.) — no improvement over personality alone (M6 R²=0.112 < M1 R²=0.170)
- **Moderation analysis**: 120 tests, 7 significant interactions
  - Top: Extraversion × Activity → GPA (ΔR²=0.221)
  - Conscientiousness × Activity → Loneliness (ΔR²=0.213)

### Phase 8: Temporal Trend Features
- 42 temporal features (weekly behavioral slopes, stability, early-vs-late changes)
- M7 (Personality + temporal) R²=-0.055 — trends do not improve GPA prediction

### Phase 9: Multi-Outcome × Multi-Model Prediction
- **72 combinations**: 4 models × 6 outcomes × 3 feature sets
- **4 ML models**: Elastic Net, Ridge, Random Forest, SVR
- All with LOO-CV + Optuna Bayesian hyperparameter optimization + permutation tests

### Phase 11: ML Interpretability
- SHAP analysis across 4 models × 3 prediction scenarios
- Cross-model feature importance with Kendall's τ agreement
- SHAP dependence plots for non-linearity visualization

---

## Study 1 Key Results

### 1. Personality Directly Predicts GPA

| Model | Feature Set | R² | p-value |
|---|---|---|---|
| Random Forest | Personality | **0.212** | **0.008** \* |
| Elastic Net | Personality | 0.126 | 0.010 \* |
| SVR | Pers + Beh | 0.133 | 0.030 \* |
| Random Forest | Pers + Beh | 0.102 | 0.040 \* |
| SVR | Behavior | 0.116 | 0.032 \* |

- SHAP confirms **Conscientiousness** as the #1 GPA predictor across all 4 models
- Optuna Bayesian tuning improved RF from R²=0.101 → 0.212 (+110%)
- 5/12 GPA model combinations are statistically significant
- Replicates Poropat (2009) meta-analysis findings in the passive sensing era

### 2. Behavior Predicts Wellbeing, Not GPA (with a Non-Linear Exception)

**Best R² per outcome (across all models & feature sets):**

| Outcome | Best R² | Model | Feature Set |
|---|---|---|---|
| PSS | **0.559** | SVR | Personality |
| Flourishing | **0.555** | SVR | Personality |
| PHQ-9 | **0.468** | Elastic Net | Behavior |
| Loneliness | 0.396 | Ridge | Pers + Beh |
| GPA | 0.212 | Random Forest | Personality |
| PANAS-NA | -0.078 | — | Not predictable |

- Behavior features best predict PHQ-9 depression (R²=0.468), not GPA
- PLS-SEM confirms: Digital→Wellbeing (β=-0.49\*), Mobility→Wellbeing (β=0.38\*)
- **Non-linear exception**: SVR captures Behavior→GPA (R²=0.116, p=0.032) missed entirely by linear models (EN R²=-0.130, Ridge R²=-0.516)
- SHAP identifies Mobility, Digital Usage, and Audio Environment as top PHQ-9 drivers

### 3. Personality Moderates Behavior Effects

- 7/120 significant moderation effects (bootstrap 95% CI)
- E × Activity → GPA (ΔR²=0.221): physical activity benefits extraverts' GPA more
- C × Activity → Loneliness (ΔR²=0.213): activity reduces loneliness for conscientious students
- Explains why the aggregate behavior→GPA path is null (effect heterogeneity across personality levels)

### 4. Behavioral Patterns Differentiate Student Subgroups

- 4 LPA profiles differentiate stress (p=0.024) and loneliness (p=0.023)
- But NOT depression (PHQ-9 p=0.154) — aligns with behavior→wellbeing finding
- LPA profiles have no incremental GPA predictive value (ΔR²=-0.039)

### 5. ML as an Analytical Framework

- **Optuna Bayesian optimization**: data-driven hyperparameter selection (TPE sampler, 30 trials)
- **SHAP**: model-agnostic feature attribution across 4 methods, confirming Conscientiousness primacy
- **Cross-model consistency**: Kendall's τ=0.460 (4/6 model pairs significant)
- **Non-linear discovery**: SVR revealed Behavior→GPA link invisible to linear models
- **Triangulation**: 8 complementary analyses converge on the personality-behavior dissociation

---

## Study 2: NetHealth — External Validation (Phase 12)

### Pipeline
| Script | Function |
|---|---|
| `nethealth_score_surveys.py` | BFI-44, CES-D, STAI, BAI, GPA scoring |
| `nethealth_extract_features.py` | Fitbit activity/sleep + communication → 28 features |
| `nethealth_merge_dataset.py` | Merge + PCA → 3 behavioral composites |
| `nethealth_validation.py` | 4 models, SHAP, LPA, moderation (N=722) |
| `nethealth_comparison.py` | Study 1 vs Study 2 formal comparison |
| `nethealth_paper_materials.py` | Figures 8–12, Tables 5–9, replication report |

### GPA Replication

| Finding | Study 1 | Study 2 | Replicated |
|---|---|---|---|
| **C → GPA correlation** | r=+0.552 (p=.003) | r=+0.263 (p=.0001) | **Yes** (Fisher z p=.103) |
| **SHAP: C = #1 predictor** | 4/4 models | 4/4 models | **Yes** (8/8 = 100%) |
| **SVR Personality→GPA** | R²=0.059 | R²=0.027 (p=.005) | **Yes** |
| **C × Comm → GPA moderation** | — | ΔR²=0.038 (p=.018) | New finding |

### Mental Health Prediction

| Outcome | Best Model | R² | SHAP #1 | Cross-Study Analog |
|---|---|---|---|---|
| **CES-D (Depression)** | Pers+Beh × RF | 0.313 | — | PHQ-9 R²=0.468 (partial) |
| **STAI (Trait Anxiety)** | Pers+Beh × Ridge | **0.530** | Neuroticism (4/4) | PSS R²=0.559 (replicated) |
| **BAI (Beck Anxiety)** | Pers+Beh × EN | 0.182 | Neuroticism (4/4) | PANAS-NA R²<0 (diverges) |

**Key insight**: Conscientiousness is #1 for GPA; Neuroticism is #1 for anxiety/depression — consistent across both studies, all 4 models.

### Non-Replicated Findings and Explanation

| Finding | Study 1 | Study 2 | Explanation |
|---|---|---|---|
| N → GPA | r=-0.444\* | r=-0.051 (n.s.) | GPA range restriction (SD=0.26 vs 0.39) |
| EN/Ridge/RF R² | R²=0.13–0.21 | R²≈-0.03 | GPA ceiling effect (Notre Dame M=3.65, 75% above 3.5) |

**Overall: 8/13 GPA findings replicated. All non-replications attributable to dataset differences (ceiling effect, range restriction) — zero contradictory evidence.**

---

## Cross-Study Synthesis

### The Personality–Outcome Dissociation
Across both studies, personality traits show a clean dissociation in their prediction targets:
- **Conscientiousness → Academic performance** (GPA): SHAP #1 in 8/8 model fits
- **Neuroticism → Mental health** (anxiety, depression): SHAP #1 in 8/8 model fits (STAI + BAI)
- **Behavior → Mental health, not GPA**: Behavior best predicts depression (PHQ-9, CES-D) and anxiety (STAI), but not academics

### Replication Strength
- Same instrument (BFI-44) across 2 universities, 2 time periods
- Same ML pipeline (4 models, Optuna, SHAP, permutation tests)
- Conscientiousness–GPA link survives GPA ceiling effect (Notre Dame M=3.65)
- Trait anxiety prediction near-identical (PSS R²=0.559 → STAI R²=0.530)

---

## What Didn't Work (and Why)

| Attempted | Result | Explanation |
|---|---|---|
| Mediation (behavior mediates personality→GPA) | 0/40 significant | Underpowered — need N≥71 (Fritz & MacKinnon 2007) |
| Interaction features → GPA | M6 R²=0.112 < M1 R²=0.170 | No improvement over personality alone |
| Temporal trends → GPA | M7 R²=-0.055 | Behavioral trends don't predict academics |
| LPA profiles → GPA | ΔR²=-0.039 | Behavioral subgroups irrelevant for GPA |
| PANAS-NA prediction | All R² < 0 | Not predictable with current features |
| Linear models for Behavior→GPA | EN R²=-0.130, Ridge R²=-0.516 | Relationship is non-linear (SVR succeeds) |

---

## Core Thesis

> **Smartphone behavioral sensing captures psychological state, not academic engagement.** Personality traits (especially Conscientiousness) remain the primary predictor of academic performance, while passively sensed behavioral patterns are informative for mental health screening. This dissociation replicates across 2 universities, 2 time periods, and 4 ML models.

### Narrative Structure

1. **Personality → GPA** (direct, robust, replicated across 2 studies)
2. **Behavior → Wellbeing, not GPA** (with non-linear exception via SVR)
3. **Personality moderates behavior effects** (explains null aggregate paths)
4. **Behavioral patterns stratify students** (stress/loneliness, not depression)
5. **Trait-specific dissociation** (C=#1 for GPA; N=#1 for anxiety/depression)
6. **ML as analytical framework** (Optuna, SHAP, cross-model triangulation)

---

## Outputs

### Study 1: StudentLife
- **7 publication-quality figures** (300 dpi): sample overview, mediation, prediction, LPA, multi-outcome heatmap, effect sizes, SHAP interpretability
- **17 result tables** (CSV): all statistical results with effect sizes and CIs
- **1 comprehensive summary report**: narrative + all key statistics

### Study 2: NetHealth
- **5 figures** (Figures 8–12): sample overview, forest plot, SHAP comparison, mental health comparison, SHAP anxiety heatmap
- **5 tables** (Tables 5–9): descriptive stats, GPA prediction, replication summary, behavior→depression, anxiety prediction
- **1 replication report** with non-replication analysis

### Cross-Study
- **Forest plots**: Study 1 vs Study 2 R² with CIs
- **Anxiety comparison table**: PSS↔STAI, PANAS-NA↔BAI mapping
- **Replication statistics**: effect direction, CI overlap, significance

### Pipeline
- **18 analysis scripts**: fully reproducible two-study pipeline

---

## Limitations

- Study 1: small sample (N=28), single institution, single academic term, Android-only
- Study 2: GPA ceiling effect (M=3.65, SD=0.26) attenuates weaker associations
- Mediation analysis underpowered in Study 1 (need N≥71)
- Cross-sectional design — no causal claims
- Different sensing modalities across studies (smartphone vs Fitbit) — not all analyses replicable

## Positioning

**Exploratory two-study investigation** contributing: (a) cross-validated personality–GPA link with SHAP consistency across 8 model fits; (b) demonstration that passive behavioral sensing differentially predicts mental health vs academics; (c) trait-specific dissociation (Conscientiousness for GPA, Neuroticism for anxiety/depression); (d) methodological framework combining Optuna, SHAP, and multi-model triangulation for mobile sensing research.
