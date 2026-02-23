# Big Five Personality, Smartphone Behavior, and Academic Performance

A two-study investigation of Big Five personality traits, passively-sensed behavioral patterns, psychological wellbeing, and academic performance (GPA). Study 1 discovers effects in a small intensive sample; Study 2 validates the core finding in an independent larger dataset.

**Core finding**: Conscientiousness is the dominant personality predictor of GPA — replicated across 2 universities, 2 time periods, 4 ML models, and confirmed via SHAP feature importance (8/8 = 100% consistency).

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

- **N = 722** total participants (220 with BFI + GPA, 498 with BFI + CES-D)
- **28 behavioral features**: Fitbit activity (12), Fitbit sleep (8), communication (8) → 3 PCA composites
- **Same instrument**: BFI-44 (Cronbach's α: 0.69–0.87)
- **Depression**: CES-D (vs PHQ-9 in Study 1)

### Replication Results

| Finding | Study 1 | Study 2 | Replicated |
|---|---|---|---|
| **C → GPA correlation** | r=+0.552 (p=.003) | r=+0.263 (p=.0001) | **Yes** (Fisher z p=.103) |
| **SHAP: C = #1 predictor** | 4/4 models | 4/4 models | **Yes** (8/8 = 100%) |
| **SVR Personality→GPA** | R²=0.059 | R²=0.027 (p=.005) | **Yes** |
| **Behavior → Depression** | R²=0.468 (PHQ-9) | R²=0.313 (CES-D) | Partial (different instruments) |
| **C × Comm → GPA moderation** | — | ΔR²=0.038 (p=.018) | New finding |

**Overall: 8/13 findings replicated (62%)**

### Non-Replicated Findings and Explanation

| Finding | Study 1 | Study 2 | Explanation |
|---|---|---|---|
| E → GPA | r=+0.185 (n.s.) | r=-0.078 (n.s.) | Neither significant; noise around zero |
| N → GPA | r=-0.444\* | r=-0.051 (n.s.) | GPA range restriction (SD=0.26 vs 0.39) attenuates weaker associations |
| EN/Ridge/RF R² | R²=0.13–0.21 | R²≈-0.03 | GPA ceiling effect: Notre Dame M=3.65, 75% above 3.5; ML models cannot split variance. Permutation tests still significant (EN p=.045, SVR p=.005) |

**ML Models**: Elastic Net, Ridge, Random Forest, SVR — LOO-CV (Study 1), 10×10-fold CV (Study 2), Optuna Bayesian hyperparameter optimization, and permutation tests.

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

src/features/                     # Feature extraction modules (13 modalities)

data/
  raw/dataset/                    # StudentLife raw data (not tracked)
  raw/nethealth/                  # NetHealth raw data (not tracked)
  processed/
    analysis_dataset.parquet       # Study 1 final dataset (28 × 156)
    nethealth/
      nethealth_analysis_dataset.parquet  # Study 2 final dataset (722 × 47)

results/
  figures/                        # Study 1: 7 publication figures (300 dpi)
  tables/                         # Study 1: 17 CSV result tables
  reports/                        # Study 1: summary report
  nethealth/
    figures/                      # Study 2: Figures 8–10
    tables/                       # Study 2: Tables 5–8
    reports/                      # Study 2: summary + non-replication analysis
  comparison/                     # Cross-study: forest plots, replication summary
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
```

## Requirements

- Python 3.10+
- See [requirements.txt](requirements.txt) for dependencies
