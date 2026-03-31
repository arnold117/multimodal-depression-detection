# Can Passive Sensing Replace Questionnaires for Mental Health Prediction?

A three-study head-to-head comparison (N=1,559, 3 universities, 15 outcomes) of brief personality questionnaires vs. weeks of continuous passive smartphone/wearable sensing for predicting mental health and academic performance.

## Studies

| | Study 1: StudentLife | Study 2: NetHealth | Study 3: GLOBEM |
|---|---|---|---|
| **University** | Dartmouth (2013) | Notre Dame (2015–2019) | U. Washington (2018–2021) |
| **N** | 28 | 722 | 809 |
| **Personality** | BFI-44 | BFI-44 | BFI-10 |
| **Sensing** | 13 modalities, 87 features | Fitbit + comm, 28 features | Fitbit + phone + GPS, 19 features + 2,597 RAPIDS |
| **MH Outcomes** | PHQ-9, PSS, Loneliness, Flourishing, PANAS | CES-D, STAI, BAI | BDI-II, STAI, PSS-10, CESD, UCLA |
| **Academic** | GPA | GPA | — |

## Key Results

**Questionnaires dominate at the population level.** Personality wins 14/15 outcome comparisons (93%). Two BFI items (10 seconds, R²=0.36) outperform 28 sensing features collected over weeks (R²=-0.16). Even modern deep learning (1D-CNN) and foundation models (MOMENT) cannot rescue sensing — all produce negative R². Neuroticism is the #1 mental health predictor in 28/28 SHAP models across all 3 studies; Conscientiousness is #1 for GPA in 8/8 models. Sensing features are highly reliable (ICC=0.73–0.98) but fundamentally disconnected from mental health outcomes.

**But sensing has value under specific conditions:**

| Condition | Evidence | Effect Size |
|-----------|----------|-------------|
| **Lagged early warning** | Autoregressive + sensing beats autoregressive alone | +0.031 R² |
| **Communication metadata** | SMS/call logs improve depression prediction (S2) | +0.030 R² |
| **Idiographic monitoring** | 17% of individuals show person-specific R² > 0.3 | Variable |
| **Sleep + nonlinear models** | RF captures sleep-anxiety link (S2) | +0.055 R² |
| **Engagement signal** | Device non-wear correlates with anxiety | r = -0.12 |
| **Clinical classification** | Pers+Beh AUC improves over Pers-only (S2) | +0.06–0.08 AUC |

**Practical implication**: Screen with a brief questionnaire (2–5 items), then deploy sensing only for high-risk individuals where personalized monitoring adds value.

## Project Structure

```
scripts/
  00_shared/                         # Shared utilities
    score_surveys.py                 # Survey scoring functions
    multi_outcome_prediction.py      # 4 models x outcomes x feature sets
    ml_interpretability.py           # SHAP analysis + cross-model importance

  01_data_preparation/               # Feature extraction + dataset merging
    s1_extract_features.py           # StudentLife: raw sensor -> 87 features
    s1_merge_dataset.py              # Merge + PCA composites
    s1_temporal_features.py          # Weekly slope, CV, delta features
    s2_score_surveys.py              # NetHealth: BFI-44, CES-D, STAI, GPA
    s2_extract_features.py           # Fitbit + communication -> 28 features
    s2_merge_dataset.py
    s3_score_surveys.py              # GLOBEM: BFI-10, BDI-II, STAI, PSS, UCLA
    s3_extract_features.py           # Fitbit + phone + GPS -> features
    s3_merge_dataset.py              # Merge 4 cohorts + PCA composites

  02_core_analyses/                  # Main analyses
    s2_validation.py                 # Study 2 replication (4 models, SHAP, LPA)
    s2_comparison.py                 # Study 1 vs 2 formal comparison
    s3_validation.py                 # Study 3 MH prediction + SHAP + COVID
    s3_comparison.py                 # Three-study comparison
    s3_longitudinal.py               # Weekly trajectory + pre-post change
    meta_analysis.py                 # Random-effects meta-analysis
    clinical_utility.py              # Classification AUC, incremental validity
    mlp_robustness.py                # MLP + Optuna vs traditional models

  03_robustness/                     # 44 supplementary robustness checks
    robustness_01_core.py            # Reliability, ablation, RAPIDS
    robustness_02_extended.py        # Power, disattenuation, calibration
    robustness_03_sensing_tests.py   # Reverse prediction, residuals, stacking
    robustness_04_temporal.py        # Dose-response, prospective, within-person
    robustness_05_pro_sensing.py     # Idiographic, interaction, ipsative
    robustness_06_frontier.py        # S2 deep dive, weekly, nonlinear
    robustness_07_missingness.py     # Missing data as signal
    robustness_08_synthesis.py       # Cost-effectiveness, literature benchmark
    robustness_09_fdr_correction.py  # FDR correction across all tests
    robustness_10_rapids_fast.py     # Fast RAPIDS comparison (Ridge-only)
    robustness_11_deep_learning.py   # 1D-CNN, MOMENT foundation model, GradientBoosting+Optuna
    robustness_12_nns_comparison.py  # NNS practical significance from classification results
    robustness_13_temporal_reliability.py  # ICC decay curves across time windows

  04_supplementary/                  # Study 1 supplementary analyses
    s1_elastic_net.py                # Elastic Net with LOO-CV
    s1_mediation.py                  # Bootstrap mediation
    s1_plssem.py                     # PLS-SEM structural model
    s1_latent_profiles.py            # Gaussian mixture LPA
    s1_moderation.py                 # Personality x behavior moderation

  05_paper_materials/                # Publication figures + tables
    s1_paper_materials.py
    s2_paper_materials.py
    s3_paper_materials.py

src/features/                        # Feature extraction modules (13 modalities)

data/
  raw/                               # Raw data (not tracked)
    dataset/                         # StudentLife
    nethealth/                       # NetHealth
    globem/                          # GLOBEM (4 cohorts)
  processed/
    analysis_dataset.parquet         # Study 1 final dataset
    nethealth/                       # Study 2 final dataset
    globem/                          # Study 3 final dataset

results/
  core/                              # Main findings (CSV + figures)
  robustness/                        # 41 robustness check outputs
  supplementary/                     # Additional analyses
  by_study/                          # Per-study results (s1, s2, s3)
  tables/                            # Publication-ready tables

report/                              # Detailed analysis reports (7 chapters)
paper/                               # LaTeX manuscript
docs/                                # Dataset evaluation inventories
```

## Reproducing Results

```bash
pip install -r requirements.txt

# Study 1: StudentLife (raw data in data/raw/dataset/)
python scripts/01_data_preparation/s1_extract_features.py
python scripts/01_data_preparation/s1_merge_dataset.py
python scripts/01_data_preparation/s1_temporal_features.py

# Study 2: NetHealth (raw data in data/raw/nethealth/)
python scripts/01_data_preparation/s2_score_surveys.py
python scripts/01_data_preparation/s2_extract_features.py
python scripts/01_data_preparation/s2_merge_dataset.py
python scripts/02_core_analyses/s2_validation.py
python scripts/02_core_analyses/s2_comparison.py

# Study 3: GLOBEM (raw data in data/raw/globem/)
python scripts/01_data_preparation/s3_score_surveys.py
python scripts/01_data_preparation/s3_extract_features.py
python scripts/01_data_preparation/s3_merge_dataset.py
python scripts/02_core_analyses/s3_validation.py
python scripts/02_core_analyses/s3_comparison.py
python scripts/02_core_analyses/s3_longitudinal.py

# Cross-study
python scripts/02_core_analyses/meta_analysis.py
python scripts/02_core_analyses/clinical_utility.py
python scripts/02_core_analyses/mlp_robustness.py

# Robustness (44 analyses)
python scripts/03_robustness/robustness_01_core.py
python scripts/03_robustness/robustness_02_extended.py
# ... through robustness_10_rapids_fast.py
python scripts/03_robustness/robustness_11_deep_learning.py   # requires torch, momentfm
python scripts/03_robustness/robustness_12_nns_comparison.py
python scripts/03_robustness/robustness_13_temporal_reliability.py
```

## Requirements

- Python 3.11+
- PyTorch 2.11+ with MPS support (for robustness_11)
- momentfm (for MOMENT foundation model baseline)
- See [requirements.txt](requirements.txt) for dependencies
