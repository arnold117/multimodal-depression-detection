# Big Five Personality, Smartphone Behavior, and Academic Performance

Investigating the relationships between Big Five personality traits, passively-sensed smartphone behavioral patterns, psychological wellbeing, and academic performance (GPA) using the StudentLife dataset.

**Core finding**: Personality (especially Conscientiousness) predicts GPA; smartphone behavior predicts psychological wellbeing (PHQ-9 R²=0.468) but not GPA — except through non-linear models (SVR R²=0.116, p=0.032).

## Dataset

[StudentLife](https://studentlife.cs.dartmouth.edu/) (Dartmouth, 2013): 10-week longitudinal study of 48 college students with continuous smartphone sensing, surveys, and academic records.

- **N = 28** participants with complete Big Five + GPA data
- **87 behavioral features** from 13 sensing modalities → 8 PCA composites
- **6 outcome measures**: GPA, PHQ-9, PSS, Loneliness, Flourishing, PANAS-NA
- **Big Five Inventory** (BFI-44, Cronbach's α: 0.67–0.86)

## Methods (8 Complementary Analyses)

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

**ML Models**: Elastic Net, Ridge, Random Forest, SVR — all with LOO-CV, Optuna Bayesian hyperparameter optimization, and permutation tests.

## Key Results

| Finding | Evidence |
|---|---|
| **Conscientiousness → GPA** | RF R²=0.212 (p=0.008), EN R²=0.126 (p=0.010); SHAP-confirmed #1 across all 4 models |
| **Behavior → Wellbeing (not GPA)** | Behavior predicts PHQ-9 (R²=0.468); PLS-SEM: Digital→Wellbeing β=-0.49\*, Mobility→Wellbeing β=0.38\* |
| **Non-linear Behavior→GPA** | SVR captures it (R²=0.116, p=0.032); linear models fail (EN R²=-0.130, Ridge R²=-0.516) |
| **Personality moderates behavior** | 7/120 significant; E×Activity→GPA ΔR²=0.221, C×Activity→Loneliness ΔR²=0.213 |
| **Behavioral subgroups** | 4 LPA profiles distinguish stress (p=.024) and loneliness (p=.023), not depression |
| **Cross-model consistency** | Feature ranking agreement: Kendall's τ=0.460 (4/6 pairs significant) |

## Project Structure

```
scripts/                          # Executable pipeline scripts
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
  paper_materials.py               # Publication-ready figures and report

src/features/                     # Feature extraction modules (13 modalities)
  temporal.py                      # Temporal alignment and user timelines
  gps.py                           # GPS mobility (14 features)
  app.py                           # App usage patterns (11 features)
  communication.py                 # Call + SMS (14 features)
  activity.py                      # Physical activity (11 features)
  phonelock.py                     # Screen time and unlock patterns (6 features)
  bluetooth.py                     # Social proximity via Bluetooth (4 features)
  conversation.py                  # Face-to-face interactions (5 features)
  education.py                     # Piazza engagement + deadlines (10 features)
  ema.py                           # Ecological momentary assessment (6 features)
  wifi.py                          # WiFi location proxy (3 features)
  audio.py                         # Audio environment classification (3 features)

data/
  raw/dataset/                    # StudentLife raw data (not tracked)
  processed/
    features/                      # Per-modality and combined feature parquets
    scores/                        # Survey scores and GPA
    analysis_dataset.parquet       # Final merged dataset (28 × 156)

results/
  figures/                        # 7 publication figures + analysis plots (300 dpi)
  tables/                         # 17 CSV result tables
  reports/                        # Summary report with narrative
```

## Reproducing Results

```bash
pip install -r requirements.txt

# Full pipeline (raw data required in data/raw/dataset/)
python scripts/extract_features.py        # ~5 min
python scripts/score_surveys.py           # ~5 sec
python scripts/merge_dataset.py           # ~5 sec

# Analyses (can run independently after pipeline)
python scripts/elastic_net.py             # ~10 min (bootstrap + permutation)
python scripts/mediation_analysis.py      # ~3 min (10,000 bootstrap)
python scripts/plssem_model.py            # ~5 min (5,000 bootstrap)
python scripts/latent_profiles.py         # ~1 min
python scripts/moderation_analysis.py     # ~2 min (bootstrap CI)
python scripts/temporal_features.py       # ~1 min
python scripts/multi_outcome_prediction.py # ~10 min (Optuna + permutation)
python scripts/ml_interpretability.py     # ~5 min (SHAP + Optuna)

# Publication materials
python scripts/paper_materials.py         # ~10 sec
```

## Requirements

- Python 3.10+
- See [requirements.txt](requirements.txt) for dependencies
