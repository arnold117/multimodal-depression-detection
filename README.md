# Big Five Personality, Smartphone Behavior, and Academic Performance

Investigating the relationships between Big Five personality traits, passively-sensed smartphone behavioral patterns, psychological wellbeing, and academic performance (GPA) using the StudentLife dataset.

## Dataset

[StudentLife](https://studentlife.cs.dartmouth.edu/) (Dartmouth, 2013): 10-week longitudinal study of 48 college students with continuous smartphone sensing, surveys, and academic records.

- **N = 28** participants with complete Big Five + GPA data
- **87 behavioral features** from 13 sensing modalities
- **6 psychological measures** (PHQ-9, PSS, Loneliness, Flourishing, PANAS)
- **Big Five Inventory** (BFI-44)

## Methods

| Method | Script | Research Question |
|---|---|---|
| Elastic Net | `scripts/elastic_net.py` | Which data source best predicts GPA? |
| Bootstrap Mediation | `scripts/mediation_analysis.py` | Does behavior mediate personality-GPA links? |
| PLS-SEM | `scripts/plssem_model.py` | Structural relationships between all constructs? |
| Latent Profile Analysis | `scripts/latent_profiles.py` | Are there distinct personality-behavior student types? |

## Analysis Results

| Method | Key Finding |
|---|---|
| Elastic Net | Personality alone best predicts GPA (LOO-CV R²=0.170, p=0.008). Conscientiousness (95% selection) and Neuroticism (82% selection) are key |
| Bootstrap Mediation | 40 paths tested, no significant indirect effects (expected at n=27). Strongest: C → Proximity → GPA (ab=0.117) |
| PLS-SEM | R² GPA=0.577, R² Wellbeing=0.799. Significant: Digital → Wellbeing (β=-0.49\*) and Mobility → Wellbeing (β=0.38\*) |
| Latent Profiles | 4 profiles identified. Significant differences in PSS stress (p=0.024) and loneliness (p=0.023) |

### Key Findings

- **Conscientiousness** is the strongest predictor of GPA (LOO-CV R² = 0.170, p = 0.008)
- Personality alone outperforms smartphone behavior and wellbeing measures for GPA prediction
- Digital engagement and physical mobility significantly relate to psychological wellbeing
- Four distinct personality-behavior profiles identified, differing in stress and loneliness

### Interpretation

1. **Personality → GPA is the only reliable prediction path.** Elastic Net LOO-CV R²=0.170 (p=0.008), driven by Conscientiousness (95% selected) and Neuroticism (82% selected). This aligns with decades of educational psychology research (Poropat, 2009) — validates the pipeline but is not a novel finding on its own.

2. **Phone usage hurts wellbeing; mobility improves it.** PLS-SEM: Digital → Wellbeing (β=-0.49\*) and Mobility → Wellbeing (β=0.38\*) are the only two significant structural paths. The wellbeing model explains substantial variance (R²=0.799), while many behavior → GPA paths are unstable at this sample size.

3. **Mediation effects are all non-significant — a power issue, not a pipeline issue.** At N=27, statistical power for mediation is far below the minimum (~71 for medium effects; Fritz & MacKinnon, 2007). The strongest path (C → Proximity → GPA, ab=0.117) is directionally plausible but cannot reach significance. Report this as a limitation.

4. **LPA distinguishes stress and loneliness, not depression.** PSS (p=0.024) and loneliness (p=0.023) differ across profiles, but PHQ-9 does not (p=0.154). This suggests behavioral patterns more directly reflect stress/loneliness than depression severity.

### Potential Improvements

**Worth pursuing:**
- Use continuous PHQ-9 scores instead of binary cutoff — preserves information at small N
- Feature interactions (e.g., high digital use x low mobility as a social isolation indicator), supported by PLS-SEM findings
- Weekly-level temporal features (slope, change points) instead of semester-wide aggregates, to capture behavioral deterioration trajectories
- Use LPA profile membership as a predictor variable in downstream models

**Not recommended:**
- Complex models (deep learning, Transformer) — N=27 guarantees overfitting
- Increasing bootstrap iterations for mediation — the bottleneck is sample size, not resampling count

## Project Structure

```
scripts/                    # Executable pipeline scripts
  extract_features.py       # Step 1: Raw sensor data → 87 behavioral features
  score_surveys.py          # Step 2: Survey instruments → scored measures
  merge_dataset.py          # Step 3: Merge into analysis dataset + PCA composites
  mediation_analysis.py     # Analysis: Bootstrap mediation (10,000 resamples)
  plssem_model.py           # Analysis: PLS-SEM structural model
  latent_profiles.py        # Analysis: Gaussian mixture LPA
  elastic_net.py            # Analysis: Elastic Net with LOO-CV
  paper_materials.py        # Generate publication-ready figures and report

src/features/               # Feature extraction modules (13 modalities)
  temporal.py               # Temporal alignment and user timelines
  gps.py                    # GPS mobility (14 features)
  app.py                    # App usage patterns (11 features)
  communication.py          # Call + SMS (14 features)
  activity.py               # Physical activity (11 features)
  phonelock.py              # Screen time and unlock patterns (6 features)
  bluetooth.py              # Social proximity via Bluetooth (4 features)
  conversation.py           # Face-to-face interactions (5 features)
  education.py              # Piazza engagement + deadlines (10 features)
  ema.py                    # Ecological momentary assessment (6 features)
  wifi.py                   # WiFi location proxy (3 features)
  audio.py                  # Audio environment classification (3 features)

data/
  raw/dataset/              # StudentLife raw data (not tracked)
  processed/                # Generated outputs
    features/               # Per-modality and combined feature parquets
    scores/                 # Survey scores and GPA
    analysis_dataset.parquet  # Final merged dataset (28 × 110)

results/
  figures/                  # All generated figures (300 dpi)
  tables/                   # CSV result tables
  reports/                  # Summary report
```

## Reproducing Results

```bash
pip install -r requirements.txt

# Full pipeline (raw data required in data/raw/dataset/)
python scripts/extract_features.py    # ~5 min
python scripts/score_surveys.py       # ~5 sec
python scripts/merge_dataset.py       # ~5 sec

# Analyses (can run independently after pipeline)
python scripts/mediation_analysis.py  # ~3 min (10,000 bootstrap)
python scripts/plssem_model.py        # ~5 min (5,000 bootstrap)
python scripts/latent_profiles.py     # ~1 min
python scripts/elastic_net.py         # ~10 min (bootstrap + permutation)
python scripts/paper_materials.py     # ~10 sec
```

## Requirements

- Python 3.10+
- See [requirements.txt](requirements.txt) for dependencies
