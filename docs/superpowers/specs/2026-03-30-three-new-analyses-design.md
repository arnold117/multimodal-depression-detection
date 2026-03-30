# Design: Three New Robustness Analyses (#42-44)

**Date**: 2026-03-30
**Status**: Approved

## Context

The project has 41 supplementary analyses (robustness_01-10) showing personality wins 14/15 comparisons. Three new analyses strengthen the argument by addressing likely reviewer concerns.

## Analysis 42: Deep Learning Baseline (`robustness_11_deep_learning.py`)

### Purpose
Show that even modern deep learning approaches cannot rescue passive sensing's poor predictive validity for mental health outcomes.

### Models

**1D-CNN**:
- Input: per-person daily time series matrix (days × features)
- Architecture: Conv1D(in=n_features, out=64, kernel=3) → ReLU → Conv1D(64, 32, 3) → ReLU → GlobalAvgPool → Dense(32, 1)
- Training: 5-fold CV matching existing Ridge splits, Adam optimizer, MSE loss, early stopping on val loss
- Device: MPS (Apple Silicon GPU)

**MOMENT** (CMU time-series foundation model):
- Use pretrained MOMENT encoder to extract fixed-length embeddings from daily sensing time series
- Feed embeddings into Ridge regression (no fine-tuning of MOMENT)
- This tests whether pretrained temporal representations capture mental health-relevant patterns

### Data
- **S3 (GLOBEM)**: Daily features from `/data/raw/globem/INS-W_[1-4]/FeatureData/` — steps, sleep, call, screen, location. ~90 days per person, N≈500-700 with sufficient daily data.
- **S2 (NetHealth)**: Daily Fitbit from `/data/raw/nethealth/FitbitActivity*.csv`, `FitbitSleep*.csv`. ~180 days per person, N≈45-50.

### Comparison Table
All models predict same outcomes (BDI-II, CES-D, STAI, etc.) using same CV splits:

| Model | Features | Type |
|-------|----------|------|
| Ridge(Personality) | BFI traits | Baseline (existing) |
| Ridge(Sensing PCA) | PCA composites | Existing |
| GradientBoosting(Sensing) + Optuna | Raw sensing features | Traditional ML upper bound |
| 1D-CNN(Daily sensing) | Daily time series | Deep learning |
| MOMENT → Ridge | Pretrained embeddings | Foundation model |

### Outputs
- `results/robustness/deep_learning_comparison.csv`: Study, Outcome, Model, N, R2_mean, R2_std, R2_CI_lo, R2_CI_hi
- `results/robustness/figure_deep_learning.png`: Grouped bar chart R² by model × outcome

### Key Design Decisions
- No fine-tuning of MOMENT (zero-shot embedding) — avoids overfitting on small N, and strengthens the argument: even a foundation model's representations don't help
- GradientBoosting + Optuna as "optimized traditional ML" to close the gap between Ridge and deep learning
- Same CV folds across all models for fair comparison
- Handle missing daily data via forward-fill + zero-padding to fixed length (90 days for S3, 180 for S2)

---

## Analysis 43: NNS Practical Significance (`robustness_12_nns_comparison.py`)

### Purpose
Translate statistical metrics (AUC, sensitivity, specificity) into clinically interpretable numbers: "per 100 people screened, how many more true cases does personality catch vs sensing?"

### Method
1. Load existing `results/core/clinical_classification.csv` (already has AUC, Sensitivity, Specificity, PPV, NPV per Study × Outcome × Feature set)
2. Compute NNS (Number Needed to Screen):
   - NNS = 1 / (Sensitivity × Prevalence + (1 - Specificity) × (1 - Prevalence)) ... simplified: NNS = 1 / PPV
   - Or more intuitively: per 100 screened, TP = Sensitivity × N_pos, FP = (1-Specificity) × N_neg
3. Head-to-head comparison table: Personality vs Sensing vs Combined
4. Compute "extra cases caught per 100 screened" = TP_personality - TP_sensing

### Outputs
- `results/robustness/nns_comparison.csv`: Study, Outcome, Features, N, Prevalence, NNS, TP_per_100, FP_per_100, Net_benefit
- `results/robustness/figure_nns.png`: Side-by-side bar chart showing TP per 100 screened for each feature set

### Key Design Decisions
- Reuse existing classification results — no re-running models
- Frame as "practical screening utility" not just statistical significance
- Include net benefit (TP - wFP) with w=1 (equal cost of false positive and false negative) as sensitivity analysis

---

## Analysis 44: Sensing Temporal Reliability Decay (`robustness_13_temporal_reliability.py`)

### Purpose
Show that sensing features have poor temporal stability (low ICC across time windows), explaining why they fail as predictors. Compare to personality's known test-retest reliability (~0.80-0.85 from literature).

### Method
1. Load S3 daily sensing data (steps, sleep, call, screen, location)
2. Split each person's data into non-overlapping time windows: 7, 14, 30, 60 days
3. For each window size, compute ICC(3,k) across windows per feature using pingouin
4. Plot reliability decay curve: x = window size, y = ICC, one line per modality
5. Add horizontal reference line for personality test-retest (BFI-44 ≈ 0.85, BFI-10 ≈ 0.75)

### Data
- **S3**: `/data/raw/globem/INS-W_[1-4]/FeatureData/` — daily data, ~90 days per person
- Window splits: 7d (up to 12 windows), 14d (up to 6), 30d (up to 3), 60d (up to 1 pair)
- Minimum 2 windows required per person for ICC calculation

### Outputs
- `results/robustness/temporal_reliability.csv`: Feature, Modality, Window_days, ICC, ICC_CI_lo, ICC_CI_hi, N_persons, N_windows
- `results/robustness/figure_temporal_reliability.png`: Line plot of ICC vs window size per modality, with personality reference band

### Key Design Decisions
- Use ICC(3,k) (two-way mixed, consistency) — appropriate for comparing same features across time
- Non-overlapping windows to avoid inflating reliability estimates
- Include only persons with ≥2 complete windows per window size
- Literature values for personality reliability: BFI-44 retest r ≈ 0.85 (Rammstedt & John, 2007), BFI-10 retest r ≈ 0.75 (Rammstedt & John, 2007)

---

## Shared Conventions

- All scripts follow existing `robustness_*.py` patterns: shebang, docstring, `OUT = Path("results/robustness")`, `RS = 42`, `TRAITS` list
- Data loading: `pd.read_parquet("data/processed/...")` for processed, raw CSV for daily data
- CV: RepeatedKFold(n_splits=5, n_repeats=5) with StandardScaler inside fold
- Output: CSV + PNG (dpi=300, bbox_inches="tight")
- Print progress with `flush=True`

## Dependencies

- Existing: numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, seaborn, pingouin, optuna
- New: torch (MPS), momentfm (MOMENT foundation model)
- Environment: `/Users/arnold/miniforge3/envs/qbio/bin/python`

## File Map

| Script | Output CSV | Output Figure |
|--------|-----------|---------------|
| `scripts/03_robustness/robustness_11_deep_learning.py` | `deep_learning_comparison.csv` | `figure_deep_learning.png` |
| `scripts/03_robustness/robustness_12_nns_comparison.py` | `nns_comparison.csv` | `figure_nns.png` |
| `scripts/03_robustness/robustness_13_temporal_reliability.py` | `temporal_reliability.csv` | `figure_temporal_reliability.png` |
