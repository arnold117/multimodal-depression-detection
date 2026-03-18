# Study 1: StudentLife — Discovery (Dartmouth, N=28)

> Phase 0–11 | Scripts: `extract_features.py`, `score_surveys.py`, `merge_dataset.py`, `elastic_net.py`, `mediation_analysis.py`, `plssem_model.py`, `latent_profiles.py`, `moderation_analysis.py`, `temporal_features.py`, `multi_outcome_prediction.py`, `ml_interpretability.py`, `paper_materials.py`

## Dataset

- **N=28** (with complete BFI-44 + GPA; 46 total participants, 28 with full data)
- **87 behavioral features** from 13 smartphone sensing modalities → 8 PCA composites
- **Outcomes**: GPA, PHQ-9, PSS, Loneliness, Flourishing, PANAS-NA
- **CV strategy**: Leave-one-out (small N)

## Phase 0: Data Preprocessing

### Feature Extraction (Phase 0a)
Extracted **87 raw behavioral features** from 13 smartphone sensing modalities:
- **Activity** (accelerometer-derived): stationary %, walking %, running %, unknown %
- **Audio** (microphone classifier): silence %, voice %, noise %, unique audio environments
- **Conversation**: frequency, duration, speaker turns
- **Location** (GPS): unique locations, location entropy, time at home, distance traveled
- **Screen**: unlock frequency, screen-on duration, screen-off duration
- **Bluetooth**: unique devices detected (social proxy)
- **WiFi**: unique APs (location proxy)
- **Communication** (call/SMS): call count, SMS count, call duration
- **App usage**: total, social, entertainment
- **Light sensor**: mean lux, variability
- Additional modalities: charging, phone lock

### Survey Scoring (Phase 0b)
Scored 6 validated instruments:
- **BFI-44**: Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism), internal reliability alpha = 0.67-0.86
- **PHQ-9**: Depression severity (0-27 scale)
- **PSS**: Perceived Stress Scale (0-40 scale)
- **UCLA Loneliness**: (20-80 scale)
- **Flourishing Scale**: (8-56 scale)
- **PANAS-NA**: Negative Affect (10-50 scale)
- **GPA**: Cumulative academic performance

### Dataset Merging & PCA (Phase 0c)
- Merged personality, behavioral, and outcome data: 28 participants x 156 variables
- PCA dimensionality reduction: 87 behavioral features → **8 orthogonal composites** (capturing Mobility, Digital Usage, Audio Environment, Social Activity, Communication, Sleep, Screen Habits, App Usage)
- Rationale: N=28 cannot support 87 features; PCA avoids multicollinearity and reduces overfitting risk

## Key Results

### 1. Personality Directly Predicts GPA (Phase 5, 9)

**Phase 5 (Elastic Net baseline)**: M1 Personality R²=0.170 (p=0.008). Conscientiousness coefficient = largest magnitude.

**Phase 9 (Multi-Model Matrix)**: All 4 models tested with Optuna Bayesian hyperparameter optimization (TPE sampler, 30 trials) and LOO-CV with permutation tests (199 permutations).

| Model | Feature Set | R² | p |
|-------|------------|-----|---|
| Random Forest | Personality | **0.212** | 0.008* |
| Elastic Net | Personality | 0.126 | 0.010* |
| SVR | Pers + Beh | 0.133 | 0.030* |
| Random Forest | Pers + Beh | 0.102 | 0.040* |
| SVR | Behavior | 0.116 | 0.032* |
| Ridge | Personality | 0.078 | 0.064 |
| Ridge | Pers + Beh | 0.009 | 0.310 |
| Elastic Net | Behavior | -0.130 | 0.734 |
| Ridge | Behavior | -0.516 | 0.998 |

- **5/12 GPA model combinations are statistically significant** (p<0.05)
- SHAP confirms **Conscientiousness = #1 GPA predictor** across all 4 models (100% consistency)
- Optuna Bayesian tuning improved RF from R²=0.101 → 0.212 (+110%)
- Replicates Poropat (2009) meta-analysis findings in the passive sensing era
- **Non-linear exception**: SVR captures Behavior→GPA (R²=0.116, p=0.032) missed entirely by linear models (EN R²=-0.130, Ridge R²=-0.516) — suggesting non-linear behavior-performance relationships

### 2. Behavior Predicts Wellbeing, Not GPA (Phase 9)

**Full Multi-Outcome Prediction Matrix: 72 combinations (4 models x 6 outcomes x 3 feature sets)**

Best R² per outcome (across all models and feature sets):

| Outcome | Best R² | Model | Feature Set | p |
|---------|---------|-------|-------------|---|
| PSS | **0.559** | SVR | Personality | <0.01 |
| Flourishing | **0.555** | SVR | Personality | <0.01 |
| PHQ-9 | **0.468** | Elastic Net | Behavior | <0.01 |
| Loneliness | 0.396 | Ridge | Pers + Beh | <0.01 |
| GPA | 0.212 | RF | Personality | 0.008 |
| PANAS-NA | -0.078 | — | Not predictable | n.s. |

Feature set comparison (averaged across models):

| Feature Set | GPA R² | PHQ-9 R² | PSS R² | Loneliness R² | Flourishing R² |
|-------------|--------|----------|--------|---------------|---------------|
| Personality | **0.08-0.21** | 0.07-0.18 | **0.22-0.56** | 0.09-0.30 | **0.19-0.56** |
| Behavior | -0.52 to 0.12 | **0.15-0.47** | -0.05 to 0.26 | -0.19 to 0.21 | -0.23 to 0.22 |
| Pers + Beh | 0.01-0.13 | 0.04-0.40 | 0.13-0.53 | **0.18-0.40** | 0.16-0.49 |

- Behavior features best predict PHQ-9 depression (R²=0.468), not GPA — but likely overfitting at N=28
- PLS-SEM (Phase 3): Digital→Wellbeing β=-0.49*, Mobility→Wellbeing β=0.38*
- SHAP identifies Mobility, Digital Usage, and Audio Environment as top PHQ-9 drivers
- PANAS-NA not predictable by any feature set (all R² < 0)

### 3. Mediation: Underpowered (Phase 2)

- **0/40 mediation paths significant** after bootstrap resampling (10,000 iterations)
- Tested: 5 personality traits × 8 behavioral composites → GPA
- Fritz & MacKinnon (2007): need N>=71 for medium-effect mediation; our N=28 has ~15% power
- Indirect effects near zero for all paths; direct personality→GPA path remains significant
- Conclusion: cannot evaluate mediation at this sample size, not evidence against mediation

### 4. Moderation: 7/120 Significant (Phase 7)

Tested 120 moderation effects (5 traits × 8 behaviors × 3 outcomes) using bootstrap 95% CIs:

| Interaction | Outcome | ΔR² | Interpretation |
|-------------|---------|-----|----------------|
| **E × Activity → GPA** | GPA | **0.221** | Physical activity benefits extraverts' GPA more |
| **C × Activity → Loneliness** | Loneliness | **0.213** | Activity reduces loneliness for conscientious students |
| A × Digital → GPA | GPA | 0.158 | Agreeable students less harmed by screen time |
| N × Mobility → Loneliness | Loneliness | 0.147 | Mobility reduces loneliness less for neurotic students |
| O × Audio → PSS | PSS | 0.133 | Complex audio environments stress less open students |

- 7/120 = 5.8% significant (at alpha=0.05, expect 6 by chance alone — interpret cautiously)
- Interaction features as predictors (M6): R²=0.112 < M1 R²=0.170 (no improvement over personality alone)
- Key insight: explains why the aggregate behavior→GPA path is null — effect heterogeneity across personality levels

### 5. Temporal Features: No Value (Phase 8)

- Extracted **42 temporal features**: weekly behavioral slopes (linear trends), stability (coefficient of variation), early-vs-late semester changes, acceleration/deceleration patterns
- M7 (Personality + temporal) R²=-0.055 for GPA — worse than personality alone
- Behavioral trends over the semester do not predict academic outcomes
- Interpretation: GPA prediction depends on stable trait-like patterns, not intra-semester dynamics

### 6. Latent Profiles (Phase 4)

- Gaussian Mixture Model (BIC-optimal): **4 behavioral profiles** identified
- Profile selection by BIC: k=2 (BIC=347.2), k=3 (340.1), **k=4 (335.8)**, k=5 (339.4)
- ANOVA on outcomes by profile:
  - Stress (PSS): F(3,24)=3.67, **p=0.024** — profiles differentiate stress levels
  - Loneliness: F(3,24)=3.52, **p=0.023** — profiles differentiate loneliness
  - Depression (PHQ-9): F(3,24)=1.87, p=0.154 — NOT significant
  - GPA: F(3,24)=0.92, p=0.447 — profiles have no incremental GPA value (ΔR²=-0.039)
- Aligns with behavior→wellbeing (not behavior→GPA) pattern

### 7. ML Interpretability (Phase 11)

- SHAP analysis across 4 models (EN, Ridge, RF, SVR) × 3 prediction scenarios (GPA from personality, wellbeing from behavior, GPA from combined)
- **Cross-model consistency**: Kendall's τ=0.460 (mean across 4/6 pairs significant at p<0.05)
- **Conscientiousness = #1 SHAP predictor for GPA** in 4/4 models (100% agreement)
- SHAP dependence plots reveal:
  - C→GPA: monotonically positive, roughly linear
  - N→PSS: monotonically positive with slight acceleration at high N
  - Digital Usage→PHQ-9: positive (more screen time → more depression), non-linear threshold
- Cross-model importance correlation: RF and SVR show highest agreement (τ=0.71), EN and Ridge near-identical (τ=0.94)

### 8. PLS-SEM Structural Model (Phase 3)

- Partial Least Squares Structural Equation Modeling: Personality → Behavior → Wellbeing → GPA
- Key structural paths:
  - Digital Usage → Wellbeing: beta = -0.49* (more screen time predicts worse wellbeing)
  - Mobility → Wellbeing: beta = +0.38* (more physical mobility predicts better wellbeing)
  - Personality → Wellbeing: beta = +0.42* (direct effect)
  - Wellbeing → GPA: beta = +0.31 (n.s. — indirect path weak)

## Output Files

### Tables (17 CSV)

| File | Description |
|------|-------------|
| `results/tables/descriptive_stats.csv` | Sample demographics and variable distributions |
| `results/tables/elastic_net_coefficients.csv` | EN model coefficients for GPA prediction |
| `results/tables/elastic_net_comparison.csv` | Model comparison across feature sets |
| `results/tables/mediation_simple.csv` | Simple mediation results (5 traits × 8 behaviors) |
| `results/tables/mediation_parallel.csv` | Parallel mediation with multiple mediators |
| `results/tables/mediation_composite.csv` | Composite behavioral mediator results |
| `results/tables/plssem_results.csv` | PLS-SEM path coefficients and significance |
| `results/tables/plssem_effects.csv` | Direct, indirect, and total effects |
| `results/tables/lpa_fit_indices.csv` | BIC/AIC for k=2 through k=6 profiles |
| `results/tables/lpa_profiles.csv` | Profile centroids on behavioral composites |
| `results/tables/lpa_outcome_comparison.csv` | ANOVA results: profiles vs outcomes |
| `results/tables/moderation_results.csv` | All 120 moderation tests with effect sizes |
| `results/tables/multi_outcome_matrix.csv` | Full 72-combination R² matrix (4 models × 6 outcomes × 3 feature sets) |
| `results/tables/multi_model_comparison.csv` | Model comparison summary across outcomes |
| `results/tables/effect_size_summary.csv` | Cohen's d and R² effect sizes across analyses |
| `results/tables/shap_importance.csv` | SHAP mean absolute values per feature per model |
| `results/tables/cross_model_importance.csv` | Kendall tau agreement across model pairs |

### Figures (26 PNG)

| File | Description |
|------|-------------|
| `results/figures/figure1_sample_overview.png` | Sample demographics and variable distributions |
| `results/figures/figure2_mediation_summary.png` | Mediation path diagram with effect sizes |
| `results/figures/figure3_prediction.png` | GPA prediction: predicted vs actual by model |
| `results/figures/figure4_lpa.png` | LPA profile radar chart and outcome comparison |
| `results/figures/figure5_multi_outcome.png` | Multi-outcome R² heatmap (models × outcomes × feature sets) |
| `results/figures/figure6_effect_sizes.png` | Forest plot of effect sizes across analyses |
| `results/figures/figure7_shap.png` | SHAP summary and dependence plots |
| `results/figures/correlation_heatmap.png` | Personality × behavior × outcome correlation matrix |
| `results/figures/elastic_net_coefficients.png` | EN coefficient bar chart |
| `results/figures/elastic_net_feature_importance.png` | Feature importance ranking |
| `results/figures/elastic_net_model_comparison.png` | M1-M5 model comparison |
| `results/figures/elastic_net_predicted_vs_actual.png` | Predicted vs actual scatter |
| `results/figures/lpa_bic.png` | BIC elbow plot for profile selection |
| `results/figures/lpa_outcomes.png` | Outcome means by LPA profile |
| `results/figures/lpa_radar.png` | Behavioral composites radar by profile |
| `results/figures/mediation_forest_plot.png` | Indirect effects with 95% CIs |
| `results/figures/mediation_path_diagram.png` | Path diagram with coefficients |
| `results/figures/moderation_simple_slopes.png` | Simple slopes for significant moderations |
| `results/figures/multi_model_comparison.png` | R² by model type bar chart |
| `results/figures/multi_outcome_heatmap.png` | 72-cell R² heatmap |
| `results/figures/plssem_path_diagram.png` | SEM path diagram with loadings |
| `results/figures/plssem_r_squared.png` | Variance explained per endogenous variable |
| `results/figures/shap_dependence.png` | SHAP dependence plots for top features |
| `results/figures/shap_summary.png` | SHAP beeswarm summary |
| `results/figures/cross_model_importance.png` | Cross-model feature importance comparison |
| `results/figures/effect_size_forest.png` | Forest plot of effect sizes |

### Reports

| File | Description |
|------|-------------|
| `results/reports/summary_report.txt` | Comprehensive narrative summary with all key statistics |

## Limitations

- **N=28**: underpowered for mediation (need N>=71), moderation (7/120 sig ~ chance rate), and complex models
- **Single institution**: Dartmouth College, single academic term (Fall 2013)
- **Android-only**: excludes iOS users, potential selection bias
- **PHQ-9 R²=0.468** by behavior likely overfitting — confirmed in Phase 16g: S1 is the only study where sensing outperforms personality, and this finding does not replicate in S2 or S3
- **LOO-CV**: necessary for N=28 but provides less stable variance estimates than k-fold
- **No demographic controls**: small N precludes adding covariates
