# Study 1: StudentLife — Discovery (Dartmouth, N=28)

> Phase 0–11 | Scripts: `extract_features.py`, `score_surveys.py`, `merge_dataset.py`, `elastic_net.py`, `mediation_analysis.py`, `plssem_model.py`, `latent_profiles.py`, `moderation_analysis.py`, `temporal_features.py`, `multi_outcome_prediction.py`, `ml_interpretability.py`, `paper_materials.py`

## Dataset

- **N=28** (with complete BFI-44 + GPA; 46 total participants, 28 with full data)
- **87 behavioral features** from 13 smartphone sensing modalities → 8 PCA composites
- **Outcomes**: GPA, PHQ-9, PSS, Loneliness, Flourishing, PANAS-NA
- **CV strategy**: Leave-one-out (small N)

## Key Results

### 1. Personality Directly Predicts GPA (Phase 5, 9)

| Model | Feature Set | R² | p |
|-------|------------|-----|---|
| Random Forest | Personality | **0.212** | 0.008* |
| Elastic Net | Personality | 0.126 | 0.010* |
| SVR | Pers + Beh | 0.133 | 0.030* |

- SHAP confirms **Conscientiousness = #1 GPA predictor** across all 4 models
- Optuna Bayesian tuning improved RF from R²=0.101 → 0.212 (+110%)

### 2. Behavior Predicts Wellbeing, Not GPA (Phase 9)

| Outcome | Best R² | Model | Feature Set |
|---------|---------|-------|-------------|
| PSS | **0.559** | SVR | Personality |
| Flourishing | **0.555** | SVR | Personality |
| PHQ-9 | **0.468** | Elastic Net | Behavior |
| Loneliness | 0.396 | Ridge | Pers + Beh |
| GPA | 0.212 | RF | Personality |

- Behavior features best predict PHQ-9 (R²=0.468) — but likely overfitting at N=28
- PLS-SEM: Digital→Wellbeing β=-0.49*, Mobility→Wellbeing β=0.38*

### 3. Mediation: Underpowered (Phase 2)

- 0/40 mediation paths significant (need N≥71 per Fritz & MacKinnon 2007)

### 4. Moderation: 7/120 Significant (Phase 7)

- E × Activity → GPA (ΔR²=0.221): activity benefits extraverts' GPA more
- C × Activity → Loneliness (ΔR²=0.213): activity reduces loneliness for conscientious students

### 5. Temporal Features: No Value (Phase 8)

- 42 temporal features (weekly slopes, stability) → M7 R²=-0.055 for GPA

### 6. Latent Profiles (Phase 4)

- 4 LPA profiles differentiate stress (p=0.024) and loneliness (p=0.023)
- But NOT depression (p=0.154)

### 7. ML Interpretability (Phase 11)

- SHAP cross-model consistency: Kendall's τ=0.460 (4/6 pairs significant)
- Conscientiousness = #1 for GPA in 4/4 models (100%)

## Output Files

**Tables** (17 CSV): `results/tables/` — descriptive_stats, elastic_net_*, mediation_*, plssem_*, lpa_*, moderation_results, multi_outcome_matrix, multi_model_comparison, shap_importance, cross_model_importance

**Figures** (27 PNG): `results/figures/` — figure1–7 (publication), correlation_heatmap, elastic_net_*, lpa_*, mediation_*, plssem_*, shap_*, moderation_*

**Reports**: `results/reports/summary_report.txt`

## Limitations

- N=28: underpowered for mediation, moderation, and complex models
- Single institution, single term, Android-only
- PHQ-9 R²=0.468 by behavior likely overfitting (confirmed in Phase 16g: S1 is only case where sensing > personality)
