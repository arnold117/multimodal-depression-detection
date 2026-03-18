# Study 3: GLOBEM — External Validation (UW, N=809)

> Phase 13 | Scripts: `globem_score_surveys.py`, `globem_extract_features.py`, `globem_merge_dataset.py`, `globem_validation.py`, `globem_comparison.py`, `globem_paper_materials.py`

## Dataset

- **N=809** (786 with BDI-II, 800 with BFI-10)
- **19 main features** (activity 4 + sleep 4 + call 4 + screen 3 + location 4) → 5 PCA composites
- **2,597 RAPIDS features** (for sensitivity analysis)
- **Outcomes**: BDI-II, STAI-State, PSS-10, CESD-10, UCLA Loneliness
- **Longitudinal**: Weekly PHQ-4 (~10 weeks), EMA (~18/person), Pre/Post surveys
- **4 cohorts**: INS-W_1 through INS-W_4 (includes COVID period INS-W_3)
- **CV strategy**: 10x10-fold repeated

## Data Preprocessing

### Feature Extraction
**19 main behavioral features** from multiple sensing sources:
- **Activity (4)**: daily steps, active minutes, sedentary minutes, step variability
- **Sleep (4)**: total sleep time, sleep efficiency, sleep onset time, wake time
- **Call (4)**: incoming call count, outgoing call count, call duration, unique contacts
- **Screen (3)**: screen-on frequency, screen-on duration, first/last unlock time
- **Location (4)**: unique locations visited, location entropy, time at home, distance traveled

**2,597 RAPIDS features** extracted for sensitivity analysis (Analysis 1):
- RAPIDS (Remote Assessment of Disease and Relapse — Smartphone) pipeline provides fine-grained behavioral features
- Filtered to 1,258 features after removing zero-variance and near-constant columns
- Tested with Ridge regression, PCA 90% variance, and raw features
- Result: raw 1,258 features R² = -1.1 to -16.7 (catastrophic overfitting); PCA did not kill the sensing signal — more features made predictions worse

### PCA Reduction
19 features → **5 orthogonal behavioral composites**: Activity, Sleep, Communication, Screen, Location

### Survey Scoring
- **BFI-10**: Big Five Inventory short form (2 items per dimension, alpha ~0.65 — lower reliability than BFI-44)
- **BDI-II**: Beck Depression Inventory II (0-63, mild >= 14, moderate >= 20)
- **STAI-State**: State-Trait Anxiety Inventory — State subscale
- **PSS-10**: Perceived Stress Scale — 10-item version (0-40, moderate >= 20)
- **CESD-10**: Center for Epidemiological Studies Depression — 10-item version
- **UCLA Loneliness**: UCLA Loneliness Scale (20-80)
- **PHQ-4**: Patient Health Questionnaire 4-item (weekly assessment, ~10 weeks)

### Cohort Structure
| Cohort | Period | N | Notes |
|--------|--------|---|-------|
| INS-W_1 | 2018-2019 | ~200 | Pre-COVID baseline |
| INS-W_2 | 2019-2020 | ~200 | Partial COVID overlap |
| INS-W_3 | 2020-2021 | ~200 | Full COVID period |
| INS-W_4 | 2021 | ~200 | Post-vaccine period |

## Key Results

### 1. Personality → Mental Health (Full Results)

**All 5 outcomes × 4 models × personality-only feature set:**

| Outcome | Elastic Net R² | Ridge R² | RF R² | SVR R² | Best R² | SHAP #1 |
|---------|---------------|----------|-------|--------|---------|---------|
| **STAI** | **0.195** | 0.189 | 0.168 | 0.183 | **0.195** | Neuroticism (4/4) |
| **PSS-10** | **0.137** | 0.131 | 0.112 | 0.125 | **0.137** | Neuroticism (4/4) |
| **CESD** | 0.085 | 0.083 | **0.091** | 0.079 | **0.091** | Neuroticism (4/4) |
| **BDI-II** | **0.087** | 0.082 | 0.074 | 0.080 | **0.087** | Neuroticism (4/4) |
| **UCLA** | **0.085** | 0.081 | 0.072 | 0.078 | **0.085** | Extraversion (3/4) |

All 20 personality models significant (p=0.005, permutation test with 199 permutations). **Neuroticism = #1 SHAP predictor for 16/16 depression/anxiety/stress models.** Extraversion dominates loneliness prediction (r=-0.229).

### 2. Behavior Alone (Full Results)

**All 5 outcomes × 4 models × behavior-only feature set:**

| Outcome | Elastic Net R² | Ridge R² | RF R² | SVR R² |
|---------|---------------|----------|-------|--------|
| STAI | -0.012 | -0.008 | -0.031 | -0.015 |
| PSS-10 | -0.009 | -0.006 | -0.028 | -0.013 |
| CESD | -0.015 | -0.011 | -0.037 | -0.018 |
| BDI-II | -0.018 | -0.014 | -0.042 | -0.021 |
| UCLA | -0.011 | -0.009 | -0.033 | -0.016 |

**R² <= 0 for ALL 20 behavior-only models.** Behavioral sensing has zero independent predictive power for mental health. This is not a model choice issue — all 4 algorithms agree.

### 3. Personality + Behavior (Incremental)

| Outcome | Pers R² | Pers+Beh R² | ΔR² | p_fdr | Significant? |
|---------|---------|-------------|-----|-------|-------------|
| **BDI-II** | 0.130 | 0.154 | **0.024** | **0.047*** | **Yes** |
| STAI | 0.226 | 0.244 | 0.018 | 0.079 | No |
| CESD | 0.112 | 0.130 | 0.018 | 0.097 | No |
| PSS-10 | 0.176 | 0.190 | 0.014 | 0.156 | No |
| UCLA | 0.108 | 0.116 | 0.008 | 0.494 | No |

FDR-corrected incremental validity: **only 1/5 significant** (BDI-II ΔR²=0.024, F=3.31, p=0.006, p_fdr=0.047). Small improvements for STAI (0.195→0.203) and BDI-II (0.087→0.107) in best-model comparisons, but sensing adds trivial variance after personality is accounted for.

### 4. Key Correlations (All 5 Traits × 5 Outcomes)

| Trait | BDI-II | STAI | PSS-10 | CESD | UCLA |
|-------|--------|------|--------|------|------|
| **Neuroticism** | **+0.305**** | **+0.432**** | **+0.371**** | **+0.297**** | **+0.171**** |
| Conscientiousness | -0.163** | -0.164** | -0.158** | -0.160** | -0.156** |
| Agreeableness | -0.092* | -0.147** | -0.099** | -0.107** | -0.185** |
| Extraversion | n.s. | -0.105** | n.s. | n.s. | **-0.229**** |
| Openness | n.s. | n.s. | n.s. | n.s. | n.s. |

Neuroticism shows the largest correlations with all internalizing outcomes. Extraversion uniquely predicts loneliness (r=-0.229). Openness has no significant associations with any outcome.

### 5. COVID Sensitivity

Excluding COVID cohort (INS-W_3) and re-running all personality models:

| Outcome | Full R² | Excl. COVID R² | ΔR² |
|---------|---------|----------------|-----|
| STAI | 0.195 | 0.190 | -0.005 |
| PSS-10 | 0.137 | 0.129 | -0.008 |
| CESD | 0.091 | 0.085 | -0.006 |
| BDI-II | 0.087 | 0.074 | -0.013 |
| UCLA | 0.085 | 0.079 | -0.006 |

All ΔR² < 0.015. Personality-MH associations are robust to pandemic effects. The COVID cohort does not drive or inflate the findings.

### 6. Latent Profile Analysis

- BIC-optimal solution: **k=6 behavioral profiles**
- ANOVA on outcomes by profile:
  - UCLA Loneliness: F(5,~800)=2.42, **p=0.035** — significant differentiation
  - BDI-II: F=1.78, p=0.116 — not significant
  - STAI: F=1.52, p=0.181 — not significant
  - PSS-10: F=1.41, p=0.219 — not significant
  - CESD: F=1.33, p=0.251 — not significant
- Consistent with S1: behavioral profiles differentiate loneliness but not depression/anxiety
- LPA adds no incremental predictive value beyond personality traits

### 7. Note on BFI-10 Measurement Reliability

BFI-10 (2 items/dimension) has lower reliability (alpha ~0.65) than BFI-44 (alpha ~0.80). This attenuation systematically reduces observed R² values.

**Disattenuation correction (Analysis 6)**:

| Outcome | Observed R² | Corrected R² | Attenuation factor |
|---------|-------------|-------------|-------------------|
| STAI | 0.195 | **0.333** | ~40% |
| PSS-10 | 0.137 | **0.234** | ~40% |
| CESD | 0.091 | **0.155** | ~40% |
| BDI-II | 0.087 | **0.149** | ~40% |
| UCLA | 0.085 | **0.145** | ~40% |

Even observed (attenuated) S3 personality R² values far exceed sensing R² values. Corrected values (0.15-0.33) approach S2 levels, suggesting instrument reliability — not effect size — explains the S2-S3 gap.

## Output Files

### Tables (20 CSV)

| File | Description |
|------|-------------|
| `results/globem/tables/table10_descriptive.csv` | Sample demographics, means, SDs for all variables |
| `results/globem/tables/table11_mh_prediction.csv` | Full MH prediction matrix (5 outcomes × 4 models × 3 feature sets) |
| `results/globem/tables/table12_replication.csv` | Three-study replication comparison |
| `results/globem/tables/personality_mental_health.csv` | Personality → MH regression results |
| `results/globem/tables/behavior_bdiii.csv` | BDI-II prediction by feature set and model |
| `results/globem/tables/behavior_stai.csv` | STAI prediction by feature set and model |
| `results/globem/tables/behavior_pss10.csv` | PSS-10 prediction by feature set and model |
| `results/globem/tables/behavior_cesd.csv` | CESD prediction by feature set and model |
| `results/globem/tables/behavior_ucla.csv` | UCLA prediction by feature set and model |
| `results/globem/tables/behavior_mental_health_all.csv` | All outcomes combined behavior results |
| `results/globem/tables/shap_personality_bdiii.csv` | SHAP importance for BDI-II models |
| `results/globem/tables/shap_personality_stai.csv` | SHAP importance for STAI models |
| `results/globem/tables/shap_personality_pss10.csv` | SHAP importance for PSS-10 models |
| `results/globem/tables/shap_personality_cesd.csv` | SHAP importance for CESD models |
| `results/globem/tables/shap_personality_ucla.csv` | SHAP importance for UCLA models |
| `results/globem/tables/lpa_outcomes.csv` | LPA profile differentiation on outcomes |
| `results/globem/tables/covid_sensitivity.csv` | Full vs excl-COVID model comparison |
| `results/globem/tables/phq4_trajectory.csv` | Weekly PHQ-4 trajectory analysis |
| `results/globem/tables/pre_post_change.csv` | Pre-post change correlations with personality |
| `results/globem/tables/descriptive_stats.csv` | Full descriptive statistics |

### Figures (11 PNG)

| File | Description |
|------|-------------|
| `results/globem/figures/figure13_sample_overview.png` | S3 sample demographics and distributions |
| `results/globem/figures/figure14_forest_mh_r2.png` | Three-study MH R² forest plot |
| `results/globem/figures/figure15_shap_heatmap.png` | SHAP consistency heatmap across 3 studies |
| `results/globem/figures/figure16_phq4_trajectory.png` | Weekly PHQ-4 trajectory by neuroticism level |
| `results/globem/figures/figure17_pre_post_change.png` | Pre-post MH change vs personality |
| `results/globem/figures/shap_personality_bdiii.png` | SHAP beeswarm for BDI-II prediction |
| `results/globem/figures/shap_personality_stai.png` | SHAP beeswarm for STAI prediction |
| `results/globem/figures/shap_personality_pss10.png` | SHAP beeswarm for PSS-10 prediction |
| `results/globem/figures/shap_personality_cesd.png` | SHAP beeswarm for CESD prediction |
| `results/globem/figures/shap_personality_ucla.png` | SHAP beeswarm for UCLA prediction |
| `results/globem/figures/correlation_heatmap.png` | Full correlation matrix |

### Reports

| File | Description |
|------|-------------|
| `results/globem/study3_report.txt` | Comprehensive S3 narrative with COVID sensitivity |
