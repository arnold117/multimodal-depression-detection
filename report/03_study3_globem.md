# Study 3: GLOBEM — External Validation (UW, N=809)

> Phase 13 | Scripts: `globem_score_surveys.py`, `globem_extract_features.py`, `globem_merge_dataset.py`, `globem_validation.py`, `globem_comparison.py`, `globem_paper_materials.py`

## Dataset

- **N=809** (786 with BDI-II, 800 with BFI-10)
- **19 main features** (activity 4 + sleep 4 + call 4 + screen 3 + location 4) → 5 PCA composites
- **2,597 RAPIDS features** (for sensitivity analysis)
- **Outcomes**: BDI-II, STAI-State, PSS-10, CESD-10, UCLA Loneliness
- **Longitudinal**: Weekly PHQ-4 (~10 weeks), EMA (~18/person), Pre/Post surveys
- **4 cohorts**: INS-W_1 through INS-W_4 (includes COVID period INS-W_3)
- **CV strategy**: 10×10-fold repeated

## Key Results

### 1. Personality → Mental Health

| Outcome | Best R² | Model | p | SHAP #1 |
|---------|---------|-------|---|---------|
| **STAI** | **0.195** | Elastic Net | 0.005 | Neuroticism (4/4) |
| PSS-10 | 0.137 | Elastic Net | 0.005 | Neuroticism (4/4) |
| CESD | 0.091 | RF | 0.005 | Neuroticism (4/4) |
| BDI-II | 0.087 | Elastic Net | 0.005 | Neuroticism (4/4) |
| UCLA | 0.085 | Elastic Net | 0.005 | Extraversion (3/4) |

All 20 personality models significant (p=0.005). **Neuroticism = #1 SHAP predictor for 16/16 depression/anxiety/stress models.**

### 2. Behavior Alone

**R² ≤ 0 for ALL outcomes.** Behavior has zero independent predictive power.

### 3. Personality + Behavior

Small improvements: STAI 0.195→0.203, BDI-II 0.087→0.107. But FDR-corrected incremental validity: only 1/5 significant (BDI-II ΔR²=0.024, p_fdr=0.047).

### 4. Key Correlations

| Trait | BDI-II | STAI | PSS-10 | CESD | UCLA |
|-------|--------|------|--------|------|------|
| **Neuroticism** | **+0.305** | **+0.432** | **+0.371** | **+0.297** | **+0.171** |
| Conscientiousness | -0.163 | -0.164 | -0.158 | -0.160 | -0.156 |
| Extraversion | n.s. | -0.105 | n.s. | n.s. | **-0.229** |

### 5. COVID Sensitivity

Excluding COVID cohort (INS-W_3): all ΔR² < 0.015. Results robust to pandemic effects.

### 6. LPA

k=6 profiles: UCLA Loneliness significant (F=2.42, p=0.035), other outcomes n.s.

### 7. Note on BFI-10

BFI-10 (2 items/dimension) has lower reliability (α≈0.65) than BFI-44 (α≈0.80). Disattenuation correction (Analysis 6) shows S3 personality R² would be 0.15-0.33 with perfect reliability — still far exceeding sensing.

## Output Files

**Tables** (22 CSV): `results/globem/tables/` — table10-12, personality_mental_health, behavior_*, shap_personality_*, covid_sensitivity, lpa_outcomes, phq4_trajectory, pre_post_change, descriptive_stats

**Figures** (13 PNG): `results/globem/figures/` — figure13-17, shap_personality_*, correlation_heatmap

**Reports**: `results/globem/study3_report.txt`
