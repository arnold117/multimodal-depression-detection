# Study 2: NetHealth — Validation (Notre Dame, N=722)

> Phase 12 | Scripts: `nethealth_score_surveys.py`, `nethealth_extract_features.py`, `nethealth_merge_dataset.py`, `nethealth_validation.py`, `nethealth_comparison.py`, `nethealth_paper_materials.py`

## Dataset

- **N=722** (220 with BFI+GPA, 498 with BFI+MH)
- **28 raw features** (Fitbit activity 11 + sleep 8 + communication 7 + quality 2) → 3 PCA composites
- **Outcomes**: CES-D, STAI-Trait, BAI, Loneliness, Self-Esteem, SELSA (romantic/family/social), GPA
- **Demographics**: Gender, native status, parent education, income, race (78 variables in BasicSurvey)
- **CV strategy**: 10×10-fold repeated stratified

## Key Results

### 1. GPA Replication

| Finding | Study 1 | Study 2 | Replicated? |
|---------|---------|---------|-------------|
| C→GPA r | +0.552 (p=.003) | +0.263 (p=.0001) | **Yes** (Fisher z p=.103) |
| SHAP C=#1 | 4/4 models | 4/4 models | **Yes** (8/8 = 100%) |
| SVR Pers→GPA | R²=0.059 | R²=0.027 (p=.005) | **Yes** |

GPA ceiling effect (M=3.65, SD=0.26) attenuates weaker associations but C→GPA survives.

### 2. Mental Health Prediction

| Outcome | Best R² | Feature Set | SHAP #1 |
|---------|---------|-------------|---------|
| **STAI (Anxiety)** | **0.530** | Pers+Beh × Ridge | Neuroticism (4/4) |
| **CES-D (Depression)** | 0.313 | Pers+Beh × RF | Neuroticism |
| **BAI (Beck Anxiety)** | 0.182 | Pers+Beh × EN | Neuroticism (4/4) |
| Self-Esteem | 0.412 | Personality × Ridge | Neuroticism (r=-0.59) |
| SELSA Social | 0.086 | Personality | Agreeableness |
| Loneliness | 0.011 | Personality | Extraversion |

**Neuroticism = #1 predictor** for all anxiety/depression outcomes. Sensing alone R² ≤ 0 for all outcomes.

### 3. Why S2 Sensing Helps (Marginally)

S2 is the **only study** where Pers+Beh notably outperforms Pers-only (AUC +0.06-0.08 in classification). Analysis 23 (deep dive) revealed:

- **Communication data (SMS/calls) is the key**: ΔR²=+0.030 for CES-D
- Activity and sleep add nothing
- S3 lacks communication data → explains S2/S3 gap

### 4. Demographic Controls (Phase 15 + Phase 16g)

With gender only (Phase 15):
- Personality ΔR² after gender: CES-D 0.329***, STAI 0.544***, BAI 0.204***
- Gender→CES-D/STAI fully mediated by personality

With full demographics (Phase 16g: gender + native + parent education + income):
- Demo R²=0.03-0.06
- **+Personality ΔR²=0.20-0.52 (all p<0.001)**
- +Sensing ΔR²≤0.009 (all n.s.)

### 5. BFI Item-Level Analysis (Phase 16c)

| Approach | CES-D R² | STAI R² | BAI R² |
|----------|----------|---------|--------|
| **2 best items** (bfi_4, bfi_24) | **0.358** | **0.507** | **0.165** |
| Neuroticism only | 0.258 | 0.477 | 0.165 |
| Full Big Five | 0.279 | 0.522 | 0.176 |
| Sensing 28 raw | -0.162 | -0.101 | -0.105 |

**BFI item 4** ("Is depressed, blue") r=0.605 and **BFI item 24** ("Is emotionally stable") r=0.415 together outperform all sensing approaches.

## Output Files

**Tables** (18 CSV): `results/nethealth/tables/` — table5-9, behavior_*, personality_gpa_validation, shap_personality_*, lpa_outcomes, moderation_results, descriptive_stats, cross_model_kendall

**Figures** (11 PNG): `results/nethealth/figures/` — figure8-12, shap_personality_*, correlation_heatmap

**Reports**: `results/nethealth/reports/study2_summary.txt`

## Replication Summary

8/13 GPA findings replicated from Study 1. All non-replications attributable to GPA ceiling effect. Zero contradictory evidence. Neuroticism→MH finding is new (S1 underpowered for MH).
