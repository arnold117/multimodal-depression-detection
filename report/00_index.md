# Complete Analysis Report: Personality vs Passive Sensing for Mental Health Prediction

> **Three-Study Investigation (N=1,559) with 41 Supplementary Robustness Checks**
>
> Arnold | NUS Department of Psychological Medicine | Supervisor: Prof. Cyrus Ho

---

## Report Structure

| File | Content | Phase |
|------|---------|-------|
| [01_study1_studentlife.md](01_study1_studentlife.md) | Study 1: Discovery (Dartmouth, N=28) | Phase 0–11 |
| [02_study2_nethealth.md](02_study2_nethealth.md) | Study 2: Validation (Notre Dame, N=722) | Phase 12 |
| [03_study3_globem.md](03_study3_globem.md) | Study 3: External Validation (UW, N=809) | Phase 13 |
| [04_cross_study.md](04_cross_study.md) | Cross-Study Synthesis: Meta-Analysis + Longitudinal | Phase 14 |
| [05_clinical_utility.md](05_clinical_utility.md) | Clinical Utility + Methodology | Phase 15 |
| [06_supplementary.md](06_supplementary.md) | 41 Supplementary Analyses (complete) | Phase 16a–g |
| [07_grand_synthesis.md](07_grand_synthesis.md) | Grand Synthesis + Research Questions | All |

---

## The One-Paragraph Summary

Across 3 universities (Dartmouth, Notre Dame, UW), 1,559 participants, 15 mental health and academic outcomes, and 41 supplementary analyses testing every conceivable methodological variation, **a 5-minute personality questionnaire — or even 2 items answered in 10 seconds (R²=0.36) — consistently and dramatically outperforms weeks of continuous passive smartphone/wearable sensing (R²=-0.16) for mental health prediction.** Personality wins 14/15 outcome comparisons (93%). Sensing shows marginal value only under specific conditions: lagged early-warning prediction (+0.03 R²), communication metadata (+0.03), and for ~17% of individuals whose mood trajectories happen to correlate with behavioral patterns (per-person |r|=0.35). The field should pivot from "sensing replaces questionnaires" to "sensing complements questionnaires for personalized monitoring."

---

## Data Sources

| | Study 1: StudentLife | Study 2: NetHealth | Study 3: GLOBEM |
|---|---|---|---|
| **University** | Dartmouth (2013) | Notre Dame (2015–2019) | U. Washington (2018–2021) |
| **N** | 28 | 722 | 809 |
| **Personality** | BFI-44 (α: 0.67–0.86) | BFI-44 (α: 0.69–0.87) | BFI-10 (short form) |
| **Sensing** | 13 modalities → 87 features | Fitbit + comm → 28 features | Fitbit + phone + GPS → 19 features + 2,597 RAPIDS |
| **MH Outcomes** | PHQ-9, PSS, Loneliness, Flourishing, PANAS | CES-D, STAI, BAI, Loneliness, Self-Esteem, SELSA | BDI-II, STAI, PSS-10, CESD, UCLA |
| **Academic** | GPA | GPA | — |
| **Longitudinal** | 10 weeks | Multi-semester | Weekly PHQ-4 (~10 wk), EMA (~18/person), Pre/Post |
| **Demographics** | — | Gender, SES, race, parent education (78 vars) | Cohort only |

**Total: N=1,559, 15 unique outcomes, 34 analysis scripts, 236 output files**

---

## Pipeline Overview (34 Scripts)

```
Phase 0–11:  Study 1 discovery (8 analyses: EN, mediation, PLS-SEM, LPA, moderation, temporal, multi-model, SHAP)
Phase 12:    Study 2 replication (GPA + MH validation, SHAP, LPA, moderation)
Phase 13:    Study 3 external validation (5 MH outcomes, SHAP, LPA, COVID sensitivity)
Phase 14:    Meta-analysis (random-effects, pooled r) + Longitudinal (weekly trajectory, pre/post)
Phase 15:    Clinical utility (classification AUC, incremental validity, SHAP vs β, demographic controls, MLP robustness)
Phase 16a-g: 41 supplementary analyses (feature engineering, measurement, temporal, person-level, pro-sensing, complete data)
```
