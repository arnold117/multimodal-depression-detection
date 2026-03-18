# Grand Synthesis

## Five Research Questions — Answers

### RQ1: Can passive sensing replace personality questionnaires for MH prediction?

**No.** Personality wins 14/15 outcome comparisons (93%) across 3 universities, 15 outcomes, 4+ ML models. Mean personality R²=0.126, mean sensing R²=-0.153. Two BFI items (10 seconds, R²=0.36) outperform 28 sensing features (weeks, R²=-0.16).

| Study | Outcomes | Personality wins |
|-------|----------|-----------------|
| S1 | 4 | 3/4 (75%) |
| S2 | 6 | 6/6 (100%) |
| S3 | 5 | 5/5 (100%) |
| **Total** | **15** | **14/15 (93%)** |

**Key figure**: `results/comparison/supplementary/figure_grand_synthesis.png`

---

### RQ2: Can sensing track real-time mood fluctuations?

**At the group level: No.** Within-person centered R² ≈ 0 for both weekly PHQ-4 (#24) and EMA (#35).

**At the individual level: Moderate, but inconsistent.** Per-person mean |r| = 0.33–0.35 between EMA mood and concurrent sensing. 17% of individuals show idiographic R² > 0.3. But the direction of the sensing-mood relationship varies across people — it's not a nomothetic signal.

**Implication**: Sensing is potentially an **idiographic monitoring tool** (personalized, requires individual calibration) rather than a **nomothetic screening tool** (one model for all).

---

### RQ3: Do findings generalize across outcome domains, demographics, and universities?

**Yes.** Personality dominates across:
- **MH domains**: depression (CES-D, BDI-II, CESD, PHQ-9), anxiety (STAI, BAI), stress (PSS), loneliness (UCLA), self-esteem
- **Academic**: GPA (C=#1 in 8/8 models)
- **Demographics**: Personality ΔR²=0.20-0.52 after controlling gender, race, native status, parent education, and family income (all p<0.001)
- **Universities**: Dartmouth (2013), Notre Dame (2015-2019), UW (2018-2021)
- **Instruments**: BFI-44 and BFI-10; PHQ-9, CES-D, BDI-II, STAI, BAI, PSS, UCLA
- **COVID**: ΔR² < 0.015 when excluding pandemic cohort

---

### RQ4: What is the minimum viable assessment?

| Approach | Time | R² (CES-D) | Cost |
|----------|------|------------|------|
| **2 BFI items** | **10 sec** | **0.358** | **$0** |
| Neuroticism score | 1 min | 0.258 | $0 |
| Full Big Five | 5 min | 0.279 | $0 |
| All 44 BFI items | 5 min | 0.328 | $0 |
| Sensing (PCA) | weeks | -0.011 | $100+ |
| Sensing (28 raw) | weeks | -0.162 | $100+ |
| Pers + Sensing | weeks | 0.321 | $100+ |

The best two items are **BFI-4** ("Is depressed, blue", r=0.605) and **BFI-24** ("Is emotionally stable, not easily upset", r=0.415). These are essentially direct mood probes embedded in a personality questionnaire.

**Key figure**: `results/comparison/supplementary/figure_cost_effectiveness.png`

---

### RQ5: Under what specific conditions does sensing have value?

| Condition | Evidence | Effect Size | Practical? |
|-----------|----------|-------------|-----------|
| **Early warning** (lagged) | Auto+Sens R²=0.613 vs Auto 0.582 | +0.031 | Marginal |
| **Communication metadata** | Pers+Comm ΔR²=+0.030 (S2 CES-D) | +0.030 | Requires call/SMS logs |
| **Sleep + nonlinear** | RF 0.316 > Ridge 0.261 (S2 CES-D) | +0.055 | S2 only |
| **Individual monitoring** | 17% idiographic R²>0.3 | Variable | Needs calibration period |
| **Engagement signal** | Not wearing ↔ anxiety (r=-0.12) | Small | Easy to implement |

**When sensing does NOT help** (33/41 analyses): raw features, feature selection, variability, inertia, ipsative, stacking, interaction, subgroup, error rescue, dose-response, within-person, prospective change, social jet lag, worst-20%, cross-study transfer...

---

## The Nuanced Conclusion

Passive sensing is not useless — it is **contextually marginal**. The field has been pursuing sensing as a **nomothetic replacement** for questionnaires (one model predicts everyone). Our evidence across 41 analyses suggests it should instead be developed as an **idiographic complement** (personalized monitoring after initial trait assessment).

The practical recommendation:
1. **Screen** with a brief questionnaire (2–5 items, 10–60 seconds)
2. **Monitor** high-risk individuals with sensing (for those where it works)
3. **Alert** based on deviation from personal baseline
4. Do NOT use sensing as a standalone population-level predictor

---

## Complete Output Inventory

### By Phase

| Phase | Scripts | CSV | PNG | Key Finding |
|-------|---------|-----|-----|-------------|
| 0–11 (S1) | 12 | 17 | 27 | C→GPA, N→MH, behavior→PHQ-9 |
| 12 (S2) | 6 | 18 | 11 | 8/13 GPA replicated, STAI R²=0.53 |
| 13 (S3) | 6 | 22 | 13 | 20/20 pers models sig, beh R²≤0 |
| 14 (Meta) | 2 | 5 | 3 | N→Anx r=0.63, N→Dep r=0.44 |
| 15 (Clinical) | 2 | 6 | 2 | AUC 0.65-0.86, FDR 1/8, MLP<Ridge |
| 16a-g (Supp) | 8 | 34 | 10 | 41 analyses, pers wins 14/15 |
| **Total** | **36** | **~102** | **~66** | |

### Key Figures for Paper

| Figure | File | Content |
|--------|------|---------|
| Grand Synthesis | `supplementary/figure_grand_synthesis.png` | Pers vs Sens across 15 outcomes |
| Cost-Effectiveness | `supplementary/figure_cost_effectiveness.png` | R² by assessment method |
| Dose-Response | `supplementary/figure_dose_response.png` | 7d=92d flat curve |
| Reliability | `supplementary/figure_reliability.png` | Sensing r=0.94-0.999 |
| Idiographic | `supplementary/figure_idiographic.png` | 17% individual R²>0.3 |
| Literature | `supplementary/figure_literature_benchmark.png` | Our AUC=0.57 = field ceiling |
| Meta Forest | `comparison/meta_analysis_forest.png` | Pooled effect sizes |
| Clinical AUC | `comparison/figure18_clinical_classification.png` | Pers vs Beh AUC |
