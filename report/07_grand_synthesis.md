# Grand Synthesis

## Five Research Questions — Answers

### RQ1: Can passive sensing replace personality questionnaires for MH prediction?

**No.** Personality wins 14/15 outcome comparisons (93%) across 3 universities, 15 outcomes, 4+ ML models. Mean personality R²=0.126, mean sensing R²=-0.153. Two BFI items (10 seconds, R²=0.36) outperform 28 sensing features (weeks, R²=-0.16). This result holds even with modern deep learning (1D-CNN: R²=-0.03 to -0.10) and foundation models (MOMENT: R²=-1.0 to -1.7). The problem is signal, not model.

| Study | Outcomes Tested | Personality Wins | Details |
|-------|----------------|-----------------|---------|
| S1 | 4 (GPA, PHQ-9, PSS, Loneliness) | 3/4 (75%) | PHQ-9 is the one exception (R²=0.468 behavior, likely overfit at N=28) |
| S2 | 6 (CES-D, STAI, BAI, SE, SELSA, Loneliness) | 6/6 (100%) | Sensing R² <= 0 for all outcomes |
| S3 | 5 (BDI-II, STAI, PSS-10, CESD, UCLA) | 5/5 (100%) | Sensing R² <= 0 for all outcomes |
| **Total** | **15** | **14/15 (93%)** | The one "win" for sensing (S1 PHQ-9) does not replicate |

The cost-effectiveness comparison is stark:
- **2 BFI items**: 10 seconds, $0, R²=0.358 for CES-D
- **Neuroticism subscale**: 1 minute, $0, R²=0.258 for CES-D
- **Full BFI-44**: 5 minutes, $0, R²=0.279 for CES-D
- **28 sensing features**: weeks of data collection, $100+ device cost, R²=-0.162 for CES-D
- **2,597 RAPIDS features**: weeks of data, R²=-1.1 to -16.7 (catastrophic overfitting)
- **1D-CNN on daily time series**: R²=-0.03 to -0.10 (deep learning cannot rescue sensing)
- **MOMENT foundation model embeddings**: R²=-1.0 to -1.7 (pretrained representations also fail)

Critically, sensing features are not unreliable — ICC(3,k) ranges from 0.73 to 0.98 across time windows (Analysis 44). Sensing captures stable, reliable behavioral patterns that are simply disconnected from mental health outcomes. The problem is construct relevance, not measurement quality.

**Key figures**: `results/core/figure_grand_synthesis.png`, `results/robustness/figure_deep_learning.png`

---

### RQ2: Can sensing track real-time mood fluctuations?

**At the group level: No.** Within-person centered R² approximately 0 for both weekly PHQ-4 (Analysis 24: pooled within-person R²=-0.003, 3,149 person-weeks) and EMA (Analysis 35: within-person R² approximately 0, ~18 assessments/person). Behavioral variability features also fail (Analysis 12: all R² approximately 0). Temporal autocorrelation/inertia features similarly null (Analysis 17: R² <= 0).

**At the individual level: Moderate, but inconsistent.** Per-person mean |r| = 0.33-0.35 between EMA mood and concurrent sensing. 23% of individuals show idiographic R² > 0, and 17% show idiographic R² > 0.3 (Analysis 20). However:
- The direction of the sensing-mood relationship varies across individuals — it is not a nomothetic signal
- High PHQ-4 variability does not predict who benefits from sensing
- Cannot identify a priori who will show idiographic signal

**Implication**: Sensing is potentially an **idiographic monitoring tool** (personalized, requires individual calibration period) rather than a **nomothetic screening tool** (one model fits all). This fundamentally changes the field's value proposition for passive sensing.

---

### RQ3: Do findings generalize across outcome domains, demographics, and universities?

**Yes.** Personality dominates across every dimension tested:
- **MH domains**: depression (CES-D R²=0.28, BDI-II R²=0.09, CESD R²=0.09, PHQ-9 R²=0.07), anxiety (STAI R²=0.53/0.20, BAI R²=0.18), stress (PSS R²=0.56/0.14), loneliness (UCLA R²=0.09), self-esteem (R²=0.41)
- **Academic**: GPA (C=#1 SHAP in 8/8 models across S1 and S2; meta-analytic r=0.376)
- **Demographics**: Personality ΔR²=0.20-0.52 after controlling gender, race, native status, parent education, and family income (all p<0.001). Demographics alone explain only R²=0.03-0.06
- **Universities**: Dartmouth (2013), Notre Dame (2015-2019), UW (2018-2021) — three distinct institutions, regions, and time periods
- **Instruments**: Both BFI-44 (alpha 0.67-0.87) and BFI-10 (alpha ~0.65); 7 MH instruments (PHQ-9, CES-D, BDI-II, STAI-Trait, STAI-State, BAI, PSS-10, UCLA)
- **COVID**: ΔR² < 0.015 when excluding pandemic cohort (INS-W_3)
- **ML algorithms**: 4 algorithms (EN, Ridge, RF, SVR) converge; MLP neural network also tested and underperforms
- **Longitudinal**: Personality predicts PHQ-4 mean level (r=+0.339) but not slope (r=-0.012) — trait interpretation confirmed

---

### RQ4: What is the minimum viable assessment?

| Approach | Time | R² (CES-D) | R² (STAI) | Cost | Practical? |
|----------|------|------------|-----------|------|-----------|
| **2 BFI items** | **10 sec** | **0.358** | **0.507** | **$0** | Highly practical |
| Neuroticism score | 1 min | 0.258 | 0.477 | $0 | Practical |
| Full Big Five | 5 min | 0.279 | 0.522 | $0 | Practical |
| All 44 BFI items | 5 min | 0.328 | — | $0 | Practical |
| Sensing (3 PCA) | weeks | -0.011 | -0.008 | $100+ | Impractical |
| Sensing (28 raw) | weeks | -0.162 | -0.101 | $100+ | Impractical |
| Pers + Sensing | weeks | 0.321 | 0.530 | $100+ | Marginal gain |

The best two items are **BFI-4** ("Is depressed, blue", r=0.605 with CES-D) and **BFI-24** ("Is emotionally stable, not easily upset", r=0.415 with CES-D). These are essentially direct mood probes embedded in a personality questionnaire. The finding that two self-report items outperform weeks of passive monitoring underscores the fundamental limitation of behavioral sensing for between-person mental health prediction.

**Key figure**: `results/comparison/supplementary/figure_cost_effectiveness.png`

---

### RQ5: Under what specific conditions does sensing have value?

**When sensing helps (8/41 analyses show some signal):**

| Condition | Evidence | Effect Size | Practical? |
|-----------|----------|-------------|-----------|
| **Early warning** (lagged) | Auto+Sens R²=0.613 vs Auto 0.582 | +0.031 | Marginal — requires prior MH data |
| **Communication metadata** | Pers+Comm ΔR²=+0.030 (S2 CES-D) | +0.030 | Requires call/SMS logs (privacy concern) |
| **Sleep + nonlinear** | RF 0.316 > Ridge 0.261 (S2 CES-D) | +0.055 | S2 only, does not replicate in S3 |
| **Individual monitoring** | 17% idiographic R²>0.3 | Variable | Needs calibration period, cannot predict who benefits |
| **Engagement signal** | Not wearing ↔ anxiety (r=-0.12) | Small | Easy to implement as missingness flag |
| **S2 classification** | Pers+Beh AUC +0.06-0.08 vs Pers-only | Moderate | Driven by communication features |
| **S3 BDI-II incremental** | ΔR²=0.024, p_fdr=0.047 | Small | Only surviving FDR-corrected test |
| **Concurrent weekly** | Pooled between-person R²=0.011 | Trivial | Not practical |

**When sensing does NOT help (33/41 analyses):**
Raw features (Analysis 1), feature selection (Analysis 13), behavioral variability (Analysis 12), temporal inertia (Analysis 17), ipsative/person-centered features (Analysis 22), stacking ensemble (Analysis 11), personality x sensing interaction (Analysis 21), subgroup analysis (Analysis 7), error rescue (Analysis 28), dose-response 7-92 days (Analysis 10), within-person tracking (Analysis 24), prospective change prediction (Analysis 15), social jet lag (Analysis 30), worst-20% rescue (Analysis 31), cross-study transfer (Analysis 14), residualized prediction (Analysis 9), reverse prediction of personality (Analysis 8), and more.

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

### Complete File Count by Phase

| Phase | Description | Scripts | CSV Tables | PNG Figures | Total Files |
|-------|-------------|---------|------------|-------------|-------------|
| 0-11 | Study 1 (StudentLife) | 12 | 17 | 26 | 55 |
| 12 | Study 2 (NetHealth) | 6 | 18 | 9 | 33 |
| 13 | Study 3 (GLOBEM) | 6 | 20 | 11 | 37 |
| 14 | Meta-Analysis + Longitudinal | 2 | 5 | 3 | 10 |
| 15 | Clinical Utility + MLP | 2 | 6 | 2 | 10 |
| 16a-g | Supplementary Analyses | 8 | 34 | 10 | 52 |
| — | Cross-study comparison | — | 11 | 4 | 15 |
| **Total** | | **36** | **~111** | **~65** | **~212** |

---

## What We Learned: Key Methodological Insights

1. **PCA does not kill sensing signals — there is no signal to kill.** Analysis 1 tested 2,597 raw RAPIDS features against 5 PCA composites. Raw features performed catastrophically worse (R²=-1.1 to -16.7), while PCA composites performed at R² approximately 0. More features means more noise, not more signal.

2. **Sensing measurement quality is excellent — the problem is signal, not noise.** Split-half reliability for all 19 S3 features ranges from r=0.94 to r=0.999 (Analysis 3). Signal-to-noise ratios are adequate (0.6-17.5). The features are reliably measured; they simply do not predict mental health at the between-person level.

3. **More data collection time does not help.** Dose-response analysis (Analysis 10) shows sensing R² is flat from 7 days to 92 days. The signal is not buried in short measurement windows — it is absent.

4. **The personality-MH relationship is linear.** MLP neural networks underperform Ridge/EN regression in every comparison, confirming that non-linear modeling cannot rescue sensing predictions (MLP Check). With 5 personality features, the relationship to MH is well-captured by linear models.

5. **Two questionnaire items outperform weeks of passive monitoring.** BFI items 4 and 24 (R²=0.358 for CES-D) exceed 28 sensing features (R²=-0.162). This is because these items directly probe the construct of interest (emotional stability) rather than trying to infer it from behavioral proxies.

6. **Communication metadata is the only sensing modality with any signal.** S2's communication features (SMS/call metadata) drive its marginal Pers+Beh advantage. S3 lacks these features, which explains the S2-S3 gap. Activity and sleep add essentially nothing.

7. **Sensing may work idiographically but not nomothetically.** 17% of individuals show person-specific R² > 0.3, but the sensing-mood relationship direction varies across people. A single population-level model cannot capture these heterogeneous individual patterns.

8. **Cross-model triangulation provides robust evidence.** Using 4 ML algorithms (EN, Ridge, RF, SVR) with Optuna tuning and SHAP interpretation, plus permutation testing, protects against method-specific artifacts. The convergence of 4 algorithms on the same conclusion strengthens inference.

---

## Limitations

1. **Sample characteristics**: All three samples are college students at selective US universities. Generalization to clinical populations, older adults, non-Western samples, or community settings is unknown.

2. **Personality measurement**: S3 uses BFI-10 (2 items/dimension, alpha ~0.65), which attenuates observed effects. While disattenuation correction addresses this statistically, actual studies with full-length instruments would provide stronger evidence.

3. **Cross-sectional design**: Despite longitudinal trajectory and pre-post analyses confirming the trait interpretation, the core prediction models are cross-sectional. Causal claims cannot be made from observational data.

4. **Sensing modality coverage**: Each study captures different behavioral domains. S1 has the richest smartphone data (13 modalities), S2 has communication metadata, S3 has the most features. No single study captures all possible sensing modalities (e.g., GPS, social media, wearable physiology).

5. **Study 1 overfitting**: N=28 is insufficient for reliable ML modeling. The S1 behavior→PHQ-9 finding (R²=0.468) is almost certainly an overfit, as it does not replicate in S2 or S3.

6. **Mediation untested at scale**: The mediation hypothesis (personality → behavior → outcomes) remains untested at adequate sample sizes. S1 (N=28) lacked power; S2 and S3 were not designed for mediation.

7. **Historical data**: S1 data is from 2013, which may not reflect current smartphone usage patterns or sensing capabilities. However, the personality findings are robust across all time periods tested (2013-2021).

8. **No clinical diagnosis**: All MH outcomes are self-report symptom scales, not clinical diagnoses. Performance in clinical diagnostic prediction may differ.

---

## Recommendations for the Field

1. **Include personality as a baseline comparator.** Any study claiming passive sensing predicts mental health should compare against a 5-minute personality questionnaire. Our meta-analytic personality R² values (0.09-0.53) provide benchmarks.

2. **Report behavior-only models.** Many studies report only combined personality+behavior models, obscuring that sensing adds zero unique variance. Behavior-only R² should be a required comparison.

3. **Apply FDR correction for incremental validity.** Uncorrected p-values inflate false-positive rates when testing multiple outcomes. Our 3/8 uncorrected vs 1/8 FDR-corrected comparison illustrates this.

4. **Invest in idiographic approaches.** If sensing has value, it is for personalized monitoring (17% of individuals), not population screening. Research should shift toward N-of-1 designs with calibration periods.

5. **Prioritize communication metadata.** Among all behavioral features, communication (call/SMS) metadata shows the most consistent marginal signal. Activity and sleep data add essentially nothing to personality-based predictions.

6. **Use the Screen-then-Monitor paradigm.** Screen with brief questionnaires (2-5 items), then deploy sensing only for high-risk individuals where personalized monitoring might add value.

7. **Report negative results.** Our 33/41 null analyses represent important evidence that the field needs to internalize. Publication bias toward positive results distorts the field's understanding of sensing's true predictive capacity.
