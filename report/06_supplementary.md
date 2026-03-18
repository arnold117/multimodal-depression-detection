# 41 Supplementary Analyses — Complete Results

> Phase 16a–g | 8 scripts | 34 CSV + 10 PNG output files
>
> All outputs in: `results/comparison/supplementary/`

For detailed write-ups of each analysis including tables and methodology, see [supplementary_analyses_report.md](../supplementary_analyses_report.md). This file provides a concise summary.

---

## Phase 16a: Core Robustness (Analyses 1–7)

**Script**: `supplementary_extended.py`, `supplementary_core.py`, `supplementary_rapids_fast.py`

| # | Analysis | Result | Conclusion |
|---|----------|--------|------------|
| 1 | Raw RAPIDS (1258 feat) vs PCA | Raw R²=-3 to -17 | **More features = catastrophically worse** |
| 2 | Clinical calibration + DeLong | Brier 0.12-0.25, DeLong all n.s. | Good calibration; no AUC difference Pers vs Pers+Beh |
| 3 | Sensing reliability (split-half) | r=0.94–0.999 | Sensing is highly reliable — problem is signal, not noise |
| 4 | Modality ablation | All ΔR² < 0.03 | Sleep best (+0.024 for STAI); no hidden gem |
| 5 | Power analysis | 6/8 power ≥ 0.80 | Most null results are true nulls |
| 6 | Disattenuation | S3 R² corrected up 40% | Still personality >> sensing after correction |
| 7 | Subgroup | All subgroups R² ≤ 0 | No differential sensing utility |

---

## Phase 16b: Alternative Signals (Analyses 8–14)

**Script**: `supplementary_phase16b.py`

| # | Analysis | Result | Conclusion |
|---|----------|--------|------------|
| 8 | Sensing → Personality | Mean R²=0.005 | Sensing can't predict personality |
| 9 | Residualized prediction | Beh→Residual all R² ≤ 0 | Zero unique info beyond personality |
| 10 | Dose-response (7-92 days) | Flat curve | More data doesn't help |
| 11 | Stacking ensemble | Δ ≤ 0.01 | Fusion method not the problem |
| 12 | Variability features (SD/CV) | All R² ≈ 0 | Behavioral volatility uninformative |
| 13 | Feature selection (top-5) | R² = 0.004-0.010 | Smart selection still fails |
| 14 | Cross-study transfer | S2→S3 R²=-0.42 | Instrument mismatch dominates |

---

## Phase 16c: Temporal & Item-Level (Analyses 15–19)

**Script**: `supplementary_phase16c.py`

| # | Analysis | Result | Conclusion |
|---|----------|--------|------------|
| 15 | Prospective (Pre→Post) | Change R² < 0 | Neither pers nor sensing predicts MH change |
| 16 | Within-person (PHQ-4 × sensing) | Mean within-r ≈ 0 | Near-zero group-level within-person signal |
| 17 | Inertia (autocorrelation) | R² ≤ 0 | Critical slowing down not supported |
| 18 | S2 raw 28 features | R²=-0.10 to -0.16 | Raw worse than PCA (confirms #1) |
| **19** | **Item-level (2 BFI items)** | **R²=0.36 vs sensing -0.16** | **10 sec questionnaire >> weeks of sensing** |

---

## Phase 16d: Pro-Sensing (Analyses 20–25)

**Script**: `supplementary_phase16d.py`

| # | Analysis | Result | Conclusion |
|---|----------|--------|------------|
| **20** | **Idiographic models** | **17% of people R²>0.3** | Sensing works for some — can't predict who |
| 21 | Personality × Sensing interaction | All Δ ≤ 0 | No moderation effect |
| 22 | Ipsative (person-centered) | Trace for STAI (R²=0.021) | Minimal |
| **23** | **S2 deep dive** | **Comm ΔR²=+0.030 for CES-D** | Communication data is S2's secret |
| 24 | Weekly panel (3149 obs) | Within-person R²=-0.003 | Zero within-person signal |
| **25** | **Nonlinear sleep (RF)** | **S2 CES-D: RF 0.316 > Ridge 0.261** | Nonlinear effect exists in S2 |

---

## Phase 16e: Context & Fairness (Analyses 26–31)

**Script**: `supplementary_phase16e.py`

| # | Analysis | Result | Conclusion |
|---|----------|--------|------------|
| 26 | Shared method variance | SR/Obj ratio = 3.1×; Harman 29% | ~30% inflation but below concern threshold |
| **27** | **Missing data as signal** | **Completeness↔STAI r=-0.12**** | Not wearing device = more anxious |
| **28** | **Lagged (this week→next)** | **Auto+Sens R²=0.613 (+0.031)** | Sensing's best result: early warning |
| 29 | Error analysis (hard cases) | Beh→Residual R² ≈ 0 | Cannot rescue personality's failures |
| 30 | Cost-effectiveness | 2 items R²/min=2.11; sensing=0 | Infinite cost-efficiency gap |
| 31 | Literature benchmark | Our AUC=0.57 = GLOBEM=Muller | We match published ceiling |

---

## Phase 16f: Final Checks (Analyses 32–34)

**Script**: `supplementary_phase16f.py`

| # | Analysis | Result | Conclusion |
|---|----------|--------|------------|
| 32 | Missingness pattern | Steps/sleep miss→BDI r=0.12* | Missingness pattern weakly informative |
| 33 | Social jet lag | All ΔR² ≤ 0 | Weekend/weekday shift doesn't predict MH |
| 34 | Worst 20% rescue | All R² ≤ 0 for worst-predicted | Sensing fails for hardest cases too |

---

## Phase 16g: Complete Data Utilization (Analyses 35–41)

**Script**: `supplementary_phase16g.py`

| # | Analysis | Result | Conclusion |
|---|----------|--------|------------|
| **35** | **EMA × sensing (3382 obs)** | **Within R²≈0; per-person \|r\|=0.35** | Moderate individual signal, cancels at group level |
| 37 | S2 extended outcomes | Self-esteem: Pers R²=0.41 | N dominates self-esteem; pattern consistent |
| 38 | Full demographics (SES/race) | Pers ΔR²=0.20-0.52*** after demo | Personality robust to all SES controls |
| 39 | Sensing → GPA | R²=-0.044 | Sensing fails for academics too |
| 40 | S1 consistency check | PHQ-9: Sens>Pers (N=27 overfit) | Only exception at tiny N |
| **41** | **Grand synthesis (15 outcomes)** | **Pers wins 14/15 (93%)** | Definitive cross-study conclusion |

---

## Summary: 10 Positive Signals (Out of 41)

| # | Finding | Effect | Condition |
|---|---------|--------|-----------|
| 1 | Lagged prediction | ΔR²=+0.031 | Over autoregressive |
| 2 | Communication → CES-D | ΔR²=+0.030 | S2 SMS/call only |
| 3 | Sleep → STAI | ΔR²=+0.024 | Best modality |
| 4 | Missingness → STAI | ΔR²=+0.018 | Completeness feature |
| 5 | 17% idiographic R²>0.3 | Variable | Can't predict who |
| 6 | RF nonlinear (S2) | +0.055 | S2 CES-D only |
| 7 | Completeness ↔ anxiety | r=-0.12** | Engagement signal |
| 8 | Sens@Spec=0.80 in S2 | +0.17 sens | S2 classification |
| 9 | EMA per-person |r|=0.35 | Moderate | Direction varies |
| 10 | Personality survives SES | ΔR²=0.20-0.52*** | All demographics controlled |
