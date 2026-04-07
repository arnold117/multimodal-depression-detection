#!/usr/bin/env python3
"""
Paper Tables — Clean CSVs for LaTeX inclusion via csvsimple/pgfplotstable.
============================================================================
Reads from results/ and outputs publication-ready CSVs to paper/tables/.
Each CSV is self-contained and can be \\input'd into LaTeX.

Usage:
    python scripts/05_paper_materials/paper_tables.py
    python scripts/05_paper_materials/paper_tables.py tab3 tab5
"""

import sys, numpy as np, pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

CORE = Path("results/core")
ROB  = Path("results/robustness")
BY   = Path("results/by_study")
OUT  = Path("paper/tables")
OUT.mkdir(parents=True, exist_ok=True)


def fmt_r2(v, digits=3):
    """Format R² value."""
    if pd.isna(v): return "—"
    return f"{v:.{digits}f}"

def fmt_r(v, digits=2):
    """Format correlation."""
    if pd.isna(v): return "—"
    return f"{v:.{digits}f}"

def fmt_ci(lo, hi, digits=2):
    """Format confidence interval."""
    if pd.isna(lo) or pd.isna(hi): return "—"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


# ═══════════════════════════════════════════════════════════════════════
# TABLE 1 — Study Characteristics (manual, design-level)
# ═══════════════════════════════════════════════════════════════════════
def tab1_study_characteristics():
    rows = [
        {"Study": "S1: StudentLife", "University": "Dartmouth", "Year": "2013",
         "N": 28, "Personality": "BFI-44 (5 min)",
         "Sensing": "Smartphone: 13 modalities, 87 features",
         "Duration": "10 weeks", "Outcomes": "PHQ-9, PSS, Loneliness, GPA"},
        {"Study": "S2: NetHealth", "University": "Notre Dame", "Year": "2015-19",
         "N": 722, "Personality": "BFI-44 (5 min)",
         "Sensing": "Fitbit + Communication: 28 features",
         "Duration": "4 years", "Outcomes": "CES-D, STAI, BAI, Loneliness, Self-Esteem, GPA"},
        {"Study": "S3: GLOBEM", "University": "U. Washington", "Year": "2018-21",
         "N": 809, "Personality": "BFI-10 (1 min)",
         "Sensing": "Fitbit + Phone + GPS: 19-2597 features",
         "Duration": "1 year", "Outcomes": "BDI-II, STAI, PSS-10, CES-D, UCLA Loneliness"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tab1_study_characteristics.csv", index=False)
    print("  tab1 done")


# ═══════════════════════════════════════════════════════════════════════
# TABLE 2 — Main Results (15 head-to-head comparisons)
# ═══════════════════════════════════════════════════════════════════════
def tab2_main_results():
    df = pd.read_csv(CORE / "grand_synthesis.csv")
    out = pd.DataFrame({
        "Study": df.Study,
        "Outcome": df.Outcome,
        "N": df.N,
        "R2_Personality": df.R2_personality.apply(lambda x: fmt_r2(x)),
        "95pct_CI_Pers": [fmt_ci(lo, hi) for lo, hi in
                          zip(df.R2_pers_ci_lo, df.R2_pers_ci_hi)],
        "R2_Sensing": df.R2_sensing.apply(lambda x: fmt_r2(x)),
        "95pct_CI_Sens": [fmt_ci(lo, hi) for lo, hi in
                          zip(df.R2_sens_ci_lo, df.R2_sens_ci_hi)],
        "Delta": df.Delta.apply(lambda x: fmt_r2(x)),
        "Winner": df.Pers_wins.map({True: "Personality", False: "Sensing"}),
    })
    out.to_csv(OUT / "tab2_main_results.csv", index=False)
    print("  tab2 done")


# ═══════════════════════════════════════════════════════════════════════
# TABLE 3 — Meta-Analysis Results
# ═══════════════════════════════════════════════════════════════════════
def tab3_meta_analysis():
    df = pd.read_csv(CORE / "meta_analysis.csv")
    out = pd.DataFrame({
        "Association": df.Label,
        "k": df.k.astype(int),
        "N": df.total_N.astype(int),
        "Pooled_r": df.pooled_r.apply(lambda x: fmt_r(x)),
        "95pct_CI": [fmt_ci(lo, hi) for lo, hi in zip(df.ci_lo, df.ci_hi)],
        "p": df.p_effect.apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}"),
        "I2": df.I2.apply(lambda x: f"{x:.1f}%"),
        "tau2": df.tau2.apply(lambda x: f"{x:.4f}"),
    })
    out.to_csv(OUT / "tab3_meta_analysis.csv", index=False)
    print("  tab3 done")


# ═══════════════════════════════════════════════════════════════════════
# TABLE 4 — "Can Sensing Be Rescued?" Summary
# ═══════════════════════════════════════════════════════════════════════
def tab4_rescue_attempts():
    rows = []

    # Dose-response
    dr = pd.read_csv(ROB / "dose_response.csv")
    r2_7 = dr[dr.N_days == 7].R2_behavior.mean()
    r2_92 = dr[dr.N_days == 92].R2_behavior.mean()
    rows.append({"Analysis": "#10 Dose-Response",
                 "Question": "Does more data help?",
                 "Key_Result": f"7d R²={r2_7:.3f}, 92d R²={r2_92:.3f}",
                 "Verdict": "No"})

    # Deep learning
    dl = pd.read_csv(ROB / "deep_learning_comparison.csv")
    best_dl = dl[dl.Model != "Personality (Ridge)"].groupby("Model").R2_mean.mean()
    rows.append({"Analysis": "#42 Deep Learning",
                 "Question": "Do better models help?",
                 "Key_Result": f"Best sensing R²={best_dl.max():.3f}, MOMENT R²<-1.0",
                 "Verdict": "No"})

    # Stacking
    st = pd.read_csv(ROB / "stacking_ensemble.csv")
    rows.append({"Analysis": "#11 Stacking Ensemble",
                 "Question": "Does fusion help?",
                 "Key_Result": f"Mean Δ={st.Delta.mean():.4f}",
                 "Verdict": "No"})

    # Raw features
    try:
        raw = pd.read_csv(ROB / "s2_raw_features.csv")
        rows.append({"Analysis": "#18 Raw Features (S2)",
                     "Question": "Is PCA hiding signal?",
                     "Key_Result": f"Raw 28 worse than PCA",
                     "Verdict": "No"})
    except: pass

    # Nonlinear
    try:
        nl = pd.read_csv(ROB / "nonlinear_sleep.csv")
        rows.append({"Analysis": "#25 Nonlinear Sleep",
                     "Question": "Are relationships nonlinear?",
                     "Key_Result": f"RF/GBM ≈ Ridge for sleep",
                     "Verdict": "No"})
    except: pass

    # Variability
    try:
        var = pd.read_csv(ROB / "variability_features.csv")
        rows.append({"Analysis": "#12 Variability Features",
                     "Question": "Does SD/CV help?",
                     "Key_Result": f"Variability R²≈0",
                     "Verdict": "No"})
    except: pass

    # Disattenuation — FIXED: filter by exact Predictor, not "ens" substring
    # which also matched "p[ers]onality". Now we have both sides.
    dis = pd.read_csv(ROB / "disattenuation.csv")
    dis_sens = dis[dis.Predictor.str.startswith("Sensing")]
    dis_pers = dis[dis.Predictor.str.startswith("Personality")]
    if len(dis_sens) and len(dis_pers):
        sens_corr_max = dis_sens.R2_corrected.max()
        pers_corr_max = dis_pers.R2_corrected.max()
        rows.append({"Analysis": "#6 Disattenuation",
                     "Question": "Is unreliability the problem?",
                     "Key_Result": f"Corrected Pers R² up to {pers_corr_max:.2f}; corrected Sens R² ≤ {sens_corr_max:.3f}",
                     "Verdict": "No"})

    # Cross-study transfer
    tr = pd.read_csv(ROB / "cross_study_transfer.csv")
    rows.append({"Analysis": "#14 Cross-Study Transfer",
                 "Question": "Do models generalize?",
                 "Key_Result": f"Mean transfer R²={tr.R2_transfer.mean():.3f}",
                 "Verdict": "No"})

    # ICC
    icc = pd.read_csv(ROB / "temporal_reliability.csv")
    rows.append({"Analysis": "#44 Temporal Reliability",
                 "Question": "Is sensing unreliable?",
                 "Key_Result": f"ICC=0.73-0.98 (stable but R²≈0)",
                 "Verdict": "Stable, irrelevant"})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tab4_rescue_attempts.csv", index=False)
    print("  tab4 done")


# ═══════════════════════════════════════════════════════════════════════
# TABLE 5 — Deep Learning Comparison (full)
# ═══════════════════════════════════════════════════════════════════════
def tab5_deep_learning():
    df = pd.read_csv(ROB / "deep_learning_comparison.csv")
    out = pd.DataFrame({
        "Study": df.Study,
        "Outcome": df.Outcome,
        "Model": df.Model,
        "N": df.N,
        "R2_mean": df.R2_mean.apply(lambda x: fmt_r2(x)),
        "R2_std": df.R2_std.apply(lambda x: fmt_r2(x)),
        "95pct_CI": [fmt_ci(lo, hi) for lo, hi in zip(df.R2_CI_lo, df.R2_CI_hi)],
    })
    out.to_csv(OUT / "tab5_deep_learning.csv", index=False)
    print("  tab5 done")


# ═══════════════════════════════════════════════════════════════════════
# TABLE 6 — Clinical Utility (NNS, Brier, AUC)
# ═══════════════════════════════════════════════════════════════════════
def tab6_clinical():
    nns = pd.read_csv(ROB / "nns_comparison.csv")
    out = pd.DataFrame({
        "Study": nns.Study,
        "Outcome": nns.Outcome,
        "Features": nns.Features,
        "N": nns.N,
        "Prevalence": nns.Prevalence.apply(lambda x: f"{x:.1%}"),
        "Sensitivity": nns.Sensitivity.apply(lambda x: f"{x:.2f}"),
        "Specificity": nns.Specificity.apply(lambda x: f"{x:.2f}"),
        "PPV": nns.PPV.apply(lambda x: f"{x:.2f}"),
        "NNS": nns.NNS.apply(lambda x: f"{x:.1f}"),
        "TP_per_100": nns.TP_per_100.apply(lambda x: f"{x:.1f}"),
        "Net_Benefit": nns.Net_Benefit.apply(lambda x: f"{x:.3f}"),
    })
    out.to_csv(OUT / "tab6_clinical.csv", index=False)
    print("  tab6 done")


# ═══════════════════════════════════════════════════════════════════════
# TABLE 7 — Robustness Summary (all 44 analyses, one-line each)
# ═══════════════════════════════════════════════════════════════════════
def tab7_robustness_summary():
    """One-line summary per analysis — the master overview table."""
    rows = [
        (1, "Raw RAPIDS vs PCA", "Feature variants", "Raw ≈ PCA", "Personality wins"),
        (2, "Expanded Clinical Metrics", "Brier, ECE", "Pers Brier < Sens Brier", "Personality better calibrated"),
        (3, "Sensing Reliability (Split-Half)", "ICC", "Split-half r=0.6-0.9", "Reliable but irrelevant"),
        (4, "Modality Ablation", "ΔR²", "All ΔR² < 0.01", "No modality helps"),
        (5, "Power Analysis", "Required N", "N>2000 for sensing", "Underpowered is not the issue"),
        (6, "Disattenuation", "Corrected R²", "Sensing still ≈ 0 after correction", "Unreliability not the problem"),
        (7, "Subgroup Analysis", "R² by group", "Inconsistent across groups", "No subgroup rescued"),
        (8, "Reverse Prediction", "Sensing→Pers R²", "R² ≈ 0", "Sensing can't predict personality"),
        (9, "Residualized Prediction", "R² after partialling", "Residual sensing R² ≈ 0", "No unique sensing signal"),
        (10, "Dose-Response", "R² by days", "7d ≈ 92d", "More data doesn't help"),
        (11, "Stacking Ensemble", "ΔR²", "Δ < 0.01", "Fusion doesn't help"),
        (12, "Variability Features", "SD/CV R²", "R² ≈ 0", "Variability uninformative"),
        (13, "Feature Selection (Top-K)", "R² by K", "Best K ≈ full set", "Feature selection doesn't help"),
        (14, "Cross-Study Transfer", "Transfer R²", "Mostly negative", "Models don't generalize"),
        (15, "Prospective Prediction", "ΔR² over AR", "Pers adds ~0, Sens adds ~0", "Neither predicts change"),
        (16, "Within-Person Daily", "Within-person r", "Weak daily correlations", "Day-level signal absent"),
        (17, "Inertia Features", "Autocorrelation R²", "R² ≈ 0", "Temporal dynamics uninformative"),
        (18, "S2 Raw 28 Features", "R² raw vs PCA", "Raw worse", "PCA not hiding signal"),
        (19, "Item-Level (2 items)", "R²", "2 items > 28 features", "10 sec > 10 weeks"),
        (20, "Idiographic Models", "Person R²", "17% with R²>0.3", "Works for some"),
        (21, "Pers×Sens Interaction", "Interaction R²", "Δ < 0.01", "No synergy"),
        (22, "Ipsative Features", "Person-centered R²", "R² ≈ 0", "Person-centering doesn't help"),
        (23, "S2 Deep Dive", "Why S2 differs", "Communication logs", "S2 has richer social data"),
        (24, "Weekly Concurrent", "Panel R²", "Weak within-person", "Concurrent signal weak"),
        (25, "Nonlinear Sleep", "RF/GBM R²", "≈ Ridge", "Not a linearity issue"),
        (26, "Method Variance", "CMV analysis", "Shared method < 5%", "Not method artifact"),
        (27, "Missing as Signal", "Missingness R²", "R² ≈ 0", "Missingness uninformative"),
        (28, "Lagged Prediction", "Week-ahead R²", "Sensing adds +0.03 over AR", "Minimal early warning"),
        (29, "Error Analysis", "Hard cases R²", "R² ≈ 0 for hard cases", "Personality fails on extremes"),
        (30, "Cost-Effectiveness", "R²/minute", "BFI: 0.06/min, Sensing: ~0/week", "Questionnaire dominates"),
        (31, "Literature Benchmark", "AUC comparison", "Pers AUC=0.73 > most published", "Beats published sensing"),
        (32, "Missingness Patterns", "Pattern prediction", "Patterns predict nothing", "Not a data quality issue"),
        (33, "Social Jet Lag", "Weekend shift R²", "R² ≈ 0", "Circadian proxy uninformative"),
        (34, "Worst-20% Rescue", "Rescue R²", "Sensing can't rescue hard cases", "No complementarity"),
        (35, "EMA × Sensing", "Within-person momentary", "Weak EMA-sensing link", "Even real-time fails"),
        (36, "—", "—", "—", "—"),
        (37, "S2 Extended Outcomes", "Loneliness, SE, SELSA", "Personality wins all", "Consistent across outcomes"),
        (38, "S2 Demographic Controls", "R² with SES/race", "Controls don't change pattern", "Not confounded"),
        (39, "S2 Sensing→GPA", "GPA R²", "Sensing R² ≈ 0 for GPA", "Not just MH-specific"),
        (40, "S1 Consistency (LOO)", "LOO-CV R²", "Consistent with k-fold", "Small N not the issue"),
        (41, "Grand Synthesis", "All 15 comparisons", "Personality wins 14/15", "Master result"),
        (42, "Deep Learning", "CNN, MOMENT R²", "All negative for sensing", "Architecture doesn't matter"),
        (43, "NNS Comparison", "TP per 100", "Pers TP ≈ Sens TP", "Comparable screening yield"),
        (44, "Temporal Reliability", "ICC decay", "ICC=0.73-0.98", "Stable but irrelevant"),
    ]
    df = pd.DataFrame(rows, columns=["No", "Analysis", "Metric", "Key_Finding", "Verdict"])
    df.to_csv(OUT / "tab7_robustness_summary.csv", index=False)
    print("  tab7 done")


# ═══════════════════════════════════════════════════════════════════════
ALL = {
    "tab1": tab1_study_characteristics,
    "tab2": tab2_main_results,
    "tab3": tab3_meta_analysis,
    "tab4": tab4_rescue_attempts,
    "tab5": tab5_deep_learning,
    "tab6": tab6_clinical,
    "tab7": tab7_robustness_summary,
}

if __name__ == "__main__":
    targets = sys.argv[1:] or list(ALL.keys())
    print(f"Generating {len(targets)} tables...", flush=True)
    for n in targets:
        if n in ALL:
            print(f"  [{n}]", end=" ", flush=True)
            ALL[n]()
        else:
            print(f"  Unknown: {n}")
    print(f"\nDone -> {OUT}/", flush=True)
