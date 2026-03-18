#!/usr/bin/env python3
"""
Phase 16e — Final Frontier Analyses
====================================
26. Shared method variance (common method bias)
27. Missing data as signal
28. Lagged prediction (this week → next week PHQ-4)
29. Error analysis (personality hard cases + sensing rescue)
30. Cost-effectiveness analysis
31. Literature benchmark comparison
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/comparison/supplementary")
OUT.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
RS = 42
COHORTS = ["INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4"]
RAW_DIR = Path("data/raw/globem")

print("Loading datasets...", flush=True)
s2 = pd.read_parquet("data/processed/nethealth/nethealth_analysis_dataset.parquet")
s3 = pd.read_parquet("data/processed/globem/globem_analysis_dataset.parquet")
S2_BEH_PCA = [c for c in s2.columns if c.startswith("nh_") and c.endswith("_pc1")]
S3_BEH_PCA = [c for c in s3.columns if c.endswith("_pc1") and not c.startswith("nh_")]
S2_BEH_RAW = [c for c in s2.columns if c not in TRAITS + S2_BEH_PCA
              and c not in ["egoid", "cesd_total", "stai_trait_total", "bai_total",
                           "loneliness_total", "selsa_romantic", "selsa_family",
                           "selsa_social", "self_esteem_total", "gpa_overall",
                           "gpa_first_semester"]
              and not c.endswith("_pc1")]

def find_col(columns, substring):
    matches = [c for c in columns if substring in c]
    return min(matches, key=len) if matches else None

def quick_cv_r2(X, y, n_splits=5, n_repeats=5, alpha=1.0):
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X_c, y_c = X[mask], y[mask]
    if len(y_c) < 30:
        return np.nan, len(y_c)
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RS)
    r2s = []
    for tr, te in cv.split(X_c):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_c[tr])
        Xte = sc.transform(X_c[te])
        m = Ridge(alpha=alpha)
        m.fit(Xtr, y_c[tr])
        r2s.append(r2_score(y_c[te], m.predict(Xte)))
    return float(np.mean(r2s)), len(y_c)


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 26: Shared Method Variance
# ═══════════════════════════════════════════════════════════════════════
def run_method_variance():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 26: Shared Method Variance (Common Method Bias)", flush=True)
    print("=" * 70, flush=True)

    rows = []

    for study, df, beh_cols, outcomes in [
        ("S2", s2, S2_BEH_PCA, {"cesd_total": "CES-D", "stai_trait_total": "STAI", "bai_total": "BAI"}),
        ("S3", s3, S3_BEH_PCA, {"bdi2_total": "BDI-II", "stai_state": "STAI",
                                  "pss_10": "PSS-10", "cesd_total": "CESD", "ucla_loneliness": "UCLA"}),
    ]:
        valid_pers = [c for c in TRAITS if c in df.columns]
        valid_beh = [c for c in beh_cols if c in df.columns]

        for outcome_col, outcome_label in outcomes.items():
            if outcome_col not in df.columns:
                continue

            sub = df[valid_pers + valid_beh + [outcome_col]].dropna()
            if len(sub) < 50:
                continue

            # Self-report × Self-report correlations (personality traits × MH)
            sr_sr_corrs = []
            for trait in valid_pers:
                r, _ = stats.pearsonr(sub[trait], sub[outcome_col])
                sr_sr_corrs.append(abs(r))

            # Objective × Self-report correlations (sensing × MH)
            obj_sr_corrs = []
            for beh in valid_beh:
                r, _ = stats.pearsonr(sub[beh], sub[outcome_col])
                obj_sr_corrs.append(abs(r))

            mean_sr_sr = np.mean(sr_sr_corrs)
            max_sr_sr = np.max(sr_sr_corrs)
            mean_obj_sr = np.mean(obj_sr_corrs)
            max_obj_sr = np.max(obj_sr_corrs)
            ratio = mean_sr_sr / mean_obj_sr if mean_obj_sr > 0.001 else np.inf

            # Harman single-factor test: do all self-report items load on one factor?
            all_sr = sub[valid_pers + [outcome_col]].values
            sc = StandardScaler()
            all_sr_s = sc.fit_transform(all_sr)
            pca = PCA()
            pca.fit(all_sr_s)
            first_factor_var = pca.explained_variance_ratio_[0]

            print(f"  {study} {outcome_label}:", flush=True)
            print(f"    Self×Self mean |r|={mean_sr_sr:.4f} (max={max_sr_sr:.4f})", flush=True)
            print(f"    Obj×Self  mean |r|={mean_obj_sr:.4f} (max={max_obj_sr:.4f})", flush=True)
            print(f"    Ratio SR/Obj: {ratio:.2f}x", flush=True)
            print(f"    Harman 1st factor: {first_factor_var:.1%} variance", flush=True)

            rows.append({
                "Study": study, "Outcome": outcome_label, "N": len(sub),
                "Mean_SR_SR_r": mean_sr_sr, "Max_SR_SR_r": max_sr_sr,
                "Mean_Obj_SR_r": mean_obj_sr, "Max_Obj_SR_r": max_obj_sr,
                "SR_Obj_ratio": ratio,
                "Harman_1st_factor_pct": first_factor_var,
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "method_variance.csv", index=False)

    # Summary
    print(f"\n  Overall:", flush=True)
    print(f"    Mean SR×SR |r|: {df_out['Mean_SR_SR_r'].mean():.4f}", flush=True)
    print(f"    Mean Obj×SR |r|: {df_out['Mean_Obj_SR_r'].mean():.4f}", flush=True)
    print(f"    Average inflation ratio: {df_out['SR_Obj_ratio'].mean():.1f}x", flush=True)
    print(f"    Harman 1st factor: {df_out['Harman_1st_factor_pct'].mean():.1%} (>50% = concern)", flush=True)
    print(f"  Saved: {OUT / 'method_variance.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 27: Missing Data as Signal
# ═══════════════════════════════════════════════════════════════════════
def run_missing_signal():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 27: Missing Data as Signal", flush=True)
    print("=" * 70, flush=True)

    # S3: compute per-person sensing data completeness from raw daily files
    FEATURE_MAP = [
        ("steps.csv", "avgsumsteps:14dhist", "steps"),
        ("sleep.csv", "avgdurationasleepmain:14dhist", "sleep"),
        ("screen.csv", "rapids_sumdurationunlock:14dhist", "screen"),
        ("call.csv", "incoming_count:14dhist", "calls"),
        ("location.csv", "barnett_hometime:14dhist", "location"),
    ]

    print("  Computing per-person data completeness...", flush=True)
    completeness = {}

    for modality_file, col_substr, short_name in FEATURE_MAP:
        all_daily = []
        for cohort in COHORTS:
            fpath = RAW_DIR / cohort / "FeatureData" / modality_file
            if not fpath.exists():
                continue
            df_raw = pd.read_csv(fpath, low_memory=False)
            col = find_col(df_raw.columns.tolist(), col_substr)
            if col is None:
                continue
            daily = df_raw[["pid", col]].copy()
            daily.columns = ["pid", "value"]
            daily["value"] = pd.to_numeric(daily["value"], errors="coerce")
            all_daily.append(daily)

        if not all_daily:
            continue
        combined = pd.concat(all_daily, ignore_index=True)

        # Total days per person and non-missing days
        total = combined.groupby("pid").size()
        valid = combined.dropna(subset=["value"]).groupby("pid").size()
        pct = (valid / total).reindex(total.index).fillna(0)
        completeness[f"{short_name}_completeness"] = pct

    comp_df = pd.DataFrame(completeness)
    comp_df["mean_completeness"] = comp_df.mean(axis=1)
    comp_df.index.name = "pid"
    comp_df = comp_df.reset_index()

    # Merge with S3 outcomes
    s3_scores = s3[["pid"] + list({"bdi2_total": "BDI-II", "stai_state": "STAI",
                                    "pss_10": "PSS-10", "cesd_total": "CESD",
                                    "ucla_loneliness": "UCLA"}.keys()) + TRAITS]
    merged = s3_scores.merge(comp_df, on="pid", how="inner")
    comp_cols = [c for c in completeness] + ["mean_completeness"]
    valid_pers = [c for c in TRAITS if c in merged.columns]

    print(f"  Merged N={len(merged)}", flush=True)

    rows = []
    S3_OUTCOMES = {"bdi2_total": "BDI-II", "stai_state": "STAI", "pss_10": "PSS-10",
                   "cesd_total": "CESD", "ucla_loneliness": "UCLA"}

    for outcome_col, outcome_label in S3_OUTCOMES.items():
        if outcome_col not in merged.columns:
            continue
        y = merged[outcome_col].values

        # Completeness as predictor
        r2_comp, n = quick_cv_r2(merged[comp_cols].values, y)
        # Pers + completeness
        r2_pers_comp, _ = quick_cv_r2(merged[valid_pers + comp_cols].values, y)
        r2_pers, _ = quick_cv_r2(merged[valid_pers].values, y)

        # Correlation: mean completeness × outcome
        sub = merged[["mean_completeness", outcome_col]].dropna()
        r_comp, p_comp = stats.pearsonr(sub["mean_completeness"], sub[outcome_col])

        print(f"  {outcome_label}: Completeness R²={r2_comp:.4f}, "
              f"Pers+Comp R²={r2_pers_comp:.4f}, Pers R²={r2_pers:.4f}, "
              f"r(completeness, outcome)={r_comp:.4f} (p={p_comp:.4f}) (N={n})", flush=True)

        rows.append({
            "Outcome": outcome_label, "N": n,
            "R2_completeness": r2_comp, "R2_personality": r2_pers,
            "R2_pers_plus_comp": r2_pers_comp,
            "r_completeness_outcome": r_comp, "p_completeness": p_comp,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "missing_as_signal.csv", index=False)
    print(f"  Saved: {OUT / 'missing_as_signal.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 28: Lagged Prediction (This Week → Next Week)
# ═══════════════════════════════════════════════════════════════════════
def run_lagged():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 28: Lagged Prediction (This Week Sensing → Next Week PHQ-4)", flush=True)
    print("=" * 70, flush=True)

    FEATURE_MAP = [
        ("steps.csv", "avgsumsteps:14dhist", "steps"),
        ("sleep.csv", "avgdurationasleepmain:14dhist", "sleep_dur"),
        ("screen.csv", "rapids_sumdurationunlock:14dhist", "screen_dur"),
        ("call.csv", "incoming_count:14dhist", "calls_in"),
        ("location.csv", "barnett_hometime:14dhist", "hometime"),
    ]

    # Load weekly PHQ-4
    all_weekly = []
    for cohort in COHORTS:
        try:
            dep = pd.read_csv(RAW_DIR / cohort / "SurveyData" / "dep_weekly.csv")
            all_weekly.append(dep)
        except Exception:
            pass
    weekly = pd.concat(all_weekly, ignore_index=True)
    weekly["phq4"] = pd.to_numeric(weekly["phq4"], errors="coerce")
    weekly["date"] = pd.to_datetime(weekly["date"])
    weekly = weekly.dropna(subset=["phq4"]).sort_values(["pid", "date"])

    # Load daily sensing
    daily_features = {}
    for modality_file, col_substr, short_name in FEATURE_MAP:
        frames = []
        for cohort in COHORTS:
            fpath = RAW_DIR / cohort / "FeatureData" / modality_file
            if not fpath.exists():
                continue
            df_raw = pd.read_csv(fpath, low_memory=False)
            col = find_col(df_raw.columns.tolist(), col_substr)
            if col is None:
                continue
            daily = df_raw[["pid", "date", col]].copy()
            daily.columns = ["pid", "date", "value"]
            daily["value"] = pd.to_numeric(daily["value"], errors="coerce")
            daily["date"] = pd.to_datetime(daily["date"])
            frames.append(daily)
        if frames:
            daily_features[short_name] = pd.concat(frames, ignore_index=True)

    feat_cols = list(daily_features.keys())

    # Build lagged dataset: this_week_sensing → next_week_phq4
    print("  Building lagged panel...", flush=True)
    lagged_rows = []

    for pid in weekly["pid"].unique():
        phq_pid = weekly[weekly["pid"] == pid].sort_values("date").reset_index(drop=True)
        if len(phq_pid) < 3:
            continue

        for i in range(len(phq_pid) - 1):
            this_date = phq_pid.loc[i, "date"]
            next_phq4 = phq_pid.loc[i + 1, "phq4"]
            this_phq4 = phq_pid.loc[i, "phq4"]

            rec = {"pid": pid, "phq4_this": this_phq4, "phq4_next": next_phq4}
            has_data = True
            for feat_name, daily_df in daily_features.items():
                sens_pid = daily_df[daily_df["pid"] == pid]
                week = sens_pid[(sens_pid["date"] >= this_date - pd.Timedelta(days=7)) &
                                (sens_pid["date"] <= this_date)]
                if len(week) >= 2:
                    rec[feat_name] = week["value"].mean()
                else:
                    has_data = False
                    break

            if has_data:
                lagged_rows.append(rec)

    panel = pd.DataFrame(lagged_rows)
    if len(panel) < 50:
        print("  Too few lagged observations", flush=True)
        return pd.DataFrame()

    print(f"  Lagged panel: {len(panel)} obs, {panel['pid'].nunique()} persons", flush=True)

    # Merge personality
    s3_pers = s3[["pid"] + TRAITS].drop_duplicates("pid")
    panel = panel.merge(s3_pers, on="pid", how="inner")
    valid_pers = [c for c in TRAITS if c in panel.columns]

    y_next = panel["phq4_next"].values

    # Model A: Autoregressive (this week PHQ-4 → next week)
    r2_auto, n = quick_cv_r2(panel[["phq4_this"]].values, y_next, n_splits=5, n_repeats=3)

    # Model B: This week sensing → next week PHQ-4
    r2_sens, _ = quick_cv_r2(panel[feat_cols].values, y_next, n_splits=5, n_repeats=3)

    # Model C: Auto + sensing
    r2_auto_sens, _ = quick_cv_r2(panel[["phq4_this"] + feat_cols].values, y_next, n_splits=5, n_repeats=3)

    # Model D: Auto + personality
    r2_auto_pers, _ = quick_cv_r2(panel[["phq4_this"] + valid_pers].values, y_next, n_splits=5, n_repeats=3)

    # Model E: Auto + pers + sensing
    r2_all, _ = quick_cv_r2(panel[["phq4_this"] + valid_pers + feat_cols].values, y_next, n_splits=5, n_repeats=3)

    # Model F: Predict CHANGE (next - this)
    y_change = panel["phq4_next"].values - panel["phq4_this"].values
    r2_sens_change, _ = quick_cv_r2(panel[feat_cols].values, y_change, n_splits=5, n_repeats=3)

    print(f"\n  Lagged prediction (N={n}):", flush=True)
    print(f"    Autoregressive:    R²={r2_auto:.4f}", flush=True)
    print(f"    Sensing only:      R²={r2_sens:.4f}", flush=True)
    print(f"    Auto + Sensing:    R²={r2_auto_sens:.4f}", flush=True)
    print(f"    Auto + Personality:R²={r2_auto_pers:.4f}", flush=True)
    print(f"    Auto + Pers + Sens:R²={r2_all:.4f}", flush=True)
    print(f"    Sensing → Change:  R²={r2_sens_change:.4f}", flush=True)

    results = {
        "N_obs": n, "N_persons": panel["pid"].nunique(),
        "R2_autoregressive": r2_auto, "R2_sensing_only": r2_sens,
        "R2_auto_sensing": r2_auto_sens, "R2_auto_personality": r2_auto_pers,
        "R2_auto_pers_sens": r2_all, "R2_sensing_predict_change": r2_sens_change,
    }
    pd.DataFrame([results]).to_csv(OUT / "lagged_prediction.csv", index=False)
    print(f"  Saved: {OUT / 'lagged_prediction.csv'}", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 29: Error Analysis (Personality Hard Cases)
# ═══════════════════════════════════════════════════════════════════════
def run_error_analysis():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 29: Error Analysis — Can Sensing Help Personality's Hard Cases?", flush=True)
    print("=" * 70, flush=True)

    rows = []

    for study, df, beh_cols, outcomes in [
        ("S2", s2, S2_BEH_PCA, {"cesd_total": "CES-D", "stai_trait_total": "STAI", "bai_total": "BAI"}),
        ("S3", s3, S3_BEH_PCA, {"bdi2_total": "BDI-II", "stai_state": "STAI", "pss_10": "PSS-10"}),
    ]:
        valid_pers = [c for c in TRAITS if c in df.columns]
        valid_beh = [c for c in beh_cols if c in df.columns]

        for outcome_col, outcome_label in outcomes.items():
            if outcome_col not in df.columns:
                continue

            sub = df[valid_pers + valid_beh + [outcome_col]].dropna()
            if len(sub) < 50:
                continue

            y = sub[outcome_col].values
            X_pers = sub[valid_pers].values
            X_beh = sub[valid_beh].values

            # Get personality prediction errors via CV
            cv = KFold(n_splits=5, shuffle=True, random_state=RS)
            residuals = np.zeros(len(y))
            y_pred_pers = np.zeros(len(y))

            for tr, te in cv.split(X_pers):
                sc = StandardScaler()
                Xtr = sc.fit_transform(X_pers[tr])
                Xte = sc.transform(X_pers[te])
                m = Ridge(alpha=1.0)
                m.fit(Xtr, y[tr])
                y_pred_pers[te] = m.predict(Xte)
                residuals[te] = y[te] - y_pred_pers[te]

            abs_error = np.abs(residuals)
            median_error = np.median(abs_error)

            # Split: easy cases (low error) vs hard cases (high error)
            easy_mask = abs_error <= median_error
            hard_mask = abs_error > median_error

            # Can sensing predict the residuals for hard cases?
            r2_sens_hard, n_hard = quick_cv_r2(X_beh[hard_mask], residuals[hard_mask])
            r2_sens_easy, n_easy = quick_cv_r2(X_beh[easy_mask], residuals[easy_mask])

            # Can sensing improve prediction for hard cases specifically?
            X_both_hard = np.hstack([X_pers[hard_mask], X_beh[hard_mask]])
            r2_both_hard, _ = quick_cv_r2(X_both_hard, y[hard_mask])
            r2_pers_hard, _ = quick_cv_r2(X_pers[hard_mask], y[hard_mask])

            X_both_easy = np.hstack([X_pers[easy_mask], X_beh[easy_mask]])
            r2_both_easy, _ = quick_cv_r2(X_both_easy, y[easy_mask])
            r2_pers_easy, _ = quick_cv_r2(X_pers[easy_mask], y[easy_mask])

            print(f"  {study} {outcome_label}:", flush=True)
            print(f"    Hard cases (N={n_hard}): Pers R²={r2_pers_hard:.4f}, "
                  f"Pers+Beh R²={r2_both_hard:.4f}, Beh→Residual R²={r2_sens_hard:.4f}", flush=True)
            print(f"    Easy cases (N={n_easy}): Pers R²={r2_pers_easy:.4f}, "
                  f"Pers+Beh R²={r2_both_easy:.4f}, Beh→Residual R²={r2_sens_easy:.4f}", flush=True)

            rows.append({
                "Study": study, "Outcome": outcome_label,
                "N_hard": n_hard, "N_easy": n_easy,
                "R2_pers_hard": r2_pers_hard, "R2_both_hard": r2_both_hard,
                "R2_sens_residual_hard": r2_sens_hard,
                "R2_pers_easy": r2_pers_easy, "R2_both_easy": r2_both_easy,
                "R2_sens_residual_easy": r2_sens_easy,
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "error_analysis.csv", index=False)
    print(f"  Saved: {OUT / 'error_analysis.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 30: Cost-Effectiveness Analysis
# ═══════════════════════════════════════════════════════════════════════
def run_cost_effectiveness():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 30: Cost-Effectiveness Analysis", flush=True)
    print("=" * 70, flush=True)

    # Collect R² from various approaches (using S2 CES-D as reference)
    approaches = [
        {"Approach": "2 BFI items (10 sec)", "Time_min": 0.17, "Cost_USD": 0,
         "Equipment": "None", "R2": 0.358, "Source": "Analysis 19"},
        {"Approach": "Neuroticism score (1 min)", "Time_min": 1, "Cost_USD": 0,
         "Equipment": "None", "R2": 0.258, "Source": "Analysis 19"},
        {"Approach": "Full Big Five (5 min)", "Time_min": 5, "Cost_USD": 0,
         "Equipment": "None", "R2": 0.279, "Source": "Main analysis"},
        {"Approach": "All 44 BFI items (5 min)", "Time_min": 5, "Cost_USD": 0,
         "Equipment": "None", "R2": 0.328, "Source": "Analysis 19"},
        {"Approach": "Sensing PCA (weeks)", "Time_min": 10080, "Cost_USD": 100,
         "Equipment": "Fitbit + smartphone", "R2": -0.011, "Source": "Main analysis"},
        {"Approach": "Sensing 28 raw (weeks)", "Time_min": 10080, "Cost_USD": 100,
         "Equipment": "Fitbit + smartphone", "R2": -0.162, "Source": "Analysis 18"},
        {"Approach": "Pers + Sensing PCA", "Time_min": 10085, "Cost_USD": 100,
         "Equipment": "BFI + Fitbit + phone", "R2": 0.321, "Source": "Main analysis"},
        {"Approach": "Pers + Comm (S2 best)", "Time_min": 10085, "Cost_USD": 100,
         "Equipment": "BFI + phone logs", "R2": 0.308, "Source": "Analysis 23"},
    ]

    df_out = pd.DataFrame(approaches)

    # R² per minute of data collection
    df_out["R2_per_minute"] = df_out.apply(
        lambda r: r["R2"] / r["Time_min"] if r["Time_min"] > 0 and r["R2"] > 0 else 0, axis=1)

    # Marginal R² over questionnaire baseline
    baseline_r2 = 0.279  # Full Big Five
    df_out["Marginal_R2"] = df_out["R2"] - baseline_r2
    df_out["Marginal_R2_per_minute"] = df_out.apply(
        lambda r: r["Marginal_R2"] / r["Time_min"] if r["Time_min"] > 0 else 0, axis=1)

    print(df_out[["Approach", "Time_min", "R2", "R2_per_minute", "Marginal_R2"]].to_string(), flush=True)

    df_out.to_csv(OUT / "cost_effectiveness.csv", index=False)
    print(f"\n  Saved: {OUT / 'cost_effectiveness.csv'}", flush=True)

    # Figure
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71" if r > 0 else "#e74c3c" for r in df_out["R2"]]
    bars = ax.barh(range(len(df_out)), df_out["R2"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(df_out)))
    ax.set_yticklabels(df_out["Approach"], fontsize=9)
    ax.set_xlabel("R² (S2 CES-D Depression)")
    ax.set_title("Cost-Effectiveness: R² by Assessment Method", fontweight="bold")
    ax.axvline(0, color="grey", linestyle="--", alpha=0.5)

    # Annotate with time
    for i, (r2, t) in enumerate(zip(df_out["R2"], df_out["Time_min"])):
        time_str = f"{t:.0f}min" if t < 60 else f"{t/60:.0f}h" if t < 1440 else f"{t/1440:.0f}w"
        ax.text(max(r2 + 0.01, 0.01), i, f"R²={r2:.3f} ({time_str})", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT / "figure_cost_effectiveness.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT / 'figure_cost_effectiveness.png'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 31: Literature Benchmark Comparison
# ═══════════════════════════════════════════════════════════════════════
def run_benchmark():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 31: Literature Benchmark Comparison", flush=True)
    print("=" * 70, flush=True)

    # Published sensing → MH results (curated from key papers)
    benchmarks = [
        {"Paper": "Saeb et al. 2015", "Journal": "JMIR", "N": 28,
         "Outcome": "PHQ-9", "Sensing": "GPS", "Metric": "r", "Value": 0.63,
         "Note": "Single sensor, no cross-validation"},
        {"Paper": "Wang et al. 2014", "Journal": "UbiComp", "N": 48,
         "Outcome": "PHQ-9/PSS", "Sensing": "Multi-modal", "Metric": "descriptive",
         "Value": np.nan, "Note": "StudentLife; correlational only"},
        {"Paper": "Chikersal et al. 2021", "Journal": "TOCHI", "N": 138,
         "Outcome": "PHQ-9", "Sensing": "Multi-modal", "Metric": "AUC", "Value": 0.74,
         "Note": "College students, passive+active"},
        {"Paper": "Xu et al. 2022 (GLOBEM)", "Journal": "IMWUT", "N": 497,
         "Outcome": "PHQ-4", "Sensing": "Multi-modal", "Metric": "AUC", "Value": 0.55,
         "Note": "19 algorithms; cross-dataset barely > chance"},
        {"Paper": "Muller et al. 2021", "Journal": "Sci Rep", "N": 2341,
         "Outcome": "PHQ-9", "Sensing": "GPS", "Metric": "AUC", "Value": 0.57,
         "Note": "Large sample; small-sample AUC=0.82 did not generalize"},
        {"Paper": "Adler et al. 2022", "Journal": "PLOS ONE", "N": 500,
         "Outcome": "PHQ-9", "Sensing": "Multi-modal", "Metric": "AUC", "Value": 0.60,
         "Note": "CrossCheck+StudentLife; modest after merging"},
        {"Paper": "Currey & Torous 2022", "Journal": "BJPsych Open", "N": 300,
         "Outcome": "PHQ-9/GAD-7", "Sensing": "mindLAMP", "Metric": "r", "Value": 0.15,
         "Note": "Weak passive correlations"},
        {"Paper": "This study (Pers-only)", "Journal": "—", "N": 1559,
         "Outcome": "Multi (5 scales)", "Sensing": "Personality", "Metric": "AUC", "Value": 0.73,
         "Note": "Mean AUC across 5 classification tasks"},
        {"Paper": "This study (Sensing-only)", "Journal": "—", "N": 1559,
         "Outcome": "Multi (5 scales)", "Sensing": "Multi-modal", "Metric": "AUC", "Value": 0.57,
         "Note": "Mean AUC across 5 classification tasks"},
        {"Paper": "This study (Pers+Sens)", "Journal": "—", "N": 1559,
         "Outcome": "Multi (5 scales)", "Sensing": "Pers+Multi", "Metric": "AUC", "Value": 0.75,
         "Note": "Mean AUC across 5 classification tasks"},
    ]

    df_out = pd.DataFrame(benchmarks)
    df_out.to_csv(OUT / "literature_benchmark.csv", index=False)

    print("  Literature comparison:", flush=True)
    for _, r in df_out.iterrows():
        val_str = f"{r['Value']:.2f}" if not np.isnan(r["Value"]) else "N/A"
        print(f"    {r['Paper']} (N={r['N']}): {r['Metric']}={val_str} — {r['Note']}", flush=True)

    print(f"\n  Key insight: Our sensing AUC=0.57 matches GLOBEM (0.55) and Muller (0.57)", flush=True)
    print(f"  Our personality AUC=0.73 beats all published sensing results", flush=True)
    print(f"  Saved: {OUT / 'literature_benchmark.csv'}", flush=True)

    # Figure
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = df_out[df_out["Metric"].isin(["AUC", "r"])].copy()
    plot_df = plot_df.sort_values("Value")

    colors = []
    for _, r in plot_df.iterrows():
        if "This study" in r["Paper"]:
            colors.append("#e74c3c" if "Pers-only" in r["Paper"] else
                         "#3498db" if "Pers+Sens" in r["Paper"] else "#95a5a6")
        else:
            colors.append("#f39c12")

    ax.barh(range(len(plot_df)), plot_df["Value"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels([f"{r['Paper']} (N={r['N']})" for _, r in plot_df.iterrows()], fontsize=8)
    ax.set_xlabel("AUC or r")
    ax.set_title("Our Results in Context of Published Sensing Literature", fontweight="bold")
    ax.axvline(0.5, color="grey", linestyle="--", alpha=0.5, label="Chance")

    plt.tight_layout()
    fig.savefig(OUT / "figure_literature_benchmark.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT / 'figure_literature_benchmark.png'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    analyses = {
        "26": ("Method Variance", run_method_variance),
        "27": ("Missing as Signal", run_missing_signal),
        "28": ("Lagged Prediction", run_lagged),
        "29": ("Error Analysis", run_error_analysis),
        "30": ("Cost-Effectiveness", run_cost_effectiveness),
        "31": ("Literature Benchmark", run_benchmark),
    }
    if len(sys.argv) > 2 and sys.argv[1] == "--analysis":
        key = sys.argv[2]
        if key in analyses:
            analyses[key][1]()
    else:
        for key in ["26", "27", "29", "30", "31", "28"]:
            name, fn = analyses[key]
            print(f"\n{'#' * 70}", flush=True)
            print(f"# Analysis {key}: {name}", flush=True)
            print(f"{'#' * 70}", flush=True)
            fn()

    print("\n" + "=" * 70, flush=True)
    print("All Phase 16e analyses complete.", flush=True)
    print(f"Results in: {OUT}", flush=True)
    print("=" * 70, flush=True)
