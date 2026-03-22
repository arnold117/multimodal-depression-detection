#!/usr/bin/env python3
"""
Phase 16g — Complete Data Utilization
======================================
Use ALL remaining data sources:
35. S3 EMA × daily sensing (within-person momentary, ~18 EMA/person)
36. S3 EMA affect prediction (positive/negative affect × sensing)
37. S2 extended outcomes (loneliness, self-esteem, SELSA subscales)
38. S2 full demographic controls (SES, race, parent education)
39. S2 sensing → GPA
40. S1 in supplementary (consistency check at N=28)
41. Grand synthesis: cross-outcome meta-summary
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/comparison/supplementary")
TRAITS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
RS = 42
COHORTS = ["INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4"]
RAW_DIR = Path("data/raw/globem")

print("Loading datasets...", flush=True)
s1 = pd.read_parquet("data/processed/analysis_dataset.parquet")
s2 = pd.read_parquet("data/processed/nethealth/nethealth_analysis_dataset.parquet")
s3 = pd.read_parquet("data/processed/globem/globem_analysis_dataset.parquet")

S1_BEH_PCA = [c for c in s1.columns if c.endswith("_pc1")]
S2_BEH_PCA = [c for c in s2.columns if c.startswith("nh_") and c.endswith("_pc1")]
S3_BEH_PCA = [c for c in s3.columns if c.endswith("_pc1") and not c.startswith("nh_")]

def find_col(columns, substring):
    matches = [c for c in columns if substring in c]
    return min(matches, key=len) if matches else None

def quick_cv_r2(X, y, n_splits=5, n_repeats=5, alpha=1.0):
    """Quick Ridge CV, returns dict with R2_mean, R2_ci_lo, R2_ci_hi, N."""
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X_c, y_c = X[mask], y[mask]
    if len(y_c) < 20:
        return {"R2_mean": np.nan, "R2_ci_lo": np.nan, "R2_ci_hi": np.nan, "N": len(y_c)}
    ns = min(n_splits, len(y_c) // 3)  # small-N guard for S1
    if ns < 2:
        return {"R2_mean": np.nan, "R2_ci_lo": np.nan, "R2_ci_hi": np.nan, "N": len(y_c)}
    cv = RepeatedKFold(n_splits=ns, n_repeats=n_repeats, random_state=RS)
    r2s = []
    for tr, te in cv.split(X_c):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_c[tr])
        Xte = sc.transform(X_c[te])
        m = Ridge(alpha=alpha)
        m.fit(Xtr, y_c[tr])
        r2s.append(r2_score(y_c[te], m.predict(Xte)))
    r2s = np.array(r2s)
    return {
        "R2_mean": float(np.mean(r2s)),
        "R2_ci_lo": float(np.percentile(r2s, 2.5)),
        "R2_ci_hi": float(np.percentile(r2s, 97.5)),
        "N": len(y_c),
    }


DAILY_FEATURES = [
    ("steps.csv", "avgsumsteps:14dhist", "steps"),
    ("sleep.csv", "avgdurationasleepmain:14dhist", "sleep_dur"),
    ("sleep.csv", "avgefficiencymain:14dhist", "sleep_eff"),
    ("screen.csv", "rapids_sumdurationunlock:14dhist", "screen_dur"),
    ("call.csv", "incoming_count:14dhist", "calls_in"),
    ("location.csv", "barnett_hometime:14dhist", "hometime"),
]

def load_daily_sensing():
    """Load daily sensing features for all cohorts."""
    daily = {}
    for modality_file, col_substr, short_name in DAILY_FEATURES:
        frames = []
        for cohort in COHORTS:
            fpath = RAW_DIR / cohort / "FeatureData" / modality_file
            if not fpath.exists():
                continue
            df_raw = pd.read_csv(fpath, low_memory=False)
            col = find_col(df_raw.columns.tolist(), col_substr)
            if col is None:
                continue
            d = df_raw[["pid", "date", col]].copy()
            d.columns = ["pid", "date", "value"]
            d["value"] = pd.to_numeric(d["value"], errors="coerce")
            d["date"] = pd.to_datetime(d["date"])
            frames.append(d)
        if frames:
            daily[short_name] = pd.concat(frames, ignore_index=True)
    return daily


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 35: S3 EMA × Daily Sensing (Within-Person Momentary)
# ═══════════════════════════════════════════════════════════════════════
def run_ema_sensing():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 35: S3 EMA × Daily Sensing (Within-Person Momentary)", flush=True)
    print("=" * 70, flush=True)

    # Load EMA (W2-W4 have full data)
    all_ema = []
    for cohort in ["INS-W_2", "INS-W_3", "INS-W_4"]:
        try:
            ema = pd.read_csv(RAW_DIR / cohort / "SurveyData" / "ema.csv")
            ema["cohort"] = cohort
            all_ema.append(ema)
        except Exception:
            pass

    ema_df = pd.concat(all_ema, ignore_index=True)
    ema_df["date"] = pd.to_datetime(ema_df["date"])
    ema_cols = ["phq4_EMA", "pss4_EMA", "positive_affect_EMA", "negative_affect_EMA"]
    for c in ema_cols:
        ema_df[c] = pd.to_numeric(ema_df[c], errors="coerce")

    print(f"  EMA: {len(ema_df)} obs, {ema_df['pid'].nunique()} participants", flush=True)

    # Load daily sensing
    daily = load_daily_sensing()
    feat_names = list(daily.keys())

    # Build EMA × concurrent sensing panel
    print("  Building EMA × sensing panel...", flush=True)
    panel_rows = []

    for pid in ema_df["pid"].unique():
        ema_pid = ema_df[ema_df["pid"] == pid].sort_values("date")

        for _, row in ema_pid.iterrows():
            ema_date = row["date"]
            rec = {"pid": pid}
            for ec in ema_cols:
                rec[ec] = row.get(ec, np.nan)

            # Get sensing for ±3 days around EMA
            has_data = True
            for feat_name, daily_df in daily.items():
                sens_pid = daily_df[daily_df["pid"] == pid]
                window = sens_pid[(sens_pid["date"] >= ema_date - pd.Timedelta(days=3)) &
                                   (sens_pid["date"] <= ema_date + pd.Timedelta(days=3))]
                if len(window) >= 1:
                    rec[feat_name] = window["value"].mean()
                else:
                    has_data = False
                    break

            if has_data:
                panel_rows.append(rec)

    panel = pd.DataFrame(panel_rows)
    panel = panel.dropna(subset=feat_names + ["phq4_EMA"])
    print(f"  Panel: {len(panel)} obs, {panel['pid'].nunique()} persons", flush=True)

    # Merge personality
    s3_pers = s3[["pid"] + TRAITS].drop_duplicates("pid")
    panel = panel.merge(s3_pers, on="pid", how="inner")
    valid_pers = [c for c in TRAITS if c in panel.columns]

    rows = []
    ema_outcomes = {
        "phq4_EMA": "PHQ-4 (EMA)", "pss4_EMA": "PSS-4 (EMA)",
        "positive_affect_EMA": "Positive Affect", "negative_affect_EMA": "Negative Affect",
    }

    for ema_col, ema_label in ema_outcomes.items():
        if ema_col not in panel.columns:
            continue
        panel[ema_col] = pd.to_numeric(panel[ema_col], errors="coerce")
        y = panel[ema_col].values
        if np.isnan(y).all():
            continue

        # Between-person: pooled
        _res_sens = quick_cv_r2(panel[feat_names].values, y, n_splits=5, n_repeats=3)
        r2_sens, n = _res_sens["R2_mean"], _res_sens["N"]
        _res_pers = quick_cv_r2(panel[valid_pers].values, y, n_splits=5, n_repeats=3)
        r2_pers = _res_pers["R2_mean"]
        _res_both = quick_cv_r2(panel[valid_pers + feat_names].values, y, n_splits=5, n_repeats=3)
        r2_both = _res_both["R2_mean"]

        # Within-person: person-mean centered
        panel_c = panel.copy()
        pmeans = panel.groupby("pid")[[ema_col] + feat_names].transform("mean")
        y_c = (panel[ema_col] - pmeans[ema_col]).values
        X_c = (panel[feat_names] - pmeans[feat_names]).values
        _res_within = quick_cv_r2(X_c, y_c, n_splits=5, n_repeats=3)
        r2_within = _res_within["R2_mean"]

        # Per-person correlations
        within_rs = []
        for pid, grp in panel.groupby("pid"):
            if len(grp) < 5:
                continue
            for feat in feat_names:
                vals = grp[[feat, ema_col]].dropna()
                if len(vals) >= 5:
                    r, p = stats.pearsonr(vals[feat], vals[ema_col])
                    if not np.isnan(r):
                        within_rs.append({"pid": pid, "feature": feat, "r": r, "p": p})

        within_df = pd.DataFrame(within_rs)
        if len(within_df) > 0:
            mean_abs_r = within_df["r"].abs().mean()
            n_sig = (within_df["p"] < 0.05).sum()
            pct_sig = n_sig / len(within_df)
        else:
            mean_abs_r = np.nan
            n_sig = 0
            pct_sig = 0

        print(f"  {ema_label} (N_obs={n}):", flush=True)
        print(f"    Between: Sens R²={r2_sens:.4f}, Pers R²={r2_pers:.4f}, Both R²={r2_both:.4f}", flush=True)
        print(f"    Within (centered): R²={r2_within:.4f}", flush=True)
        print(f"    Per-person: mean |r|={mean_abs_r:.4f}, sig={n_sig}/{len(within_df)} ({pct_sig:.1%})", flush=True)

        rows.append({
            "Outcome": ema_label, "N_obs": n, "N_persons": panel["pid"].nunique(),
            "R2_sensing_between": r2_sens, "R2_personality_between": r2_pers,
            "R2_combined_between": r2_both, "R2_within_person": r2_within,
            "Mean_abs_within_r": mean_abs_r, "Pct_sig_within": pct_sig,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "ema_sensing.csv", index=False)
    print(f"  Saved: {OUT / 'ema_sensing.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 37: S2 Extended Outcomes
# ═══════════════════════════════════════════════════════════════════════
def run_s2_extended():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 37: S2 Extended Outcomes (Loneliness, Self-Esteem, SELSA)", flush=True)
    print("=" * 70, flush=True)

    valid_pers = [c for c in TRAITS if c in s2.columns]
    valid_beh = [c for c in S2_BEH_PCA if c in s2.columns]

    extended = {
        "loneliness_total": "Loneliness", "self_esteem_total": "Self-Esteem",
        "selsa_romantic": "SELSA Romantic", "selsa_family": "SELSA Family",
        "selsa_social": "SELSA Social",
    }

    rows = []
    for col, label in extended.items():
        if col not in s2.columns:
            continue
        y = s2[col].values

        _res_pers = quick_cv_r2(s2[valid_pers].values, y)
        r2_pers, n = _res_pers["R2_mean"], _res_pers["N"]
        _res_beh = quick_cv_r2(s2[valid_beh].values, y)
        r2_beh = _res_beh["R2_mean"]
        _res_both = quick_cv_r2(s2[valid_pers + valid_beh].values, y)
        r2_both = _res_both["R2_mean"]

        # Top personality predictor
        sub = s2[valid_pers + [col]].dropna()
        best_trait, best_r = "", 0
        for t in valid_pers:
            r, _ = stats.pearsonr(sub[t], sub[col])
            if abs(r) > abs(best_r):
                best_r = r
                best_trait = t

        print(f"  {label} (N={n}): Pers R²={r2_pers:.4f}, Beh R²={r2_beh:.4f}, "
              f"Both R²={r2_both:.4f}, Top={best_trait} (r={best_r:.3f})", flush=True)

        rows.append({
            "Outcome": label, "N": n,
            "R2_personality": r2_pers, "R2_sensing": r2_beh, "R2_combined": r2_both,
            "Top_trait": best_trait, "Top_r": best_r,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "s2_extended_outcomes.csv", index=False)
    print(f"  Saved: {OUT / 's2_extended_outcomes.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 38: S2 Full Demographic Controls
# ═══════════════════════════════════════════════════════════════════════
def run_demographics():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 38: S2 Full Demographic Controls (SES, Race, Education)", flush=True)
    print("=" * 70, flush=True)

    basic = pd.read_csv("data/raw/nethealth/BasicSurvey(3-6-20).csv", low_memory=False)

    # Encode demographics
    basic["female"] = (basic["gender_1"] == "Female").astype(float)
    basic["native"] = (basic["native_1"] == "native").astype(float)
    basic["parent_grad"] = basic["momeduc_1"].apply(
        lambda x: 1.0 if "Graduate" in str(x) else 0.0)
    basic["high_income"] = basic["parentincome_1"].apply(
        lambda x: 1.0 if "$250,000" in str(x) or "$200,000" in str(x) else 0.0)

    demo_cols = ["female", "native", "parent_grad", "high_income"]
    basic_clean = basic[["egoid"] + demo_cols].dropna()

    s2_demo = s2.merge(basic_clean, on="egoid", how="inner")
    print(f"  S2 with full demographics: {len(s2_demo)}", flush=True)

    valid_pers = [c for c in TRAITS if c in s2_demo.columns]
    valid_beh = [c for c in S2_BEH_PCA if c in s2_demo.columns]
    outcomes = {"cesd_total": "CES-D", "stai_trait_total": "STAI", "bai_total": "BAI"}

    rows = []
    for col, label in outcomes.items():
        if col not in s2_demo.columns:
            continue

        all_cols = demo_cols + valid_pers + valid_beh + [col]
        sub = s2_demo[all_cols].dropna()
        if len(sub) < 50:
            continue
        y = sub[col].values
        n = len(sub)

        # Hierarchical: Demo → +Pers → +Beh
        X1 = sm.add_constant(sub[demo_cols].values)
        X2 = sm.add_constant(sub[demo_cols + valid_pers].values)
        X3 = sm.add_constant(sub[demo_cols + valid_pers + valid_beh].values)

        m1 = sm.OLS(y, X1).fit()
        m2 = sm.OLS(y, X2).fit()
        m3 = sm.OLS(y, X3).fit()

        dr2_pers = m2.rsquared - m1.rsquared
        dr2_beh = m3.rsquared - m2.rsquared

        # F-tests
        df_num_pers = len(valid_pers)
        df_den_pers = n - X2.shape[1]
        f_pers = (dr2_pers / df_num_pers) / ((1 - m2.rsquared) / df_den_pers)
        p_pers = 1 - stats.f.cdf(f_pers, df_num_pers, df_den_pers)

        df_num_beh = len(valid_beh)
        df_den_beh = n - X3.shape[1]
        f_beh = (dr2_beh / df_num_beh) / ((1 - m3.rsquared) / df_den_beh)
        p_beh = 1 - stats.f.cdf(f_beh, df_num_beh, df_den_beh)

        print(f"  {label} (N={n}):", flush=True)
        print(f"    Demo R²={m1.rsquared:.4f}", flush=True)
        print(f"    +Pers R²={m2.rsquared:.4f} (ΔR²={dr2_pers:.4f}, p={p_pers:.4f})", flush=True)
        print(f"    +Beh  R²={m3.rsquared:.4f} (ΔR²={dr2_beh:.4f}, p={p_beh:.4f})", flush=True)

        rows.append({
            "Outcome": label, "N": n,
            "R2_demographics": m1.rsquared,
            "R2_demo_pers": m2.rsquared, "DR2_personality": dr2_pers, "p_personality": p_pers,
            "R2_demo_pers_beh": m3.rsquared, "DR2_behavior": dr2_beh, "p_behavior": p_beh,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "full_demographics.csv", index=False)
    print(f"  Saved: {OUT / 'full_demographics.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 39: S2 Sensing → GPA
# ═══════════════════════════════════════════════════════════════════════
def run_sensing_gpa():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 39: S2 Sensing → GPA", flush=True)
    print("=" * 70, flush=True)

    valid_pers = [c for c in TRAITS if c in s2.columns]
    valid_beh = [c for c in S2_BEH_PCA if c in s2.columns]

    y = s2["gpa_overall"].values
    _res_pers = quick_cv_r2(s2[valid_pers].values, y)
    r2_pers, n = _res_pers["R2_mean"], _res_pers["N"]
    _res_beh = quick_cv_r2(s2[valid_beh].values, y)
    r2_beh = _res_beh["R2_mean"]
    _res_both = quick_cv_r2(s2[valid_pers + valid_beh].values, y)
    r2_both = _res_both["R2_mean"]

    # Conscientiousness only
    _res_c = quick_cv_r2(s2[["conscientiousness"]].values, y)
    r2_c = _res_c["R2_mean"]

    print(f"  GPA (N={n}):", flush=True)
    print(f"    C only R²={r2_c:.4f}, Pers R²={r2_pers:.4f}, "
          f"Beh R²={r2_beh:.4f}, Both R²={r2_both:.4f}", flush=True)

    results = {"N": n, "R2_conscientiousness": r2_c, "R2_personality": r2_pers,
               "R2_sensing": r2_beh, "R2_combined": r2_both}
    pd.DataFrame([results]).to_csv(OUT / "sensing_gpa.csv", index=False)
    print(f"  Saved: {OUT / 'sensing_gpa.csv'}", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 40: S1 Consistency Check
# ═══════════════════════════════════════════════════════════════════════
def run_s1_check():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 40: S1 Consistency Check (N=28)", flush=True)
    print("=" * 70, flush=True)

    valid_pers = [c for c in TRAITS if c in s1.columns]
    valid_beh = [c for c in S1_BEH_PCA if c in s1.columns]

    outcomes = {
        "phq9_total": "PHQ-9", "pss_total": "PSS", "loneliness_total": "Loneliness",
        "flourishing_total": "Flourishing", "gpa_overall": "GPA",
    }

    rows = []
    for col, label in outcomes.items():
        if col not in s1.columns:
            continue
        y = s1[col].values

        # LOO-CV (N=28 too small for k-fold)
        from sklearn.model_selection import LeaveOneOut
        loo = LeaveOneOut()

        for feat_name, feat_cols in [("Personality", valid_pers), ("Sensing", valid_beh),
                                      ("Combined", valid_pers + valid_beh)]:
            X = s1[feat_cols].values
            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X_c, y_c = X[mask], y[mask]
            if len(y_c) < 10:
                continue

            preds = []
            for tr, te in loo.split(X_c):
                sc = StandardScaler()
                Xtr = sc.fit_transform(X_c[tr])
                Xte = sc.transform(X_c[te])
                m = Ridge(alpha=1.0)
                m.fit(Xtr, y_c[tr])
                preds.append((y_c[te[0]], m.predict(Xte)[0]))

            y_true = [p[0] for p in preds]
            y_pred = [p[1] for p in preds]
            r2 = r2_score(y_true, y_pred)

            rows.append({"Outcome": label, "Features": feat_name, "N": len(y_c), "R2_LOO": r2})

        # Print
        sub = [r for r in rows if r["Outcome"] == label]
        if sub:
            p = next((r["R2_LOO"] for r in sub if r["Features"] == "Personality"), np.nan)
            b = next((r["R2_LOO"] for r in sub if r["Features"] == "Sensing"), np.nan)
            c = next((r["R2_LOO"] for r in sub if r["Features"] == "Combined"), np.nan)
            print(f"  {label} (N={sub[0]['N']}): Pers={p:.4f}, Sens={b:.4f}, Both={c:.4f}", flush=True)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "s1_consistency.csv", index=False)
    print(f"  Saved: {OUT / 's1_consistency.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 41: Grand Synthesis — Cross-Outcome Meta-Summary
# ═══════════════════════════════════════════════════════════════════════
def run_grand_synthesis():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 41: Grand Synthesis — All Studies × All Outcomes", flush=True)
    print("=" * 70, flush=True)

    # Collect ALL personality vs sensing R² across every outcome
    all_comparisons = []

    # S1 (LOO-CV)
    s1_outcomes = {"phq9_total": "PHQ-9", "pss_total": "PSS",
                   "loneliness_total": "Loneliness", "gpa_overall": "GPA"}
    valid_pers = [c for c in TRAITS if c in s1.columns]
    valid_beh = [c for c in S1_BEH_PCA if c in s1.columns]
    for col, label in s1_outcomes.items():
        if col not in s1.columns:
            continue
        _res_p = quick_cv_r2(s1[valid_pers].values, s1[col].values, n_splits=3, n_repeats=3)
        _res_b = quick_cv_r2(s1[valid_beh].values, s1[col].values, n_splits=3, n_repeats=3)
        r2_p, n = _res_p["R2_mean"], _res_p["N"]
        r2_b = _res_b["R2_mean"]
        all_comparisons.append({
            "Study": "S1", "Outcome": label,
            "Domain": "GPA" if "gpa" in col else "MH",
            "N": _res_p["N"],
            "R2_personality": _res_p["R2_mean"],
            "R2_pers_ci_lo": _res_p["R2_ci_lo"],
            "R2_pers_ci_hi": _res_p["R2_ci_hi"],
            "R2_sensing": _res_b["R2_mean"],
            "R2_sens_ci_lo": _res_b["R2_ci_lo"],
            "R2_sens_ci_hi": _res_b["R2_ci_hi"],
            "Pers_wins": _res_p["R2_mean"] > _res_b["R2_mean"],
            "Delta": _res_p["R2_mean"] - _res_b["R2_mean"],
        })

    # S2
    s2_outcomes = {"cesd_total": "CES-D", "stai_trait_total": "STAI-Trait", "bai_total": "BAI",
                   "loneliness_total": "Loneliness", "self_esteem_total": "Self-Esteem",
                   "gpa_overall": "GPA"}
    valid_pers = [c for c in TRAITS if c in s2.columns]
    valid_beh = [c for c in S2_BEH_PCA if c in s2.columns]
    for col, label in s2_outcomes.items():
        if col not in s2.columns:
            continue
        _res_p = quick_cv_r2(s2[valid_pers].values, s2[col].values)
        _res_b = quick_cv_r2(s2[valid_beh].values, s2[col].values)
        r2_p, n = _res_p["R2_mean"], _res_p["N"]
        r2_b = _res_b["R2_mean"]
        all_comparisons.append({
            "Study": "S2", "Outcome": label,
            "Domain": "GPA" if "gpa" in col else "MH",
            "N": _res_p["N"],
            "R2_personality": _res_p["R2_mean"],
            "R2_pers_ci_lo": _res_p["R2_ci_lo"],
            "R2_pers_ci_hi": _res_p["R2_ci_hi"],
            "R2_sensing": _res_b["R2_mean"],
            "R2_sens_ci_lo": _res_b["R2_ci_lo"],
            "R2_sens_ci_hi": _res_b["R2_ci_hi"],
            "Pers_wins": _res_p["R2_mean"] > _res_b["R2_mean"],
            "Delta": _res_p["R2_mean"] - _res_b["R2_mean"],
        })

    # S3
    s3_outcomes = {"bdi2_total": "BDI-II", "stai_state": "STAI-State", "pss_10": "PSS-10",
                   "cesd_total": "CESD", "ucla_loneliness": "UCLA"}
    valid_pers = [c for c in TRAITS if c in s3.columns]
    valid_beh = [c for c in S3_BEH_PCA if c in s3.columns]
    for col, label in s3_outcomes.items():
        if col not in s3.columns:
            continue
        _res_p = quick_cv_r2(s3[valid_pers].values, s3[col].values)
        _res_b = quick_cv_r2(s3[valid_beh].values, s3[col].values)
        r2_p, n = _res_p["R2_mean"], _res_p["N"]
        r2_b = _res_b["R2_mean"]
        all_comparisons.append({
            "Study": "S3", "Outcome": label,
            "Domain": "MH",
            "N": _res_p["N"],
            "R2_personality": _res_p["R2_mean"],
            "R2_pers_ci_lo": _res_p["R2_ci_lo"],
            "R2_pers_ci_hi": _res_p["R2_ci_hi"],
            "R2_sensing": _res_b["R2_mean"],
            "R2_sens_ci_lo": _res_b["R2_ci_lo"],
            "R2_sens_ci_hi": _res_b["R2_ci_hi"],
            "Pers_wins": _res_p["R2_mean"] > _res_b["R2_mean"],
            "Delta": _res_p["R2_mean"] - _res_b["R2_mean"],
        })

    df_all = pd.DataFrame(all_comparisons)

    df_all.to_csv(OUT / "grand_synthesis.csv", index=False)

    print(f"\n  Total comparisons: {len(df_all)}", flush=True)
    print(f"  Personality wins: {df_all['Pers_wins'].sum()}/{len(df_all)} "
          f"({df_all['Pers_wins'].mean():.0%})", flush=True)
    print(f"  Mean Pers R²: {df_all['R2_personality'].mean():.4f}", flush=True)
    print(f"  Mean Sens R²: {df_all['R2_sensing'].mean():.4f}", flush=True)
    print(f"  Mean advantage: {df_all['Delta'].mean():.4f}", flush=True)

    print(f"\n  By domain:", flush=True)
    for domain in df_all["Domain"].unique():
        sub = df_all[df_all["Domain"] == domain]
        print(f"    {domain}: Pers R²={sub['R2_personality'].mean():.4f}, "
              f"Sens R²={sub['R2_sensing'].mean():.4f}, "
              f"Pers wins {sub['Pers_wins'].sum()}/{len(sub)}", flush=True)

    print(f"\n  Full table:", flush=True)
    print(df_all[["Study", "Outcome", "N", "R2_personality", "R2_sensing", "Delta"]].to_string(), flush=True)

    # Figure: paired comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    labels = [f"{r['Study']} {r['Outcome']}" for _, r in df_all.iterrows()]
    x = np.arange(len(labels))
    w = 0.35

    ax.bar(x - w/2, df_all["R2_personality"], w, label="Personality", color="#e74c3c")
    ax.bar(x + w/2, df_all["R2_sensing"], w, label="Sensing", color="#95a5a6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("R² (Cross-Validated)")
    ax.set_title(f"Grand Synthesis: Personality vs Sensing Across All {len(df_all)} Outcomes\n"
                 f"Personality wins {df_all['Pers_wins'].sum()}/{len(df_all)} comparisons",
                 fontsize=12, fontweight="bold")
    ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)

    plt.tight_layout()
    fig.savefig(OUT / "figure_grand_synthesis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {OUT / 'figure_grand_synthesis.png'}", flush=True)
    print(f"  Saved: {OUT / 'grand_synthesis.csv'}", flush=True)
    return df_all


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    analyses = {
        "37": ("S2 Extended Outcomes", run_s2_extended),
        "38": ("Full Demographics", run_demographics),
        "39": ("Sensing → GPA", run_sensing_gpa),
        "40": ("S1 Consistency", run_s1_check),
        "35": ("EMA × Sensing", run_ema_sensing),
        "41": ("Grand Synthesis", run_grand_synthesis),
    }
    if len(sys.argv) > 2 and sys.argv[1] == "--analysis":
        key = sys.argv[2]
        if key in analyses:
            analyses[key][1]()
    else:
        for key in ["37", "38", "39", "40", "35", "41"]:
            name, fn = analyses[key]
            print(f"\n{'#' * 70}", flush=True)
            print(f"# Analysis {key}: {name}", flush=True)
            print(f"{'#' * 70}", flush=True)
            fn()

    print("\n" + "=" * 70, flush=True)
    print("All Phase 16g analyses complete.", flush=True)
    print("=" * 70, flush=True)
