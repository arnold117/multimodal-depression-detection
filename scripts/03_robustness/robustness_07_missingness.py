#!/usr/bin/env python3
"""
Phase 16f — Remaining Analyses
===============================
32. Missingness pattern prediction (clustering on missing patterns)
33. Weekend vs weekday behavioral shift (social jet lag)
34. Prediction error × sensing rescue (personality worst 20%)
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/robustness")
TRAITS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
RS = 42
COHORTS = ["INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4"]
RAW_DIR = Path("data/raw/globem")

print("Loading datasets...", flush=True)
s3 = pd.read_parquet("data/processed/globem/globem_analysis_dataset.parquet")
S3_BEH_PCA = [c for c in s3.columns if c.endswith("_pc1") and not c.startswith("nh_")]

S3_OUTCOMES = {"bdi2_total": "BDI-II", "stai_state": "STAI", "pss_10": "PSS-10",
               "cesd_total": "CESD", "ucla_loneliness": "UCLA"}

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
# ANALYSIS 32: Missingness Pattern Prediction
# ═══════════════════════════════════════════════════════════════════════
def run_missingness_pattern():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 32: Missingness Pattern Prediction", flush=True)
    print("=" * 70, flush=True)

    MODALITY_FILES = ["steps.csv", "sleep.csv", "screen.csv", "call.csv", "location.csv"]

    print("  Computing per-person missingness patterns...", flush=True)
    miss_features = {}

    for modality_file in MODALITY_FILES:
        mod_name = modality_file.replace(".csv", "")
        all_daily = []
        for cohort in COHORTS:
            fpath = RAW_DIR / cohort / "FeatureData" / modality_file
            if not fpath.exists():
                continue
            df_raw = pd.read_csv(fpath, low_memory=False)
            # Just use first numeric column after pid/date
            num_cols = [c for c in df_raw.columns if c not in ["Unnamed: 0", "pid", "date"]]
            if not num_cols:
                continue
            col = num_cols[0]
            daily = df_raw[["pid", "date", col]].copy()
            daily.columns = ["pid", "date", "value"]
            daily["value"] = pd.to_numeric(daily["value"], errors="coerce")
            daily["date"] = pd.to_datetime(daily["date"])
            all_daily.append(daily)

        if not all_daily:
            continue
        combined = pd.concat(all_daily, ignore_index=True)

        # Per-person: total days, missing days, longest gap, missing streaks
        for pid, grp in combined.groupby("pid"):
            if pid not in miss_features:
                miss_features[pid] = {}
            total = len(grp)
            missing = grp["value"].isna().sum()
            miss_rate = missing / total if total > 0 else 1.0
            miss_features[pid][f"{mod_name}_miss_rate"] = miss_rate
            miss_features[pid][f"{mod_name}_n_days"] = total

            # Longest consecutive missing streak
            is_missing = grp.sort_values("date")["value"].isna().values
            if len(is_missing) > 0:
                max_streak = 0
                current = 0
                for m in is_missing:
                    if m:
                        current += 1
                        max_streak = max(max_streak, current)
                    else:
                        current = 0
                miss_features[pid][f"{mod_name}_max_gap"] = max_streak
            else:
                miss_features[pid][f"{mod_name}_max_gap"] = 0

    feat_df = pd.DataFrame.from_dict(miss_features, orient="index")
    feat_df.index.name = "pid"
    feat_df = feat_df.reset_index()

    s3_scores = s3[["pid"] + list(S3_OUTCOMES.keys()) + TRAITS]
    merged = s3_scores.merge(feat_df, on="pid", how="inner")

    miss_cols = [c for c in feat_df.columns if c != "pid"]
    valid_pers = [c for c in TRAITS if c in merged.columns]

    print(f"  Merged N={len(merged)}, {len(miss_cols)} missingness features", flush=True)

    rows = []
    for outcome_col, outcome_label in S3_OUTCOMES.items():
        if outcome_col not in merged.columns:
            continue
        y = merged[outcome_col].values

        r2_miss, n = quick_cv_r2(merged[miss_cols].values, y)
        r2_pers, _ = quick_cv_r2(merged[valid_pers].values, y)
        r2_both, _ = quick_cv_r2(merged[valid_pers + miss_cols].values, y)

        # Correlations: miss rate × outcome
        miss_rate_cols = [c for c in miss_cols if c.endswith("_miss_rate")]
        sig_corrs = []
        for mc in miss_rate_cols:
            sub = merged[[mc, outcome_col]].dropna()
            if len(sub) > 30:
                r, p = stats.pearsonr(sub[mc], sub[outcome_col])
                if p < 0.05:
                    sig_corrs.append(f"{mc}:r={r:.3f}")

        print(f"  {outcome_label}: Miss R²={r2_miss:.4f}, Pers R²={r2_pers:.4f}, "
              f"Pers+Miss R²={r2_both:.4f} (N={n})", flush=True)
        if sig_corrs:
            print(f"    Sig: {', '.join(sig_corrs)}", flush=True)

        rows.append({
            "Outcome": outcome_label, "N": n,
            "R2_missingness": r2_miss, "R2_personality": r2_pers,
            "R2_pers_plus_miss": r2_both,
            "Sig_correlations": "; ".join(sig_corrs),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "missingness_pattern.csv", index=False)
    print(f"  Saved: {OUT / 'missingness_pattern.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 33: Weekend vs Weekday Shift (Social Jet Lag)
# ═══════════════════════════════════════════════════════════════════════
def run_social_jetlag():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 33: Weekend vs Weekday Behavioral Shift (Social Jet Lag)", flush=True)
    print("=" * 70, flush=True)

    FEATURE_MAP = [
        ("steps.csv", "avgsumsteps:14dhist", "steps"),
        ("sleep.csv", "avgdurationasleepmain:14dhist", "sleep_dur"),
        ("screen.csv", "rapids_sumdurationunlock:14dhist", "screen_dur"),
        ("call.csv", "incoming_count:14dhist", "calls_in"),
        ("location.csv", "barnett_hometime:14dhist", "hometime"),
    ]

    print("  Computing weekday/weekend splits...", flush=True)
    shift_features = {}

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
            daily = df_raw[["pid", "date", col]].copy()
            daily.columns = ["pid", "date", "value"]
            daily["value"] = pd.to_numeric(daily["value"], errors="coerce")
            daily["date"] = pd.to_datetime(daily["date"])
            all_daily.append(daily)

        if not all_daily:
            continue
        combined = pd.concat(all_daily, ignore_index=True).dropna(subset=["value"])
        combined["is_weekend"] = combined["date"].dt.dayofweek >= 5

        # Per person: weekday mean, weekend mean, shift
        weekday = combined[~combined["is_weekend"]].groupby("pid")["value"].mean()
        weekend = combined[combined["is_weekend"]].groupby("pid")["value"].mean()
        common = weekday.index.intersection(weekend.index)

        if len(common) < 50:
            continue

        shift = weekend.loc[common] - weekday.loc[common]
        abs_shift = shift.abs()
        ratio = weekend.loc[common] / weekday.loc[common].clip(lower=0.001)

        shift_features[f"{short_name}_shift"] = shift
        shift_features[f"{short_name}_abs_shift"] = abs_shift
        shift_features[f"{short_name}_wkend_wkday_ratio"] = ratio

    feat_df = pd.DataFrame(shift_features)
    feat_df.index.name = "pid"
    feat_df = feat_df.reset_index()

    s3_scores = s3[["pid"] + list(S3_OUTCOMES.keys()) + TRAITS]
    merged = s3_scores.merge(feat_df, on="pid", how="inner")

    shift_cols = [c for c in shift_features if c in merged.columns]
    valid_pers = [c for c in TRAITS if c in merged.columns]

    print(f"  Merged N={len(merged)}, {len(shift_cols)} shift features", flush=True)

    rows = []
    for outcome_col, outcome_label in S3_OUTCOMES.items():
        if outcome_col not in merged.columns:
            continue
        y = merged[outcome_col].values

        r2_shift, n = quick_cv_r2(merged[shift_cols].values, y)
        r2_pers, _ = quick_cv_r2(merged[valid_pers].values, y)
        r2_both, _ = quick_cv_r2(merged[valid_pers + shift_cols].values, y)

        # Significant correlations
        sig_corrs = []
        for sc in shift_cols:
            sub = merged[[sc, outcome_col]].dropna()
            if len(sub) > 30:
                r, p = stats.pearsonr(sub[sc], sub[outcome_col])
                if p < 0.05:
                    sig_corrs.append(f"{sc}:r={r:.3f}")

        print(f"  {outcome_label}: Shift R²={r2_shift:.4f}, Pers R²={r2_pers:.4f}, "
              f"Pers+Shift R²={r2_both:.4f} (N={n})", flush=True)
        if sig_corrs:
            print(f"    Sig: {', '.join(sig_corrs)}", flush=True)

        rows.append({
            "Outcome": outcome_label, "N": n,
            "R2_shift": r2_shift, "R2_personality": r2_pers,
            "R2_pers_plus_shift": r2_both,
            "Sig_correlations": "; ".join(sig_corrs),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "social_jetlag.csv", index=False)
    print(f"  Saved: {OUT / 'social_jetlag.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 34: Prediction Error × Sensing Rescue (Worst 20%)
# ═══════════════════════════════════════════════════════════════════════
def run_worst20_rescue():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 34: Sensing Rescue for Personality's Worst 20%", flush=True)
    print("=" * 70, flush=True)

    from sklearn.model_selection import KFold

    rows = []
    valid_pers = [c for c in TRAITS if c in s3.columns]
    valid_beh = [c for c in S3_BEH_PCA if c in s3.columns]

    for outcome_col, outcome_label in S3_OUTCOMES.items():
        if outcome_col not in s3.columns:
            continue

        sub = s3[valid_pers + valid_beh + [outcome_col]].dropna()
        if len(sub) < 100:
            continue

        y = sub[outcome_col].values
        X_pers = sub[valid_pers].values
        X_beh = sub[valid_beh].values

        # Get personality residuals via CV
        cv = KFold(n_splits=5, shuffle=True, random_state=RS)
        abs_errors = np.zeros(len(y))

        for tr, te in cv.split(X_pers):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_pers[tr])
            Xte = sc.transform(X_pers[te])
            m = Ridge(alpha=1.0)
            m.fit(Xtr, y[tr])
            abs_errors[te] = np.abs(y[te] - m.predict(Xte))

        # Worst 20% (highest absolute error)
        threshold = np.percentile(abs_errors, 80)
        worst_mask = abs_errors >= threshold
        rest_mask = ~worst_mask

        n_worst = worst_mask.sum()
        n_rest = rest_mask.sum()

        # Sensing prediction for worst 20%
        if n_worst >= 30:
            r2_beh_worst, _ = quick_cv_r2(X_beh[worst_mask], y[worst_mask])
            r2_pers_worst, _ = quick_cv_r2(X_pers[worst_mask], y[worst_mask])
            r2_both_worst, _ = quick_cv_r2(
                np.hstack([X_pers[worst_mask], X_beh[worst_mask]]), y[worst_mask])
        else:
            r2_beh_worst = r2_pers_worst = r2_both_worst = np.nan

        if n_rest >= 30:
            r2_beh_rest, _ = quick_cv_r2(X_beh[rest_mask], y[rest_mask])
            r2_pers_rest, _ = quick_cv_r2(X_pers[rest_mask], y[rest_mask])
            r2_both_rest, _ = quick_cv_r2(
                np.hstack([X_pers[rest_mask], X_beh[rest_mask]]), y[rest_mask])
        else:
            r2_beh_rest = r2_pers_rest = r2_both_rest = np.nan

        print(f"  {outcome_label}:", flush=True)
        print(f"    Worst 20% (N={n_worst}): Pers R²={r2_pers_worst:.4f}, "
              f"Beh R²={r2_beh_worst:.4f}, Both R²={r2_both_worst:.4f}", flush=True)
        print(f"    Rest 80%  (N={n_rest}):  Pers R²={r2_pers_rest:.4f}, "
              f"Beh R²={r2_beh_rest:.4f}, Both R²={r2_both_rest:.4f}", flush=True)

        # What characterizes the worst 20%? Compare personality traits
        for trait in valid_pers:
            t_worst = sub[trait].values[worst_mask]
            t_rest = sub[trait].values[rest_mask]
            t_stat, p_val = stats.ttest_ind(t_worst, t_rest)
            d = (t_worst.mean() - t_rest.mean()) / np.sqrt((t_worst.std()**2 + t_rest.std()**2) / 2)
            if p_val < 0.05:
                print(f"      {trait}: worst={t_worst.mean():.2f} vs rest={t_rest.mean():.2f}, "
                      f"d={d:.3f}, p={p_val:.4f}", flush=True)

        rows.append({
            "Outcome": outcome_label,
            "N_worst20": n_worst, "N_rest80": n_rest,
            "R2_pers_worst": r2_pers_worst, "R2_beh_worst": r2_beh_worst,
            "R2_both_worst": r2_both_worst,
            "R2_pers_rest": r2_pers_rest, "R2_beh_rest": r2_beh_rest,
            "R2_both_rest": r2_both_rest,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "worst20_rescue.csv", index=False)
    print(f"  Saved: {OUT / 'worst20_rescue.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    analyses = {
        "32": ("Missingness Pattern", run_missingness_pattern),
        "33": ("Social Jet Lag", run_social_jetlag),
        "34": ("Worst 20% Rescue", run_worst20_rescue),
    }
    if len(sys.argv) > 2 and sys.argv[1] == "--analysis":
        key = sys.argv[2]
        if key in analyses:
            analyses[key][1]()
    else:
        for key in ["32", "33", "34"]:
            name, fn = analyses[key]
            print(f"\n{'#' * 70}", flush=True)
            print(f"# Analysis {key}: {name}", flush=True)
            print(f"{'#' * 70}", flush=True)
            fn()

    print("\n" + "=" * 70, flush=True)
    print("All Phase 16f analyses complete.", flush=True)
    print("=" * 70, flush=True)
