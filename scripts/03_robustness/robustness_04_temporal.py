#!/usr/bin/env python3
"""
Phase 16c — Final Supplementary Analyses
=========================================
15. Prospective prediction (Pre sensing → Post MH)
16. Within-person daily correlation (S3 weekly PHQ-4 × daily sensing)
17. Temporal autocorrelation / inertia features
18. S2 raw features (non-PCA) full comparison
19. Item-level prediction (BFI best 2-3 items vs full scale vs sensing)
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/robustness")
OUT.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
RS = 42
COHORTS = ["INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4"]
RAW_DIR = Path("data/raw/globem")


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
# Load datasets
# ═══════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 15: Prospective Prediction (Pre sensing → Post MH)
# ═══════════════════════════════════════════════════════════════════════
def run_prospective():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 15: Prospective Prediction (Sensing → Post MH)", flush=True)
    print("=" * 70, flush=True)

    # Load pre/post surveys for all cohorts
    pre_post_pairs = [
        ("BDI2_PRE", "BDI2_POST", "BDI-II"),
        ("STAI_PRE", "STAI_POST", "STAI"),
        ("PSS_10items_PRE", "PSS_10items_POST", "PSS-10"),
        ("CESD_9items_PRE", "CESD_9items_POST", "CESD"),
        ("UCLA_10items_PRE", "UCLA_10items_POST", "UCLA"),
    ]

    all_pre = []
    all_post = []
    for cohort in COHORTS:
        try:
            pre = pd.read_csv(RAW_DIR / cohort / "SurveyData" / "pre.csv", low_memory=False)
            post = pd.read_csv(RAW_DIR / cohort / "SurveyData" / "post.csv", low_memory=False)
            pre["cohort"] = cohort
            post["cohort"] = cohort
            all_pre.append(pre)
            all_post.append(post)
        except Exception:
            pass

    pre_df = pd.concat(all_pre, ignore_index=True)
    post_df = pd.concat(all_post, ignore_index=True)

    # Merge with S3 (personality + sensing)
    s3_full = s3.copy()
    pre_cols = ["pid"] + [p[0] for p in pre_post_pairs if p[0] in pre_df.columns]
    post_cols = ["pid"] + [p[1] for p in pre_post_pairs if p[1] in post_df.columns]

    merged = s3_full.merge(pre_df[pre_cols].drop_duplicates("pid"), on="pid", how="inner")
    merged = merged.merge(post_df[post_cols].drop_duplicates("pid"), on="pid", how="inner")
    print(f"  Merged N={len(merged)}", flush=True)

    valid_pers = [c for c in TRAITS if c in merged.columns]
    valid_beh = [c for c in S3_BEH_PCA if c in merged.columns]
    rows = []

    for pre_col, post_col, label in pre_post_pairs:
        if pre_col not in merged.columns or post_col not in merged.columns:
            continue

        # Convert to numeric
        merged[pre_col] = pd.to_numeric(merged[pre_col], errors="coerce")
        merged[post_col] = pd.to_numeric(merged[post_col], errors="coerce")

        y_post = merged[post_col].values
        y_pre = merged[pre_col].values
        y_change = y_post - y_pre

        # Predict POST outcome (controlling for PRE = prospective)
        # Model A: PRE only (autoregressive baseline)
        X_pre = merged[[pre_col]].values
        r2_auto, n = quick_cv_r2(X_pre, y_post)

        # Model B: PRE + Personality
        X_pre_pers = merged[[pre_col] + valid_pers].values
        r2_pre_pers, _ = quick_cv_r2(X_pre_pers, y_post)

        # Model C: PRE + Sensing
        X_pre_beh = merged[[pre_col] + valid_beh].values
        r2_pre_beh, _ = quick_cv_r2(X_pre_beh, y_post)

        # Model D: PRE + Personality + Sensing
        X_all = merged[[pre_col] + valid_pers + valid_beh].values
        r2_all, _ = quick_cv_r2(X_all, y_post)

        # Predict CHANGE (post - pre)
        r2_pers_change, _ = quick_cv_r2(merged[valid_pers].values, y_change)
        r2_beh_change, _ = quick_cv_r2(merged[valid_beh].values, y_change)

        print(f"  {label} (N={n}):", flush=True)
        print(f"    Post: Auto={r2_auto:.4f}, +Pers={r2_pre_pers:.4f}, "
              f"+Beh={r2_pre_beh:.4f}, +Both={r2_all:.4f}", flush=True)
        print(f"    Change: Pers={r2_pers_change:.4f}, Beh={r2_beh_change:.4f}", flush=True)

        rows.append({
            "Outcome": label, "N": n,
            "R2_autoregressive": r2_auto,
            "R2_auto_plus_pers": r2_pre_pers,
            "R2_auto_plus_beh": r2_pre_beh,
            "R2_auto_plus_both": r2_all,
            "R2_pers_predict_change": r2_pers_change,
            "R2_beh_predict_change": r2_beh_change,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "prospective_prediction.csv", index=False)
    print(f"  Saved: {OUT / 'prospective_prediction.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 16: Within-Person Daily Correlation (PHQ-4 × Sensing)
# ═══════════════════════════════════════════════════════════════════════
def run_within_person():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 16: Within-Person Correlation (Weekly PHQ-4 × Sensing)", flush=True)
    print("=" * 70, flush=True)

    # Load weekly PHQ-4 for all cohorts
    all_weekly = []
    for cohort in COHORTS:
        try:
            dep = pd.read_csv(RAW_DIR / cohort / "SurveyData" / "dep_weekly.csv")
            dep["cohort"] = cohort
            all_weekly.append(dep)
        except Exception:
            pass

    weekly = pd.concat(all_weekly, ignore_index=True)
    weekly["phq4"] = pd.to_numeric(weekly["phq4"], errors="coerce")
    weekly["date"] = pd.to_datetime(weekly["date"])
    weekly = weekly.dropna(subset=["phq4"])
    print(f"  Weekly PHQ-4: {len(weekly)} obs, {weekly['pid'].nunique()} participants", flush=True)

    # Load daily sensing features
    FEATURE_MAP = [
        ("steps.csv", "avgsumsteps:14dhist", "steps"),
        ("sleep.csv", "avgdurationasleepmain:14dhist", "sleep_dur"),
        ("sleep.csv", "avgefficiencymain:14dhist", "sleep_eff"),
        ("screen.csv", "rapids_sumdurationunlock:14dhist", "screen_dur"),
        ("call.csv", "incoming_count:14dhist", "calls_in"),
        ("location.csv", "barnett_hometime:14dhist", "hometime"),
    ]

    def find_col(columns, substring):
        matches = [c for c in columns if substring in c]
        return min(matches, key=len) if matches else None

    daily_features = {}
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
        if all_daily:
            daily_features[short_name] = pd.concat(all_daily, ignore_index=True)

    print(f"  Loaded {len(daily_features)} daily sensing features", flush=True)

    # For each person: correlate weekly PHQ-4 with weekly-averaged sensing
    rows = []
    for feat_name, daily_df in daily_features.items():
        within_rs = []

        # Aggregate daily sensing to weekly (match PHQ-4 dates)
        for pid in weekly["pid"].unique():
            phq_pid = weekly[weekly["pid"] == pid].sort_values("date")
            sens_pid = daily_df[daily_df["pid"] == pid].sort_values("date")

            if len(phq_pid) < 4 or len(sens_pid) < 7:
                continue

            # For each PHQ-4 date, get average sensing from preceding 7 days
            phq_vals = []
            sens_vals = []
            for _, row in phq_pid.iterrows():
                phq_date = row["date"]
                week_sens = sens_pid[(sens_pid["date"] >= phq_date - pd.Timedelta(days=7)) &
                                     (sens_pid["date"] <= phq_date)]
                if len(week_sens) >= 2:
                    phq_vals.append(row["phq4"])
                    sens_vals.append(week_sens["value"].mean())

            if len(phq_vals) >= 4:
                r, p = stats.pearsonr(phq_vals, sens_vals)
                if not np.isnan(r):
                    within_rs.append(r)

        if within_rs:
            mean_r = float(np.mean(within_rs))
            median_r = float(np.median(within_rs))
            n_sig = sum(1 for r in within_rs if abs(r) > 0.5)
            print(f"  {feat_name}: mean within-r={mean_r:.4f}, median={median_r:.4f}, "
                  f"N_persons={len(within_rs)}, N(|r|>0.5)={n_sig}", flush=True)

            rows.append({
                "Feature": feat_name, "N_persons": len(within_rs),
                "Mean_within_r": mean_r, "Median_within_r": median_r,
                "SD_within_r": float(np.std(within_rs)),
                "N_strong_r": n_sig, "Pct_strong": n_sig / len(within_rs),
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "within_person_correlation.csv", index=False)
    print(f"  Saved: {OUT / 'within_person_correlation.csv'}", flush=True)

    # Figure
    if len(df_out) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(df_out))
        ax.bar(x, df_out["Mean_within_r"], color="#3498db", edgecolor="white")
        ax.errorbar(x, df_out["Mean_within_r"], yerr=df_out["SD_within_r"],
                    fmt="none", color="black", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(df_out["Feature"], fontsize=10)
        ax.set_ylabel("Mean Within-Person r (PHQ-4 × Sensing)")
        ax.set_title("Within-Person Correlation: Weekly PHQ-4 × Weekly Sensing",
                     fontweight="bold")
        ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(OUT / "figure_within_person.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {OUT / 'figure_within_person.png'}", flush=True)

    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 17: Temporal Autocorrelation / Inertia Features
# ═══════════════════════════════════════════════════════════════════════
def run_inertia():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 17: Temporal Autocorrelation (Inertia) Features", flush=True)
    print("=" * 70, flush=True)

    FEATURE_MAP = [
        ("steps.csv", "avgsumsteps:14dhist", "steps"),
        ("sleep.csv", "avgdurationasleepmain:14dhist", "sleep_dur"),
        ("sleep.csv", "avgefficiencymain:14dhist", "sleep_eff"),
        ("screen.csv", "rapids_sumdurationunlock:14dhist", "screen_dur"),
        ("call.csv", "incoming_count:14dhist", "calls_in"),
        ("location.csv", "barnett_hometime:14dhist", "hometime"),
    ]

    def find_col(columns, substring):
        matches = [c for c in columns if substring in c]
        return min(matches, key=len) if matches else None

    # Compute person-level autocorrelation (lag-1)
    print("  Computing autocorrelation features...", flush=True)
    inertia_features = {}

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
            all_daily.append(daily)

        if not all_daily:
            continue
        combined = pd.concat(all_daily, ignore_index=True).dropna(subset=["value"])
        combined = combined.sort_values(["pid", "date"])

        # Person-level lag-1 autocorrelation
        autocorrs = {}
        for pid, grp in combined.groupby("pid"):
            vals = grp["value"].values
            if len(vals) >= 10:
                ac = np.corrcoef(vals[:-1], vals[1:])[0, 1]
                if not np.isnan(ac):
                    autocorrs[pid] = ac

        inertia_features[f"{short_name}_autocorr"] = pd.Series(autocorrs)

    # Merge into feature matrix
    feat_df = pd.DataFrame(inertia_features)
    feat_df.index.name = "pid"
    feat_df = feat_df.reset_index()

    s3_scores = s3[["pid"] + list({"bdi2_total": "BDI-II", "stai_state": "STAI",
                                    "pss_10": "PSS-10", "cesd_total": "CESD",
                                    "ucla_loneliness": "UCLA"}.keys()) + TRAITS].copy()
    merged = s3_scores.merge(feat_df, on="pid", how="inner")
    autocorr_cols = [c for c in inertia_features if c in merged.columns]
    valid_pers = [c for c in TRAITS if c in merged.columns]

    print(f"  Merged N={len(merged)}, {len(autocorr_cols)} inertia features", flush=True)

    S3_OUTCOMES = {"bdi2_total": "BDI-II", "stai_state": "STAI", "pss_10": "PSS-10",
                   "cesd_total": "CESD", "ucla_loneliness": "UCLA"}
    rows = []
    for outcome_col, outcome_label in S3_OUTCOMES.items():
        if outcome_col not in merged.columns:
            continue
        y = merged[outcome_col].values

        r2_inertia, n = quick_cv_r2(merged[autocorr_cols].values, y)
        r2_pers, _ = quick_cv_r2(merged[valid_pers].values, y)
        r2_both, _ = quick_cv_r2(merged[valid_pers + autocorr_cols].values, y)

        # Also: correlations between autocorrelation and outcomes
        corr_strs = []
        for ac_col in autocorr_cols:
            sub = merged[[ac_col, outcome_col]].dropna()
            if len(sub) > 20:
                r, p = stats.pearsonr(sub[ac_col], sub[outcome_col])
                if p < 0.05:
                    corr_strs.append(f"{ac_col}:r={r:.3f}*")

        print(f"  {outcome_label}: Inertia R²={r2_inertia:.4f}, Pers R²={r2_pers:.4f}, "
              f"Both R²={r2_both:.4f} (N={n})", flush=True)
        if corr_strs:
            print(f"    Sig correlations: {', '.join(corr_strs)}", flush=True)

        rows.append({
            "Outcome": outcome_label, "N": n,
            "R2_inertia_only": r2_inertia, "R2_personality": r2_pers,
            "R2_pers_plus_inertia": r2_both,
            "Sig_correlations": "; ".join(corr_strs),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "inertia_features.csv", index=False)
    print(f"  Saved: {OUT / 'inertia_features.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 18: S2 Raw Features (Non-PCA) Full Comparison
# ═══════════════════════════════════════════════════════════════════════
def run_s2_raw():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 18: S2 Raw Features (Non-PCA) Full Comparison", flush=True)
    print("=" * 70, flush=True)

    valid_pers = [c for c in TRAITS if c in s2.columns]
    valid_beh_raw = [c for c in S2_BEH_RAW if c in s2.columns]

    S2_OUTCOMES = {"cesd_total": "CES-D", "stai_trait_total": "STAI", "bai_total": "BAI"}
    rows = []

    for outcome_col, outcome_label in S2_OUTCOMES.items():
        if outcome_col not in s2.columns:
            continue
        y = s2[outcome_col].values

        # Personality only
        r2_pers, n = quick_cv_r2(s2[valid_pers].values, y)
        # PCA composites only
        r2_pca, _ = quick_cv_r2(s2[S2_BEH_PCA].values, y)
        # Raw 28 features only
        r2_raw, _ = quick_cv_r2(s2[valid_beh_raw].values, y)
        # Pers + PCA
        r2_pers_pca, _ = quick_cv_r2(s2[valid_pers + S2_BEH_PCA].values, y)
        # Pers + Raw
        r2_pers_raw, _ = quick_cv_r2(s2[valid_pers + valid_beh_raw].values, y)

        print(f"  {outcome_label} (N={n}):", flush=True)
        print(f"    Pers={r2_pers:.4f}, PCA={r2_pca:.4f}, Raw28={r2_raw:.4f}, "
              f"Pers+PCA={r2_pers_pca:.4f}, Pers+Raw28={r2_pers_raw:.4f}", flush=True)

        rows.append({
            "Outcome": outcome_label, "N": n,
            "R2_personality": r2_pers,
            "R2_beh_PCA": r2_pca, "R2_beh_raw28": r2_raw,
            "R2_pers_plus_PCA": r2_pers_pca, "R2_pers_plus_raw28": r2_pers_raw,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "s2_raw_features.csv", index=False)
    print(f"  Saved: {OUT / 's2_raw_features.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 19: Item-Level Prediction (Best BFI Items vs Full Scale)
# ═══════════════════════════════════════════════════════════════════════
def run_item_level():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 19: Item-Level Prediction (Best BFI Items vs Sensing)", flush=True)
    print("=" * 70, flush=True)

    # Load S2 BFI items
    bfi_items = pd.read_parquet("data/processed/nethealth/scores/bigfive_items.parquet")
    item_cols = [c for c in bfi_items.columns if c.startswith("bfi_")]
    print(f"  S2 BFI items: {len(item_cols)} items, N={len(bfi_items)}", flush=True)

    # Merge with S2
    s2_items = s2.merge(bfi_items[["egoid"] + item_cols], on="egoid", how="inner")

    # BFI-44 Neuroticism items: 4, 9R, 14, 19, 24R, 29, 34R, 39
    # (1-indexed in BFI-44: items 4,9,14,19,24,29,34,39)
    n_items = ["bfi_4", "bfi_9", "bfi_14", "bfi_19", "bfi_24", "bfi_29", "bfi_34", "bfi_39"]
    n_items = [c for c in n_items if c in s2_items.columns]

    S2_OUTCOMES = {"cesd_total": "CES-D", "stai_trait_total": "STAI", "bai_total": "BAI"}
    valid_pers = [c for c in TRAITS if c in s2_items.columns]
    rows = []

    for outcome_col, outcome_label in S2_OUTCOMES.items():
        if outcome_col not in s2_items.columns:
            continue
        y = s2_items[outcome_col].values

        # Full Big Five (5 traits)
        r2_full, n = quick_cv_r2(s2_items[valid_pers].values, y)

        # Neuroticism only (1 trait)
        r2_n_only, _ = quick_cv_r2(s2_items[["neuroticism"]].values, y)

        # Best 2/3 N items — nested CV (item selection inside each fold)
        if n_items:
            sub = s2_items[n_items + [outcome_col]].dropna()
            X_items_all = sub[n_items].values
            y_items = sub[outcome_col].values
            cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=RS)
            r2s_best2, r2s_best3 = [], []
            best_2_counts = {}
            for tr, te in cv.split(X_items_all):
                # Select items on TRAINING data only
                fold_corrs = {}
                for j, item in enumerate(n_items):
                    r, _ = stats.pearsonr(X_items_all[tr, j], y_items[tr])
                    fold_corrs[item] = abs(r)
                fold_best2 = sorted(fold_corrs, key=fold_corrs.get, reverse=True)[:2]
                fold_best3 = sorted(fold_corrs, key=fold_corrs.get, reverse=True)[:3]
                for it in fold_best2:
                    best_2_counts[it] = best_2_counts.get(it, 0) + 1
                # Fit and predict with fold-selected items
                idx2 = [n_items.index(it) for it in fold_best2]
                idx3 = [n_items.index(it) for it in fold_best3]
                sc2 = StandardScaler()
                Xtr2 = sc2.fit_transform(X_items_all[tr][:, idx2])
                Xte2 = sc2.transform(X_items_all[te][:, idx2])
                m2 = Ridge(alpha=1.0); m2.fit(Xtr2, y_items[tr])
                r2s_best2.append(r2_score(y_items[te], m2.predict(Xte2)))
                sc3 = StandardScaler()
                Xtr3 = sc3.fit_transform(X_items_all[tr][:, idx3])
                Xte3 = sc3.transform(X_items_all[te][:, idx3])
                m3 = Ridge(alpha=1.0); m3.fit(Xtr3, y_items[tr])
                r2s_best3.append(r2_score(y_items[te], m3.predict(Xte3)))
            r2_best2 = float(np.mean(r2s_best2))
            r2_best3 = float(np.mean(r2s_best3))
            best_2 = sorted(best_2_counts, key=best_2_counts.get, reverse=True)[:2]
            best_3 = sorted(best_2_counts, key=best_2_counts.get, reverse=True)[:3]
        else:
            r2_best2, r2_best3 = np.nan, np.nan
            best_2, best_3 = [], []

        # All 44 items
        avail_items = [c for c in item_cols if c in s2_items.columns]
        r2_all_items, _ = quick_cv_r2(s2_items[avail_items].values, y)

        # Sensing PCA
        r2_sensing, _ = quick_cv_r2(s2_items[S2_BEH_PCA].values, y)

        # Sensing raw 28
        valid_beh_raw = [c for c in S2_BEH_RAW if c in s2_items.columns]
        r2_sensing_raw, _ = quick_cv_r2(s2_items[valid_beh_raw].values, y)

        print(f"  {outcome_label} (N={n}):", flush=True)
        print(f"    2 best N items={r2_best2:.4f}, 3 best={r2_best3:.4f}, "
              f"N trait={r2_n_only:.4f}, Full BFI={r2_full:.4f}", flush=True)
        print(f"    All 44 items={r2_all_items:.4f}, "
              f"Sensing PCA={r2_sensing:.4f}, Sensing raw28={r2_sensing_raw:.4f}", flush=True)
        if best_2:
            count_str = ", ".join(f"{i}={best_2_counts.get(i, 0)}/25" for i in best_2)
            print(f"    Best 2 items (nested CV): {best_2} (selected in: {count_str} folds)", flush=True)

        rows.append({
            "Outcome": outcome_label, "N": n,
            "R2_2_best_items": r2_best2, "R2_3_best_items": r2_best3,
            "R2_neuroticism_only": r2_n_only, "R2_full_big5": r2_full,
            "R2_all_44_items": r2_all_items,
            "R2_sensing_PCA": r2_sensing, "R2_sensing_raw28": r2_sensing_raw,
            "Best_2_items": "; ".join(best_2),
            "Best_3_items": "; ".join(best_3),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "item_level_prediction.csv", index=False)
    print(f"  Saved: {OUT / 'item_level_prediction.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    analyses = {
        "15": ("Prospective Prediction", run_prospective),
        "16": ("Within-Person Correlation", run_within_person),
        "17": ("Inertia Features", run_inertia),
        "18": ("S2 Raw Features", run_s2_raw),
        "19": ("Item-Level Prediction", run_item_level),
    }

    if len(sys.argv) > 2 and sys.argv[1] == "--analysis":
        key = sys.argv[2]
        if key in analyses:
            name, fn = analyses[key]
            print(f"\nRunning Analysis {key}: {name}", flush=True)
            fn()
        else:
            print(f"Unknown: {key}. Available: {list(analyses.keys())}")
    else:
        for key in ["18", "19", "15", "17", "16"]:
            name, fn = analyses[key]
            print(f"\n{'#' * 70}", flush=True)
            print(f"# Analysis {key}: {name}", flush=True)
            print(f"{'#' * 70}", flush=True)
            fn()

    print("\n" + "=" * 70, flush=True)
    print("All Phase 16c analyses complete.", flush=True)
    print(f"Results in: {OUT}", flush=True)
    print("=" * 70, flush=True)
