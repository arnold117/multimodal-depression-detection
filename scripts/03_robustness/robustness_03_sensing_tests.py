#!/usr/bin/env python3
"""
Phase 16b — Additional Supplementary Analyses
==============================================
8. Sensing → Personality reverse prediction
9. Residualized prediction (sensing on personality residuals)
10. Data quantity dose-response (S3 daily data)
11. Stacking ensemble (meta-learner)
12. Within-person variability features (S3)
13. Smart feature selection (MI / correlation top-k)
14. Cross-study transfer (S2↔S3)
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.linear_model import Ridge, RidgeCV, ElasticNetCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_predict, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/robustness")
OUT.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
RS = 42

# ═══════════════════════════════════════════════════════════════════════
# Load datasets
# ═══════════════════════════════════════════════════════════════════════
print("Loading datasets...", flush=True)
s2 = pd.read_parquet("data/processed/nethealth/nethealth_analysis_dataset.parquet")
s3 = pd.read_parquet("data/processed/globem/globem_analysis_dataset.parquet")

S2_BEH_PCA = [c for c in s2.columns if c.startswith("nh_") and c.endswith("_pc1")]
S3_BEH_PCA = [c for c in s3.columns if c.endswith("_pc1") and not c.startswith("nh_")]

# S3 raw behavioral features (non-PCA)
S3_BEH_RAW = ["steps_avg", "steps_std", "active_bout_count", "sedentary_duration",
              "sleep_duration_avg", "sleep_efficiency", "sleep_onset_duration", "sleep_episode_count",
              "call_incoming_count", "call_outgoing_count", "call_incoming_contacts", "call_outgoing_contacts",
              "screen_unlock_count", "screen_duration_total", "screen_duration_avg",
              "loc_hometime", "loc_radius_gyration", "loc_unique_locations", "loc_entropy"]
S3_BEH_RAW = [c for c in S3_BEH_RAW if c in s3.columns]

# S2 raw behavioral features
S2_BEH_RAW = [c for c in s2.columns if c not in TRAITS + S2_BEH_PCA
              and c not in ["egoid", "cesd_total", "stai_trait_total", "bai_total",
                           "loneliness_total", "selsa_romantic", "selsa_family",
                           "selsa_social", "self_esteem_total", "gpa_overall",
                           "gpa_first_semester"]
              and not c.endswith("_pc1")]

S3_OUTCOMES = {"bdi2_total": "BDI-II", "stai_state": "STAI", "pss_10": "PSS-10",
               "cesd_total": "CESD", "ucla_loneliness": "UCLA"}
S2_OUTCOMES = {"cesd_total": "CES-D", "stai_trait_total": "STAI", "bai_total": "BAI"}

print(f"  S2: {s2.shape}, S3: {s3.shape}", flush=True)
print(f"  S3 raw beh: {len(S3_BEH_RAW)}, S2 raw beh: {len(S2_BEH_RAW)}", flush=True)


def quick_cv_r2(X, y, n_splits=5, n_repeats=5, alpha=1.0):
    """Quick Ridge CV, returns mean R²."""
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
# ANALYSIS 8: Sensing → Personality (Reverse Prediction)
# ═══════════════════════════════════════════════════════════════════════
def run_reverse_prediction():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 8: Sensing → Personality (Reverse Prediction)", flush=True)
    print("=" * 70, flush=True)

    rows = []
    for study, df, beh_cols, label in [("S2", s2, S2_BEH_PCA, "S2 PCA"),
                                        ("S3", s3, S3_BEH_PCA, "S3 PCA"),
                                        ("S3", s3, S3_BEH_RAW, "S3 Raw")]:
        valid_beh = [c for c in beh_cols if c in df.columns]
        if not valid_beh:
            continue
        for trait in TRAITS:
            if trait not in df.columns:
                continue
            X = df[valid_beh].values
            y = df[trait].values
            r2, n = quick_cv_r2(X, y)
            print(f"  {label} → {trait}: R²={r2:.4f} (N={n})", flush=True)
            rows.append({"Study": study, "Features": label, "Target": trait,
                         "R2": r2, "N": n})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "reverse_prediction.csv", index=False)
    print(f"\n  Mean R² (sensing→personality): {df_out['R2'].mean():.4f}", flush=True)
    print(f"  Saved: {OUT / 'reverse_prediction.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 9: Residualized Prediction
# ═══════════════════════════════════════════════════════════════════════
def run_residualized():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 9: Residualized Prediction (Sensing on Personality Residuals)", flush=True)
    print("=" * 70, flush=True)

    rows = []
    for study, df, beh_cols, outcomes in [("S2", s2, S2_BEH_PCA, S2_OUTCOMES),
                                           ("S3", s3, S3_BEH_PCA, S3_OUTCOMES)]:
        valid_pers = [c for c in TRAITS if c in df.columns]
        valid_beh = [c for c in beh_cols if c in df.columns]

        for outcome_col, outcome_label in outcomes.items():
            if outcome_col not in df.columns:
                continue
            sub = df[valid_pers + valid_beh + [outcome_col]].dropna()
            if len(sub) < 30:
                continue

            y = sub[outcome_col].values
            X_pers = sub[valid_pers].values
            X_beh = sub[valid_beh].values

            # Step 1: Get personality residuals via CV predictions
            cv = KFold(n_splits=5, shuffle=True, random_state=RS)
            y_pred_pers = np.zeros(len(y))
            for tr, te in cv.split(X_pers):
                sc = StandardScaler()
                Xtr = sc.fit_transform(X_pers[tr])
                Xte = sc.transform(X_pers[te])
                m = Ridge(alpha=1.0)
                m.fit(Xtr, y[tr])
                y_pred_pers[te] = m.predict(Xte)

            residuals = y - y_pred_pers
            r2_pers = r2_score(y, y_pred_pers)

            # Step 2: Can sensing predict these residuals?
            r2_resid, n = quick_cv_r2(X_beh, residuals)

            # Also: total variance explained by sensing on original outcome
            r2_beh, _ = quick_cv_r2(X_beh, y)

            print(f"  {study} {outcome_label}: Pers R²={r2_pers:.4f}, "
                  f"Beh→Original R²={r2_beh:.4f}, "
                  f"Beh→Residual R²={r2_resid:.4f} (N={n})", flush=True)

            rows.append({
                "Study": study, "Outcome": outcome_label, "N": n,
                "R2_personality": r2_pers, "R2_sensing_original": r2_beh,
                "R2_sensing_residual": r2_resid,
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "residualized_prediction.csv", index=False)
    print(f"  Saved: {OUT / 'residualized_prediction.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 10: Data Quantity Dose-Response (S3)
# ═══════════════════════════════════════════════════════════════════════
def run_dose_response():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 10: Data Quantity Dose-Response (S3)", flush=True)
    print("=" * 70, flush=True)

    COHORTS = ["INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4"]
    RAW_DIR = Path("data/raw/globem")

    # Feature mapping (same as reliability analysis)
    FEATURE_MAP = [
        ("steps.csv", "avgsumsteps:14dhist", "steps_avg"),
        ("steps.csv", "stdsumsteps:14dhist", "steps_std"),
        ("sleep.csv", "avgdurationasleepmain:14dhist", "sleep_duration_avg"),
        ("sleep.csv", "avgefficiencymain:14dhist", "sleep_efficiency"),
        ("call.csv", "incoming_count:14dhist", "call_incoming_count"),
        ("call.csv", "outgoing_count:14dhist", "call_outgoing_count"),
        ("screen.csv", "rapids_countepisodeunlock:14dhist", "screen_unlock_count"),
        ("screen.csv", "rapids_sumdurationunlock:14dhist", "screen_duration_total"),
        ("location.csv", "barnett_hometime:14dhist", "loc_hometime"),
        ("location.csv", "barnett_rog:14dhist", "loc_radius_gyration"),
    ]

    def find_col(columns, substring):
        matches = [c for c in columns if substring in c]
        return min(matches, key=len) if matches else None

    # For each window size, aggregate daily data to person-level
    WINDOWS = [7, 14, 30, 60, 92]
    rows = []

    # Load all daily data for each feature
    print("  Loading daily data...", flush=True)
    daily_data = {}  # {short_name: DataFrame with pid, day_num, value}
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
        if all_daily:
            combined = pd.concat(all_daily, ignore_index=True).dropna(subset=["value"])
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.sort_values(["pid", "date"])
            combined["day_num"] = combined.groupby("pid").cumcount()
            daily_data[short_name] = combined

    print(f"  Loaded {len(daily_data)} features with daily data", flush=True)

    # S3 outcomes
    s3_scores = s3[["pid"] + list(S3_OUTCOMES.keys()) + TRAITS].copy()

    for n_days in WINDOWS:
        print(f"\n  Window: {n_days} days", flush=True)

        # Aggregate each feature using first n_days
        person_features = {}
        for short_name, daily_df in daily_data.items():
            subset = daily_df[daily_df["day_num"] < n_days]
            agg = subset.groupby("pid")["value"].mean()
            person_features[short_name] = agg

        # Merge into a feature matrix
        feat_df = pd.DataFrame(person_features)
        feat_df.index.name = "pid"
        feat_df = feat_df.reset_index()

        merged = s3_scores.merge(feat_df, on="pid", how="inner")
        beh_cols = [c for c in person_features.keys() if c in merged.columns]

        if len(merged) < 50 or len(beh_cols) < 3:
            continue

        valid_pers = [c for c in TRAITS if c in merged.columns]

        for outcome_col, outcome_label in S3_OUTCOMES.items():
            if outcome_col not in merged.columns:
                continue
            y = merged[outcome_col].values

            # Behavior only
            X_beh = merged[beh_cols].values
            r2_beh, n = quick_cv_r2(X_beh, y)

            # Personality only
            X_pers = merged[valid_pers].values
            r2_pers, _ = quick_cv_r2(X_pers, y)

            # Combined
            X_both = merged[valid_pers + beh_cols].values
            r2_both, _ = quick_cv_r2(X_both, y)

            print(f"    {outcome_label}: Beh R²={r2_beh:.4f}, Pers R²={r2_pers:.4f}, "
                  f"Both R²={r2_both:.4f} (N={n})", flush=True)

            rows.append({
                "N_days": n_days, "Outcome": outcome_label, "N": n,
                "R2_behavior": r2_beh, "R2_personality": r2_pers, "R2_combined": r2_both,
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "dose_response.csv", index=False)
    print(f"\n  Saved: {OUT / 'dose_response.csv'}", flush=True)

    # Figure
    if len(df_out) > 0:
        outcomes_list = df_out["Outcome"].unique()
        fig, axes = plt.subplots(1, len(outcomes_list), figsize=(4 * len(outcomes_list), 4))
        if len(outcomes_list) == 1:
            axes = [axes]
        for idx, outcome in enumerate(outcomes_list):
            ax = axes[idx]
            sub = df_out[df_out["Outcome"] == outcome]
            ax.plot(sub["N_days"], sub["R2_behavior"], "o-", label="Sensing", color="#95a5a6")
            ax.plot(sub["N_days"], sub["R2_personality"], "s--", label="Personality", color="#e74c3c")
            ax.plot(sub["N_days"], sub["R2_combined"], "^-", label="Combined", color="#3498db")
            ax.axhline(0, color="grey", linestyle=":", alpha=0.5)
            ax.set_xlabel("Days of Sensing Data")
            ax.set_ylabel("R²")
            ax.set_title(outcome, fontweight="bold")
            ax.legend(fontsize=7)
        plt.suptitle("Dose-Response: More Sensing Data Does Not Improve Prediction",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(OUT / "figure_dose_response.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {OUT / 'figure_dose_response.png'}", flush=True)

    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 11: Stacking Ensemble
# ═══════════════════════════════════════════════════════════════════════
def run_stacking():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 11: Stacking Ensemble (Meta-Learner)", flush=True)
    print("=" * 70, flush=True)

    rows = []

    for study, df, beh_cols, outcomes in [("S2", s2, S2_BEH_PCA, S2_OUTCOMES),
                                           ("S3", s3, S3_BEH_PCA, S3_OUTCOMES)]:
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

            # Outer CV
            cv_outer = RepeatedKFold(n_splits=5, n_repeats=5, random_state=RS)
            r2_concat, r2_stack = [], []

            for tr, te in cv_outer.split(X_pers):
                sc_p = StandardScaler()
                sc_b = StandardScaler()

                Xp_tr = sc_p.fit_transform(X_pers[tr])
                Xp_te = sc_p.transform(X_pers[te])
                Xb_tr = sc_b.fit_transform(X_beh[tr])
                Xb_te = sc_b.transform(X_beh[te])
                y_tr, y_te = y[tr], y[te]

                # Method 1: Simple concatenation
                Xc_tr = np.hstack([Xp_tr, Xb_tr])
                Xc_te = np.hstack([Xp_te, Xb_te])
                m_concat = Ridge(alpha=1.0)
                m_concat.fit(Xc_tr, y_tr)
                r2_concat.append(r2_score(y_te, m_concat.predict(Xc_te)))

                # Method 2: Stacking
                # Inner CV to get OOF predictions for training meta-learner
                cv_inner = KFold(n_splits=3, shuffle=True, random_state=RS)
                oof_pers = np.zeros(len(tr))
                oof_beh = np.zeros(len(tr))

                for itr, ite in cv_inner.split(Xp_tr):
                    m_p = Ridge(alpha=1.0)
                    m_b = Ridge(alpha=1.0)
                    m_p.fit(Xp_tr[itr], y_tr[itr])
                    m_b.fit(Xb_tr[itr], y_tr[itr])
                    oof_pers[ite] = m_p.predict(Xp_tr[ite])
                    oof_beh[ite] = m_b.predict(Xb_tr[ite])

                # Meta-learner on OOF predictions
                X_meta_tr = np.column_stack([oof_pers, oof_beh])
                meta = Ridge(alpha=0.1)
                meta.fit(X_meta_tr, y_tr)

                # Test: get base model predictions
                m_p_full = Ridge(alpha=1.0)
                m_b_full = Ridge(alpha=1.0)
                m_p_full.fit(Xp_tr, y_tr)
                m_b_full.fit(Xb_tr, y_tr)
                X_meta_te = np.column_stack([m_p_full.predict(Xp_te),
                                              m_b_full.predict(Xb_te)])
                r2_stack.append(r2_score(y_te, meta.predict(X_meta_te)))

            r2_c = float(np.mean(r2_concat))
            r2_s = float(np.mean(r2_stack))

            print(f"  {study} {outcome_label}: Concat R²={r2_c:.4f}, Stack R²={r2_s:.4f}, "
                  f"Δ={r2_s - r2_c:+.4f} (N={len(sub)})", flush=True)

            rows.append({
                "Study": study, "Outcome": outcome_label, "N": len(sub),
                "R2_concatenation": r2_c, "R2_stacking": r2_s,
                "Delta": r2_s - r2_c,
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "stacking_ensemble.csv", index=False)
    print(f"  Saved: {OUT / 'stacking_ensemble.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 12: Within-Person Variability Features (S3)
# ═══════════════════════════════════════════════════════════════════════
def run_variability():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 12: Within-Person Variability Features (S3)", flush=True)
    print("=" * 70, flush=True)

    COHORTS = ["INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4"]
    RAW_DIR = Path("data/raw/globem")

    # Key features for variability
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

    # Compute person-level variability metrics
    print("  Computing variability metrics...", flush=True)
    var_features = {}

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
        combined = pd.concat(all_daily, ignore_index=True).dropna(subset=["value"])

        # Person-level: mean, SD, CV, range, IQR
        agg = combined.groupby("pid")["value"].agg(["mean", "std", "min", "max"])
        agg[f"{short_name}_cv"] = agg["std"] / agg["mean"].abs().clip(lower=0.001)
        agg[f"{short_name}_range"] = agg["max"] - agg["min"]
        agg[f"{short_name}_mean"] = agg["mean"]
        agg[f"{short_name}_sd"] = agg["std"]

        for metric in ["mean", "sd", "cv", "range"]:
            col_name = f"{short_name}_{metric}"
            var_features[col_name] = agg[col_name]

    # Merge into feature matrix
    feat_df = pd.DataFrame(var_features)
    feat_df.index.name = "pid"
    feat_df = feat_df.reset_index()

    s3_scores = s3[["pid"] + list(S3_OUTCOMES.keys()) + TRAITS].copy()
    merged = s3_scores.merge(feat_df, on="pid", how="inner")
    print(f"  Merged N={len(merged)}, {len(var_features)} variability features", flush=True)

    mean_cols = [c for c in var_features if c.endswith("_mean")]
    sd_cols = [c for c in var_features if c.endswith("_sd")]
    cv_cols = [c for c in var_features if c.endswith("_cv")]
    all_var_cols = [c for c in var_features if c in merged.columns]
    valid_pers = [c for c in TRAITS if c in merged.columns]

    rows = []
    for outcome_col, outcome_label in S3_OUTCOMES.items():
        if outcome_col not in merged.columns:
            continue
        y = merged[outcome_col].values

        # Mean features only (= standard approach)
        r2_mean, n = quick_cv_r2(merged[mean_cols].values, y)
        # SD features only (variability)
        r2_sd, _ = quick_cv_r2(merged[sd_cols].values, y)
        # CV features only
        r2_cv, _ = quick_cv_r2(merged[cv_cols].values, y)
        # All variability features
        r2_all, _ = quick_cv_r2(merged[all_var_cols].values, y)
        # Personality + all variability
        r2_pers_var, _ = quick_cv_r2(merged[valid_pers + all_var_cols].values, y)
        # Personality only
        r2_pers, _ = quick_cv_r2(merged[valid_pers].values, y)

        print(f"  {outcome_label}: Mean R²={r2_mean:.4f}, SD R²={r2_sd:.4f}, "
              f"CV R²={r2_cv:.4f}, All R²={r2_all:.4f}, "
              f"Pers+Var R²={r2_pers_var:.4f}, Pers R²={r2_pers:.4f} (N={n})", flush=True)

        rows.append({
            "Outcome": outcome_label, "N": n,
            "R2_means_only": r2_mean, "R2_sds_only": r2_sd,
            "R2_cvs_only": r2_cv, "R2_all_variability": r2_all,
            "R2_pers_plus_var": r2_pers_var, "R2_personality": r2_pers,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "variability_features.csv", index=False)
    print(f"  Saved: {OUT / 'variability_features.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 13: Smart Feature Selection (MI / Correlation Top-K)
# ═══════════════════════════════════════════════════════════════════════
def run_feature_selection():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 13: Smart Feature Selection (Top-K)", flush=True)
    print("=" * 70, flush=True)

    rows = []
    valid_pers = [c for c in TRAITS if c in s3.columns]

    for outcome_col, outcome_label in S3_OUTCOMES.items():
        if outcome_col not in s3.columns:
            continue

        sub = s3[valid_pers + S3_BEH_RAW + [outcome_col]].dropna()
        if len(sub) < 50:
            continue

        y = sub[outcome_col].values
        X_beh = sub[S3_BEH_RAW].values
        X_pers = sub[valid_pers].values

        # Baseline: PCA composites
        X_pca = sub[[c for c in S3_BEH_PCA if c in sub.columns]].values
        r2_pca, n = quick_cv_r2(X_pca, y) if X_pca.shape[1] > 0 else (np.nan, len(sub))

        # F-test top-5
        selector = SelectKBest(f_regression, k=min(5, X_beh.shape[1]))
        mask_clean = ~np.isnan(X_beh).any(axis=1)
        selector.fit(X_beh[mask_clean], y[mask_clean])
        top5_idx = selector.get_support(indices=True)
        top5_names = [S3_BEH_RAW[i] for i in top5_idx]
        X_top5 = sub[top5_names].values
        r2_top5, _ = quick_cv_r2(X_top5, y)

        # Correlation-based top-5
        corrs = {}
        for i, feat in enumerate(S3_BEH_RAW):
            vals = sub[feat].values
            mask_c = ~np.isnan(vals)
            if mask_c.sum() > 30:
                r, _ = stats.pearsonr(vals[mask_c], y[mask_c])
                corrs[feat] = abs(r)
        top5_corr = sorted(corrs, key=corrs.get, reverse=True)[:5]
        X_top5c = sub[top5_corr].values
        r2_top5c, _ = quick_cv_r2(X_top5c, y)

        # Pers + top5 (best of F/corr)
        best_top5 = top5_names if r2_top5 > r2_top5c else top5_corr
        r2_pers_top5, _ = quick_cv_r2(sub[valid_pers + best_top5].values, y)
        r2_pers, _ = quick_cv_r2(X_pers, y)

        print(f"  {outcome_label}: PCA R²={r2_pca:.4f}, F-top5 R²={r2_top5:.4f}, "
              f"Corr-top5 R²={r2_top5c:.4f}, Pers+Top5 R²={r2_pers_top5:.4f}, "
              f"Pers R²={r2_pers:.4f}", flush=True)
        print(f"    F-top5: {top5_names}", flush=True)
        print(f"    Corr-top5: {top5_corr}", flush=True)

        rows.append({
            "Outcome": outcome_label, "N": n,
            "R2_PCA_composites": r2_pca, "R2_F_top5": r2_top5,
            "R2_Corr_top5": r2_top5c, "R2_Pers_top5": r2_pers_top5,
            "R2_personality": r2_pers,
            "F_top5_features": "; ".join(top5_names),
            "Corr_top5_features": "; ".join(top5_corr),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "feature_selection.csv", index=False)
    print(f"  Saved: {OUT / 'feature_selection.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 14: Cross-Study Transfer (S2 ↔ S3)
# ═══════════════════════════════════════════════════════════════════════
def run_cross_transfer():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 14: Cross-Study Transfer (S2 ↔ S3)", flush=True)
    print("=" * 70, flush=True)

    # Common features between S2 and S3 (approximate mapping)
    # Both have: personality traits, and we can map PCA composites conceptually
    # For a fair test: train personality model on S2, test on S3 (and vice versa)

    valid_pers = [c for c in TRAITS if c in s2.columns and c in s3.columns]
    rows = []

    # Shared MH constructs (approximate)
    transfers = [
        # (train_study, train_df, train_col, test_study, test_df, test_col, label)
        ("S2", s2, "cesd_total", "S3", s3, "cesd_total", "Depression (CES-D)"),
        ("S3", s3, "cesd_total", "S2", s2, "cesd_total", "Depression (CES-D) [reverse]"),
        ("S2", s2, "stai_trait_total", "S3", s3, "stai_state", "Anxiety (STAI)"),
        ("S3", s3, "stai_state", "S2", s2, "stai_trait_total", "Anxiety (STAI) [reverse]"),
    ]

    for train_study, train_df, train_col, test_study, test_df, test_col, label in transfers:
        if train_col not in train_df.columns or test_col not in test_df.columns:
            continue

        # Personality model
        train_sub = train_df[valid_pers + [train_col]].dropna()
        test_sub = test_df[valid_pers + [test_col]].dropna()

        if len(train_sub) < 30 or len(test_sub) < 30:
            continue

        X_train = train_sub[valid_pers].values
        y_train = train_sub[train_col].values
        X_test = test_sub[valid_pers].values
        y_test = test_sub[test_col].values

        # Standardize
        sc = StandardScaler()
        X_train_s = sc.fit_transform(X_train)
        X_test_s = sc.transform(X_test)

        # Standardize outcomes for comparability
        y_train_s = (y_train - y_train.mean()) / y_train.std()
        y_test_s = (y_test - y_test.mean()) / y_test.std()

        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train_s)
        y_pred = model.predict(X_test_s)
        r2_transfer = r2_score(y_test_s, y_pred)

        # Within-study CV for comparison
        r2_within, _ = quick_cv_r2(X_test, y_test)

        print(f"  {label}: Train {train_study} (N={len(train_sub)}) → "
              f"Test {test_study} (N={len(test_sub)}): "
              f"R²_transfer={r2_transfer:.4f}, R²_within={r2_within:.4f}", flush=True)

        rows.append({
            "Transfer": label, "Train_study": train_study, "Test_study": test_study,
            "N_train": len(train_sub), "N_test": len(test_sub),
            "R2_transfer": r2_transfer, "R2_within_study": r2_within,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "cross_study_transfer.csv", index=False)
    print(f"  Saved: {OUT / 'cross_study_transfer.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    analyses = {
        "8":  ("Reverse Prediction", run_reverse_prediction),
        "9":  ("Residualized Prediction", run_residualized),
        "10": ("Dose-Response", run_dose_response),
        "11": ("Stacking Ensemble", run_stacking),
        "12": ("Variability Features", run_variability),
        "13": ("Feature Selection", run_feature_selection),
        "14": ("Cross-Study Transfer", run_cross_transfer),
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
        for key in ["8", "9", "11", "13", "14", "12", "10"]:
            name, fn = analyses[key]
            print(f"\n{'#' * 70}", flush=True)
            print(f"# Analysis {key}: {name}", flush=True)
            print(f"{'#' * 70}", flush=True)
            fn()

    print("\n" + "=" * 70, flush=True)
    print("All Phase 16b analyses complete.", flush=True)
    print(f"Results in: {OUT}", flush=True)
    print("=" * 70, flush=True)
