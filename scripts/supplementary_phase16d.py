#!/usr/bin/env python3
"""
Phase 16d — Pro-Sensing Analyses (Devil's Advocate)
====================================================
Try every possible way to make sensing work:
20. Idiographic (person-specific) models
21. Personality × Sensing interaction
22. Person-centered (ipsative) features
23. S2 deep dive: why does sensing work there?
24. Weekly concurrent prediction (panel regression)
25. Nonlinear models on best modality (sleep)
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
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
# ANALYSIS 20: Idiographic (Person-Specific) Models
# ═══════════════════════════════════════════════════════════════════════
def run_idiographic():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 20: Idiographic (Person-Specific) Models", flush=True)
    print("=" * 70, flush=True)

    FEATURE_MAP = [
        ("steps.csv", "avgsumsteps:14dhist", "steps"),
        ("sleep.csv", "avgdurationasleepmain:14dhist", "sleep_dur"),
        ("sleep.csv", "avgefficiencymain:14dhist", "sleep_eff"),
        ("screen.csv", "rapids_sumdurationunlock:14dhist", "screen_dur"),
        ("call.csv", "incoming_count:14dhist", "calls_in"),
        ("location.csv", "barnett_hometime:14dhist", "hometime"),
    ]

    # Load weekly PHQ-4
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

    # Load daily sensing
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

    # Build weekly sensing panel aligned to PHQ-4 dates
    print("  Building person × week panel...", flush=True)
    person_week_data = []

    for pid in weekly["pid"].unique():
        phq_pid = weekly[weekly["pid"] == pid].sort_values("date")
        if len(phq_pid) < 5:
            continue

        for _, row in phq_pid.iterrows():
            phq_date = row["date"]
            week_record = {"pid": pid, "date": phq_date, "phq4": row["phq4"]}

            for feat_name, daily_df in daily_features.items():
                sens_pid = daily_df[daily_df["pid"] == pid]
                week_sens = sens_pid[(sens_pid["date"] >= phq_date - pd.Timedelta(days=7)) &
                                     (sens_pid["date"] <= phq_date)]
                week_record[feat_name] = week_sens["value"].mean() if len(week_sens) >= 2 else np.nan

            person_week_data.append(week_record)

    panel = pd.DataFrame(person_week_data)
    feat_cols = list(daily_features.keys())
    panel = panel.dropna(subset=["phq4"] + feat_cols)
    print(f"  Panel: {len(panel)} obs, {panel['pid'].nunique()} persons", flush=True)

    # Person-specific models: for each person, predict PHQ-4 from sensing
    person_results = []
    for pid, grp in panel.groupby("pid"):
        if len(grp) < 5:
            continue

        y = grp["phq4"].values
        X = grp[feat_cols].values

        # LOO-CV within person
        if len(grp) >= 6:
            r2s_loo = []
            for i in range(len(grp)):
                X_tr = np.delete(X, i, axis=0)
                y_tr = np.delete(y, i)
                X_te = X[i:i+1]
                y_te = y[i]
                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr)
                X_te_s = sc.transform(X_te)
                m = Ridge(alpha=1.0)
                m.fit(X_tr_s, y_tr)
                r2s_loo.append((y_te, m.predict(X_te_s)[0]))

            y_true = [r[0] for r in r2s_loo]
            y_pred = [r[1] for r in r2s_loo]
            if np.std(y_true) > 0:
                r2_person = r2_score(y_true, y_pred)
            else:
                r2_person = np.nan
        else:
            # Simple correlation
            sc = StandardScaler()
            X_s = sc.fit_transform(X)
            m = Ridge(alpha=1.0)
            m.fit(X_s, y)
            r2_person = r2_score(y, m.predict(X_s))  # in-sample for small n

        # Also compute per-feature correlations
        best_r = 0
        for j, feat in enumerate(feat_cols):
            r, _ = stats.pearsonr(X[:, j], y)
            if abs(r) > abs(best_r):
                best_r = r

        person_results.append({
            "pid": pid, "n_weeks": len(grp),
            "R2_person": r2_person,
            "best_feature_r": best_r,
            "phq4_mean": y.mean(), "phq4_sd": y.std(),
        })

    df_person = pd.DataFrame(person_results)
    df_person.to_csv(OUT / "idiographic_models.csv", index=False)

    # Summary statistics
    valid = df_person.dropna(subset=["R2_person"])
    print(f"\n  Person-specific models (N={len(valid)}):", flush=True)
    print(f"    Mean R²: {valid['R2_person'].mean():.4f}", flush=True)
    print(f"    Median R²: {valid['R2_person'].median():.4f}", flush=True)
    print(f"    R² > 0: {(valid['R2_person'] > 0).sum()}/{len(valid)} "
          f"({(valid['R2_person'] > 0).mean()*100:.1f}%)", flush=True)
    print(f"    R² > 0.3: {(valid['R2_person'] > 0.3).sum()}/{len(valid)} "
          f"({(valid['R2_person'] > 0.3).mean()*100:.1f}%)", flush=True)
    print(f"    Mean |best_r|: {valid['best_feature_r'].abs().mean():.4f}", flush=True)

    # Are high-variability people better predicted?
    high_var = valid[valid["phq4_sd"] > valid["phq4_sd"].median()]
    low_var = valid[valid["phq4_sd"] <= valid["phq4_sd"].median()]
    print(f"    High PHQ-4 variability: mean R²={high_var['R2_person'].mean():.4f} (N={len(high_var)})", flush=True)
    print(f"    Low PHQ-4 variability:  mean R²={low_var['R2_person'].mean():.4f} (N={len(low_var)})", flush=True)

    # Figure: histogram of person-specific R²
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1 = axes[0]
    ax1.hist(valid["R2_person"].clip(-2, 1), bins=30, color="#3498db", edgecolor="white")
    ax1.axvline(0, color="red", linestyle="--", label="R²=0")
    ax1.axvline(valid["R2_person"].median(), color="green", linestyle="--",
                label=f"Median={valid['R2_person'].median():.3f}")
    ax1.set_xlabel("Person-Specific R²")
    ax1.set_ylabel("Count")
    ax1.set_title("Idiographic Model Performance", fontweight="bold")
    ax1.legend()

    ax2 = axes[1]
    ax2.scatter(valid["phq4_sd"], valid["R2_person"].clip(-2, 1),
                alpha=0.3, s=20, color="#e74c3c")
    ax2.axhline(0, color="grey", linestyle="--", alpha=0.5)
    ax2.set_xlabel("PHQ-4 Within-Person SD")
    ax2.set_ylabel("Person-Specific R²")
    ax2.set_title("Prediction Depends on PHQ-4 Variability", fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUT / "figure_idiographic.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT / 'figure_idiographic.png'}", flush=True)

    return df_person


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 21: Personality × Sensing Interaction
# ═══════════════════════════════════════════════════════════════════════
def run_interaction():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 21: Personality × Sensing Interaction", flush=True)
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

            y = sub[outcome_col].values

            # Model A: Pers + Beh (additive)
            X_add = sub[valid_pers + valid_beh].values
            r2_add, n = quick_cv_r2(X_add, y)

            # Model B: Pers + Beh + N×Beh interactions (Neuroticism × each beh feature)
            interaction_cols = []
            for beh in valid_beh:
                col_name = f"N_x_{beh}"
                sub[col_name] = sub["neuroticism"] * sub[beh]
                interaction_cols.append(col_name)

            X_interact = sub[valid_pers + valid_beh + interaction_cols].values
            r2_interact, _ = quick_cv_r2(X_interact, y)

            # Model C: Pers + Beh + ALL Pers×Beh interactions
            all_interact_cols = []
            for pers in valid_pers:
                for beh in valid_beh:
                    col_name = f"{pers[:3]}_x_{beh}"
                    sub[col_name] = sub[pers] * sub[beh]
                    all_interact_cols.append(col_name)

            X_all_interact = sub[valid_pers + valid_beh + all_interact_cols].values
            r2_all_interact, _ = quick_cv_r2(X_all_interact, y)

            # Model D: Random Forest (captures interactions naturally)
            cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RS)
            rf_r2s = []
            for tr, te in cv.split(X_add):
                sc = StandardScaler()
                Xtr = sc.fit_transform(X_add[tr])
                Xte = sc.transform(X_add[te])
                rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=RS)
                rf.fit(Xtr, y[tr])
                rf_r2s.append(r2_score(y[te], rf.predict(Xte)))
            r2_rf = float(np.mean(rf_r2s))

            delta_n = r2_interact - r2_add
            delta_all = r2_all_interact - r2_add

            print(f"  {study} {outcome_label} (N={n}): "
                  f"Add={r2_add:.4f}, +N×Beh={r2_interact:.4f} (Δ={delta_n:+.4f}), "
                  f"+All×Beh={r2_all_interact:.4f} (Δ={delta_all:+.4f}), "
                  f"RF={r2_rf:.4f}", flush=True)

            rows.append({
                "Study": study, "Outcome": outcome_label, "N": n,
                "R2_additive": r2_add, "R2_N_interactions": r2_interact,
                "R2_all_interactions": r2_all_interact, "R2_random_forest": r2_rf,
                "Delta_N_interact": delta_n, "Delta_all_interact": delta_all,
            })

            # Cleanup temp columns
            for c in interaction_cols + all_interact_cols:
                if c in sub.columns:
                    sub.drop(columns=c, inplace=True)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "personality_sensing_interaction.csv", index=False)
    print(f"  Saved: {OUT / 'personality_sensing_interaction.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 22: Person-Centered (Ipsative) Features
# ═══════════════════════════════════════════════════════════════════════
def run_ipsative():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 22: Person-Centered (Ipsative) Features", flush=True)
    print("=" * 70, flush=True)

    FEATURE_MAP = [
        ("steps.csv", "avgsumsteps:14dhist", "steps"),
        ("sleep.csv", "avgdurationasleepmain:14dhist", "sleep_dur"),
        ("sleep.csv", "avgefficiencymain:14dhist", "sleep_eff"),
        ("screen.csv", "rapids_sumdurationunlock:14dhist", "screen_dur"),
        ("call.csv", "incoming_count:14dhist", "calls_in"),
        ("location.csv", "barnett_hometime:14dhist", "hometime"),
    ]

    # Load daily data and compute person-centered features
    print("  Computing person-centered features...", flush=True)

    all_daily = {}
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
            daily = df_raw[["pid", col]].copy()
            daily.columns = ["pid", "value"]
            daily["value"] = pd.to_numeric(daily["value"], errors="coerce")
            frames.append(daily)
        if frames:
            all_daily[short_name] = pd.concat(frames, ignore_index=True)

    # Compute ipsative features: person's last 2 weeks vs their overall mean
    ipsative_features = {}
    for short_name, daily_df in all_daily.items():
        daily_df = daily_df.dropna(subset=["value"])

        # Person mean (full period)
        person_mean = daily_df.groupby("pid")["value"].mean()
        # Person last-2-weeks mean (approximation: last 14 rows)
        last2w = daily_df.groupby("pid").tail(14).groupby("pid")["value"].mean()
        # First-2-weeks mean
        first2w = daily_df.groupby("pid").head(14).groupby("pid")["value"].mean()

        common = person_mean.index.intersection(last2w.index).intersection(first2w.index)

        # Ipsative: deviation from own mean
        ipsative_features[f"{short_name}_last2w_dev"] = (last2w.loc[common] - person_mean.loc[common])
        ipsative_features[f"{short_name}_first2w_dev"] = (first2w.loc[common] - person_mean.loc[common])
        # Trend: last2w - first2w
        ipsative_features[f"{short_name}_trend"] = (last2w.loc[common] - first2w.loc[common])

    feat_df = pd.DataFrame(ipsative_features)
    feat_df.index.name = "pid"
    feat_df = feat_df.reset_index()

    s3_scores = s3[["pid"] + list({"bdi2_total": "BDI-II", "stai_state": "STAI",
                                    "pss_10": "PSS-10"}.keys()) + TRAITS].copy()
    merged = s3_scores.merge(feat_df, on="pid", how="inner")

    ipsative_cols = [c for c in ipsative_features if c in merged.columns]
    valid_pers = [c for c in TRAITS if c in merged.columns]

    print(f"  Merged N={len(merged)}, {len(ipsative_cols)} ipsative features", flush=True)

    rows = []
    for outcome_col, outcome_label in [("bdi2_total", "BDI-II"), ("stai_state", "STAI"), ("pss_10", "PSS-10")]:
        if outcome_col not in merged.columns:
            continue
        y = merged[outcome_col].values

        # Ipsative only
        r2_ips, n = quick_cv_r2(merged[ipsative_cols].values, y)
        # Personality only
        r2_pers, _ = quick_cv_r2(merged[valid_pers].values, y)
        # Pers + Ipsative
        r2_both, _ = quick_cv_r2(merged[valid_pers + ipsative_cols].values, y)
        # Trend features only
        trend_cols = [c for c in ipsative_cols if c.endswith("_trend")]
        r2_trend, _ = quick_cv_r2(merged[trend_cols].values, y)

        print(f"  {outcome_label}: Ipsative R²={r2_ips:.4f}, Trend R²={r2_trend:.4f}, "
              f"Pers R²={r2_pers:.4f}, Pers+Ips R²={r2_both:.4f} (N={n})", flush=True)

        rows.append({
            "Outcome": outcome_label, "N": n,
            "R2_ipsative": r2_ips, "R2_trend": r2_trend,
            "R2_personality": r2_pers, "R2_pers_plus_ipsative": r2_both,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "ipsative_features.csv", index=False)
    print(f"  Saved: {OUT / 'ipsative_features.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 23: S2 Deep Dive — Why Does Sensing Work There?
# ═══════════════════════════════════════════════════════════════════════
def run_s2_deep_dive():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 23: S2 Deep Dive — Why Does Sensing Work?", flush=True)
    print("=" * 70, flush=True)

    valid_pers = [c for c in TRAITS if c in s2.columns]

    # Group S2 raw features by modality
    fitbit_activity = [c for c in S2_BEH_RAW if any(k in c for k in
                       ["steps_", "sedentary_", "light_active", "fairly_active",
                        "very_active", "total_active", "active_ratio", "fatburn", "cardio"])
                       and "n_days" not in c]
    fitbit_sleep = [c for c in S2_BEH_RAW if any(k in c for k in
                    ["sleep_duration", "sleep_interruptions", "sleep_efficiency",
                     "sleep_onset", "sleep_regularity", "time_to_fall"])
                    and "n_days" not in c]
    communication = [c for c in S2_BEH_RAW if any(k in c for k in
                     ["call_count", "call_duration", "call_outgoing", "sms_count",
                      "sms_outgoing", "total_unique_contacts", "total_comm"])
                     and "n_days" not in c]

    rows = []
    S2_OUTCOMES = {"cesd_total": "CES-D", "stai_trait_total": "STAI", "bai_total": "BAI"}

    for outcome_col, outcome_label in S2_OUTCOMES.items():
        y = s2[outcome_col].values

        # Each modality alone
        for mod_name, mod_cols in [("Activity", fitbit_activity),
                                    ("Sleep", fitbit_sleep),
                                    ("Communication", communication)]:
            valid_mod = [c for c in mod_cols if c in s2.columns]
            if not valid_mod:
                continue

            r2_mod, n = quick_cv_r2(s2[valid_mod].values, y)
            r2_pers_mod, _ = quick_cv_r2(s2[valid_pers + valid_mod].values, y)
            r2_pers, _ = quick_cv_r2(s2[valid_pers].values, y)

            print(f"  {outcome_label} + {mod_name} ({len(valid_mod)} feat): "
                  f"Mod={r2_mod:.4f}, Pers+Mod={r2_pers_mod:.4f}, "
                  f"Pers={r2_pers:.4f}, Δ={r2_pers_mod - r2_pers:+.4f} (N={n})", flush=True)

            rows.append({
                "Outcome": outcome_label, "Modality": mod_name,
                "N_features": len(valid_mod), "N": n,
                "R2_modality_only": r2_mod, "R2_personality": r2_pers,
                "R2_pers_plus_mod": r2_pers_mod,
                "Delta_R2": r2_pers_mod - r2_pers,
            })

    # Data quality comparison: S2 vs S3
    print("\n  Data quality comparison:", flush=True)
    s2_missing = s2[S2_BEH_RAW].isna().mean().mean()
    s3_beh_raw = [c for c in s3.columns if c not in TRAITS + S3_BEH_PCA
                  and c not in ["pid", "bdi2_total", "stai_state", "pss_10",
                               "cesd_total", "ucla_loneliness", "cohort"]
                  and not c.endswith("_pc1")]
    s3_missing = s3[s3_beh_raw].isna().mean().mean() if s3_beh_raw else np.nan
    print(f"    S2 raw features mean missing rate: {s2_missing:.4f}", flush=True)
    print(f"    S3 raw features mean missing rate: {s3_missing:.4f}", flush=True)

    # S2 N with complete sensing data
    s2_complete = s2[S2_BEH_RAW].dropna().shape[0]
    print(f"    S2 complete sensing cases: {s2_complete}/{len(s2)}", flush=True)

    # Communication is unique to S2 — is that the secret?
    print("\n  Communication (S2 unique) contribution:", flush=True)
    for outcome_col, outcome_label in S2_OUTCOMES.items():
        y = s2[outcome_col].values
        valid_comm = [c for c in communication if c in s2.columns]
        r2_comm, n = quick_cv_r2(s2[valid_comm].values, y)
        r2_pers, _ = quick_cv_r2(s2[valid_pers].values, y)
        r2_pers_comm, _ = quick_cv_r2(s2[valid_pers + valid_comm].values, y)
        print(f"    {outcome_label}: Comm alone={r2_comm:.4f}, "
              f"Pers+Comm={r2_pers_comm:.4f} (Δ={r2_pers_comm - r2_pers:+.4f})", flush=True)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "s2_deep_dive.csv", index=False)
    print(f"  Saved: {OUT / 's2_deep_dive.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 24: Weekly Concurrent Prediction (Panel)
# ═══════════════════════════════════════════════════════════════════════
def run_weekly_concurrent():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 24: Weekly Concurrent Prediction (Panel)", flush=True)
    print("=" * 70, flush=True)

    FEATURE_MAP = [
        ("steps.csv", "avgsumsteps:14dhist", "steps"),
        ("sleep.csv", "avgdurationasleepmain:14dhist", "sleep_dur"),
        ("sleep.csv", "avgefficiencymain:14dhist", "sleep_eff"),
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
    weekly = weekly.dropna(subset=["phq4"])

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

    # Build panel: week × person with sensing features
    print("  Building weekly panel...", flush=True)
    panel_rows = []
    for pid in weekly["pid"].unique():
        phq_pid = weekly[weekly["pid"] == pid].sort_values("date")
        for _, row in phq_pid.iterrows():
            phq_date = row["date"]
            rec = {"pid": pid, "phq4": row["phq4"]}
            for feat_name, daily_df in daily_features.items():
                sens_pid = daily_df[daily_df["pid"] == pid]
                week = sens_pid[(sens_pid["date"] >= phq_date - pd.Timedelta(days=7)) &
                                (sens_pid["date"] <= phq_date)]
                rec[feat_name] = week["value"].mean() if len(week) >= 2 else np.nan
            panel_rows.append(rec)

    panel = pd.DataFrame(panel_rows)
    feat_cols = list(daily_features.keys())
    panel = panel.dropna(subset=["phq4"] + feat_cols)

    # Merge personality
    s3_pers = s3[["pid"] + TRAITS].drop_duplicates("pid")
    panel = panel.merge(s3_pers, on="pid", how="inner")
    valid_pers = [c for c in TRAITS if c in panel.columns]

    print(f"  Panel: {len(panel)} obs, {panel['pid'].nunique()} persons", flush=True)

    # Between-person prediction (pooled panel, ignoring repeated measures)
    y = panel["phq4"].values

    r2_sens, n = quick_cv_r2(panel[feat_cols].values, y, n_splits=5, n_repeats=3)
    r2_pers, _ = quick_cv_r2(panel[valid_pers].values, y, n_splits=5, n_repeats=3)
    r2_both, _ = quick_cv_r2(panel[valid_pers + feat_cols].values, y, n_splits=5, n_repeats=3)

    print(f"\n  Pooled weekly prediction (N_obs={n}):", flush=True)
    print(f"    Sensing: R²={r2_sens:.4f}", flush=True)
    print(f"    Personality: R²={r2_pers:.4f}", flush=True)
    print(f"    Combined: R²={r2_both:.4f}", flush=True)

    # Person-mean centered (within-person only)
    panel_centered = panel.copy()
    person_means = panel.groupby("pid")[["phq4"] + feat_cols].transform("mean")
    for c in ["phq4"] + feat_cols:
        panel_centered[f"{c}_centered"] = panel[c] - person_means[c]

    y_c = panel_centered["phq4_centered"].values
    X_c = panel_centered[[f"{c}_centered" for c in feat_cols]].values

    r2_within, _ = quick_cv_r2(X_c, y_c, n_splits=5, n_repeats=3)
    print(f"    Within-person (centered): R²={r2_within:.4f}", flush=True)

    results = {
        "N_obs": n, "N_persons": panel["pid"].nunique(),
        "R2_sensing_pooled": r2_sens, "R2_personality_pooled": r2_pers,
        "R2_combined_pooled": r2_both, "R2_within_person_centered": r2_within,
    }
    pd.DataFrame([results]).to_csv(OUT / "weekly_concurrent.csv", index=False)
    print(f"  Saved: {OUT / 'weekly_concurrent.csv'}", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 25: Nonlinear Models on Best Modality (Sleep)
# ═══════════════════════════════════════════════════════════════════════
def run_nonlinear_sleep():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 25: Nonlinear Models on Sleep (Best Modality)", flush=True)
    print("=" * 70, flush=True)

    # S3 sleep features
    sleep_cols_s3 = ["sleep_duration_avg", "sleep_efficiency",
                     "sleep_onset_duration", "sleep_episode_count"]
    sleep_cols_s3 = [c for c in sleep_cols_s3 if c in s3.columns]

    # S2 sleep features
    sleep_cols_s2 = [c for c in S2_BEH_RAW if "sleep" in c and "n_days" not in c]

    valid_pers = [c for c in TRAITS if c in s3.columns]
    rows = []

    for study, df, sleep_cols, outcomes in [
        ("S3", s3, sleep_cols_s3, {"bdi2_total": "BDI-II", "stai_state": "STAI", "pss_10": "PSS-10"}),
        ("S2", s2, sleep_cols_s2, {"cesd_total": "CES-D", "stai_trait_total": "STAI", "bai_total": "BAI"}),
    ]:
        vp = [c for c in TRAITS if c in df.columns]
        vs = [c for c in sleep_cols if c in df.columns]
        if not vs:
            continue

        for outcome_col, outcome_label in outcomes.items():
            if outcome_col not in df.columns:
                continue

            sub = df[vp + vs + [outcome_col]].dropna()
            y = sub[outcome_col].values
            X_sleep = sub[vs].values
            X_pers_sleep = sub[vp + vs].values

            # Ridge (linear baseline)
            r2_ridge, n = quick_cv_r2(X_pers_sleep, y)

            # Random Forest
            cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RS)
            rf_r2s, gb_r2s = [], []
            for tr, te in cv.split(X_pers_sleep):
                sc = StandardScaler()
                Xtr = sc.fit_transform(X_pers_sleep[tr])
                Xte = sc.transform(X_pers_sleep[te])

                rf = RandomForestRegressor(n_estimators=200, max_depth=5,
                                            min_samples_leaf=10, random_state=RS)
                rf.fit(Xtr, y[tr])
                rf_r2s.append(r2_score(y[te], rf.predict(Xte)))

                gb = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                learning_rate=0.1, random_state=RS)
                gb.fit(Xtr, y[tr])
                gb_r2s.append(r2_score(y[te], gb.predict(Xte)))

            r2_rf = float(np.mean(rf_r2s))
            r2_gb = float(np.mean(gb_r2s))

            # Sleep only (nonlinear)
            rf_sleep_r2s = []
            for tr, te in cv.split(X_sleep):
                sc = StandardScaler()
                Xtr = sc.fit_transform(X_sleep[tr])
                Xte = sc.transform(X_sleep[te])
                rf = RandomForestRegressor(n_estimators=200, max_depth=5,
                                            min_samples_leaf=10, random_state=RS)
                rf.fit(Xtr, y[tr])
                rf_sleep_r2s.append(r2_score(y[te], rf.predict(Xte)))
            r2_rf_sleep_only = float(np.mean(rf_sleep_r2s))

            print(f"  {study} {outcome_label} (N={n}): "
                  f"Ridge={r2_ridge:.4f}, RF={r2_rf:.4f}, GB={r2_gb:.4f}, "
                  f"RF(sleep-only)={r2_rf_sleep_only:.4f}", flush=True)

            rows.append({
                "Study": study, "Outcome": outcome_label, "N": n,
                "N_sleep_features": len(vs),
                "R2_ridge_pers_sleep": r2_ridge,
                "R2_RF_pers_sleep": r2_rf, "R2_GB_pers_sleep": r2_gb,
                "R2_RF_sleep_only": r2_rf_sleep_only,
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "nonlinear_sleep.csv", index=False)
    print(f"  Saved: {OUT / 'nonlinear_sleep.csv'}", flush=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    analyses = {
        "21": ("Personality × Sensing Interaction", run_interaction),
        "22": ("Person-Centered Features", run_ipsative),
        "23": ("S2 Deep Dive", run_s2_deep_dive),
        "25": ("Nonlinear Sleep", run_nonlinear_sleep),
        "24": ("Weekly Concurrent", run_weekly_concurrent),
        "20": ("Idiographic Models", run_idiographic),
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
        for key in ["21", "22", "23", "25", "24", "20"]:
            name, fn = analyses[key]
            print(f"\n{'#' * 70}", flush=True)
            print(f"# Analysis {key}: {name}", flush=True)
            print(f"{'#' * 70}", flush=True)
            fn()

    print("\n" + "=" * 70, flush=True)
    print("All Phase 16d analyses complete.", flush=True)
    print(f"Results in: {OUT}", flush=True)
    print("=" * 70, flush=True)
