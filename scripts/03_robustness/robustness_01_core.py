#!/usr/bin/env python3
"""
Phase 16 — Supplementary Analyses (Core / Computationally Heavy)
================================================================
Analysis 1: Raw RAPIDS features vs PCA (S3) — ~40-60 min
Analysis 3: Sensing feature reliability (ICC/split-half)
Analysis 4: Feature ablation by modality (S2 + S3)
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.linear_model import Ridge, RidgeCV, ElasticNetCV
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold
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
RANDOM_STATE = 42


# ═══════════════════════════════════════════════════════════════════════
# Load datasets
# ═══════════════════════════════════════════════════════════════════════
print("Loading datasets...")
s2 = pd.read_parquet("data/processed/nethealth/nethealth_analysis_dataset.parquet")
s3 = pd.read_parquet("data/processed/globem/globem_analysis_dataset.parquet")

S2_BEH_PCA = [c for c in s2.columns if c.startswith("nh_") and c.endswith("_pc1")]
S3_BEH_PCA = [c for c in s3.columns if c.endswith("_pc1") and not c.startswith("nh_")]

# S3 outcomes
S3_OUTCOMES = {
    "bdi2_total": "BDI-II", "stai_state": "STAI",
    "pss_10": "PSS-10", "cesd_total": "CESD", "ucla_loneliness": "UCLA",
}

# S2 outcomes
S2_OUTCOMES = {
    "cesd_total": "CES-D", "stai_trait_total": "STAI", "bai_total": "BAI",
}

print(f"  S2: {s2.shape}, S3: {s3.shape}")


# ═══════════════════════════════════════════════════════════════════════
# Shared: kfold_predict with pipeline support
# ═══════════════════════════════════════════════════════════════════════
def kfold_r2(X, y, model_name="ridge", n_splits=10, n_repeats=10,
             preprocess_fn=None, verbose=False):
    """
    Repeated k-fold CV with StandardScaler inside fold.
    Optionally applies preprocess_fn(X_train, X_test) inside each fold.
    Returns dict with R2_mean, R2_std, R2_ci_lo, R2_ci_hi.
    """
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X_c, y_c = X[mask], y[mask]

    if len(y_c) < 30:
        return {"R2_mean": np.nan, "R2_std": np.nan,
                "R2_ci_lo": np.nan, "R2_ci_hi": np.nan, "N": len(y_c)}

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    r2s = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X_c)):
        X_tr, X_te = X_c[train_idx], X_c[test_idx]
        y_tr, y_te = y_c[train_idx], y_c[test_idx]

        # Impute NaN with training median (for RAPIDS features)
        col_medians = np.nanmedian(X_tr, axis=0)
        nan_mask_tr = np.isnan(X_tr)
        nan_mask_te = np.isnan(X_te)
        for j in range(X_tr.shape[1]):
            X_tr[nan_mask_tr[:, j], j] = col_medians[j]
            X_te[nan_mask_te[:, j], j] = col_medians[j]

        # Scale
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        # Optional preprocessing (e.g., PCA)
        if preprocess_fn is not None:
            X_tr, X_te = preprocess_fn(X_tr, X_te)

        # Model
        if model_name == "ridge":
            model = Ridge(alpha=1.0)
        elif model_name == "ridgecv":
            model = RidgeCV(alphas=np.logspace(-3, 3, 20))
        elif model_name == "elasticnet":
            model = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                                 n_alphas=20, cv=5, random_state=RANDOM_STATE,
                                 max_iter=5000)
        else:
            model = Ridge(alpha=1.0)

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        r2s.append(r2_score(y_te, y_pred))

    r2s = np.array(r2s)
    return {
        "R2_mean": float(np.mean(r2s)),
        "R2_std": float(np.std(r2s)),
        "R2_ci_lo": float(np.percentile(r2s, 2.5)),
        "R2_ci_hi": float(np.percentile(r2s, 97.5)),
        "N": len(y_c),
    }


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Raw RAPIDS Features vs PCA
# ═══════════════════════════════════════════════════════════════════════
def run_rapids_comparison():
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Raw RAPIDS Features vs PCA (S3)")
    print("=" * 70)

    # Load RAPIDS features
    rapids = pd.read_parquet("data/processed/globem/features/rapids_features.parquet")
    print(f"  RAPIDS raw: {rapids.shape}")

    # Merge with S3 outcomes
    merged = s3.merge(rapids, on="pid", how="inner")
    print(f"  Merged (S3 ∩ RAPIDS): {merged.shape[0]} participants")

    # Get RAPIDS feature columns (exclude pid, traits, outcomes, PCA)
    rapids_cols = [c for c in rapids.columns if c != "pid"]

    # Pre-filter: drop columns with >20% missing or variance < 0.01
    valid_cols = []
    for c in rapids_cols:
        vals = merged[c].values
        miss_rate = np.isnan(vals).mean()
        if miss_rate > 0.20:
            continue
        var = np.nanvar(vals)
        if var < 0.01:
            continue
        valid_cols.append(c)

    print(f"  After filtering (>20% missing, var<0.01): {len(valid_cols)} features (from {len(rapids_cols)})")

    # Personality features
    valid_pers = [c for c in TRAITS if c in merged.columns]

    # PCA preprocessing function factory
    def make_pca_fn(n_components=0.90):
        def pca_fn(X_tr, X_te):
            pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
            X_tr_pca = pca.fit_transform(X_tr)
            X_te_pca = pca.transform(X_te)
            return X_tr_pca, X_te_pca
        return pca_fn

    rows = []

    for outcome_col, outcome_label in S3_OUTCOMES.items():
        if outcome_col not in merged.columns:
            continue

        y = merged[outcome_col].values
        print(f"\n  {outcome_label} ({outcome_col}):")

        # (a) Current 5 PCA composites — behavior only
        X_pca5 = merged[S3_BEH_PCA].values
        res = kfold_r2(X_pca5, y, model_name="ridge")
        print(f"    (a) 5 PCA composites (Ridge):    R²={res['R2_mean']:.4f} [{res['R2_ci_lo']:.4f}, {res['R2_ci_hi']:.4f}]")
        rows.append({"Outcome": outcome_label, "Approach": "5 PCA composites",
                     "Model": "Ridge", "Feature_type": "Behavior", **res})

        # (b) PCA 90% variance on raw features
        X_raw = merged[valid_cols].values.copy()
        res = kfold_r2(X_raw, y, model_name="ridgecv", preprocess_fn=make_pca_fn(0.90))
        print(f"    (b) PCA 90% variance (RidgeCV):  R²={res['R2_mean']:.4f} [{res['R2_ci_lo']:.4f}, {res['R2_ci_hi']:.4f}]")
        rows.append({"Outcome": outcome_label, "Approach": "PCA 90% variance",
                     "Model": "RidgeCV", "Feature_type": "Behavior", **res})

        # (c) Raw features + Elastic Net (L1)
        res = kfold_r2(X_raw.copy(), y, model_name="elasticnet")
        print(f"    (c) Raw {len(valid_cols)} feat (ElasticNet): R²={res['R2_mean']:.4f} [{res['R2_ci_lo']:.4f}, {res['R2_ci_hi']:.4f}]")
        rows.append({"Outcome": outcome_label, "Approach": f"Raw {len(valid_cols)} features",
                     "Model": "ElasticNet", "Feature_type": "Behavior", **res})

        # (d) Raw features + Ridge (L2)
        res = kfold_r2(X_raw.copy(), y, model_name="ridgecv")
        print(f"    (d) Raw {len(valid_cols)} feat (RidgeCV):    R²={res['R2_mean']:.4f} [{res['R2_ci_lo']:.4f}, {res['R2_ci_hi']:.4f}]")
        rows.append({"Outcome": outcome_label, "Approach": f"Raw {len(valid_cols)} features",
                     "Model": "RidgeCV", "Feature_type": "Behavior", **res})

        # (e) Personality only (reference)
        X_pers = merged[valid_pers].values
        res = kfold_r2(X_pers, y, model_name="ridge")
        print(f"    (ref) Personality only (Ridge):   R²={res['R2_mean']:.4f} [{res['R2_ci_lo']:.4f}, {res['R2_ci_hi']:.4f}]")
        rows.append({"Outcome": outcome_label, "Approach": "Personality only",
                     "Model": "Ridge", "Feature_type": "Personality", **res})

        # (f) Personality + Raw RAPIDS (ElasticNet)
        X_pers_raw = np.hstack([merged[valid_pers].values, X_raw.copy()])
        res = kfold_r2(X_pers_raw, y, model_name="elasticnet")
        print(f"    (f) Pers + Raw RAPIDS (EN):      R²={res['R2_mean']:.4f} [{res['R2_ci_lo']:.4f}, {res['R2_ci_hi']:.4f}]")
        rows.append({"Outcome": outcome_label, "Approach": f"Pers + Raw {len(valid_cols)}",
                     "Model": "ElasticNet", "Feature_type": "Pers+Behavior", **res})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "rapids_comparison.csv", index=False)
    print(f"\n  Saved: {OUT / 'rapids_comparison.csv'}")

    # ── Figure: RAPIDS comparison ──
    if len(df_out) > 0:
        beh_only = df_out[df_out["Feature_type"] == "Behavior"]
        outcomes = beh_only["Outcome"].unique()
        approaches = beh_only["Approach"].unique()

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(outcomes))
        width = 0.18
        colors = plt.cm.Set2(np.linspace(0, 1, len(approaches)))

        for i, approach in enumerate(approaches):
            vals = []
            for outcome in outcomes:
                row = beh_only[(beh_only["Outcome"] == outcome) & (beh_only["Approach"] == approach)]
                vals.append(row["R2_mean"].values[0] if len(row) > 0 else 0)
            bars = ax.bar(x + i * width, vals, width, label=approach, color=colors[i])
            for j, v in enumerate(vals):
                ax.text(x[j] + i * width, max(v + 0.01, 0.01), f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7, rotation=45)

        ax.set_xticks(x + width * (len(approaches) - 1) / 2)
        ax.set_xticklabels(outcomes, fontsize=10)
        ax.set_ylabel("R² (Behavior-only)")
        ax.set_title("RAPIDS Raw Features vs PCA: Behavior-Only Prediction (S3 GLOBEM)",
                     fontsize=12, fontweight="bold")
        ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, loc="upper right")

        plt.tight_layout()
        fig.savefig(OUT / "figure_rapids_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {OUT / 'figure_rapids_comparison.png'}")

    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Sensing Feature Reliability (Split-Half)
# ═══════════════════════════════════════════════════════════════════════
def run_reliability():
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Sensing Feature Reliability (Split-Half)")
    print("=" * 70)

    COHORTS = ["INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4"]
    RAW_DIR = Path("data/raw/globem")

    # Feature mapping: (modality_file, column_substring, short_name, modality_label)
    FEATURE_MAP = [
        ("steps.csv", "avgsumsteps:14dhist", "steps_avg", "Activity"),
        ("steps.csv", "stdsumsteps:14dhist", "steps_std", "Activity"),
        ("steps.csv", "countepisodeactivebout:14dhist", "active_bout_count", "Activity"),
        ("steps.csv", "sumdurationsedentarybout:14dhist", "sedentary_duration", "Activity"),
        ("sleep.csv", "avgdurationasleepmain:14dhist", "sleep_duration_avg", "Sleep"),
        ("sleep.csv", "avgefficiencymain:14dhist", "sleep_efficiency", "Sleep"),
        ("sleep.csv", "avgdurationtofallasleepmain:14dhist", "sleep_onset_duration", "Sleep"),
        ("sleep.csv", "countepisodemain:14dhist", "sleep_episode_count", "Sleep"),
        ("call.csv", "incoming_count:14dhist", "call_incoming_count", "Communication"),
        ("call.csv", "outgoing_count:14dhist", "call_outgoing_count", "Communication"),
        ("call.csv", "incoming_distinctcontacts:14dhist", "call_incoming_contacts", "Communication"),
        ("call.csv", "outgoing_distinctcontacts:14dhist", "call_outgoing_contacts", "Communication"),
        ("screen.csv", "rapids_countepisodeunlock:14dhist", "screen_unlock_count", "Screen"),
        ("screen.csv", "rapids_sumdurationunlock:14dhist", "screen_duration_total", "Screen"),
        ("screen.csv", "rapids_avgdurationunlock:14dhist", "screen_duration_avg", "Screen"),
        ("location.csv", "barnett_hometime:14dhist", "loc_hometime", "Location"),
        ("location.csv", "barnett_rog:14dhist", "loc_radius_gyration", "Location"),
        ("location.csv", "barnett_siglocsvisited:14dhist", "loc_unique_locations", "Location"),
        ("location.csv", "barnett_siglocentropy:14dhist", "loc_entropy", "Location"),
    ]

    def find_column(columns, substring):
        matches = [c for c in columns if substring in c]
        if not matches:
            return None
        return min(matches, key=len)

    rows = []

    for modality_file, col_substr, short_name, modality_label in FEATURE_MAP:
        print(f"  Processing {short_name} ({modality_label})...", end=" ")

        all_daily = []
        for cohort in COHORTS:
            fpath = RAW_DIR / cohort / "FeatureData" / modality_file
            if not fpath.exists():
                continue
            df = pd.read_csv(fpath, low_memory=False)
            col = find_column(df.columns.tolist(), col_substr)
            if col is None:
                continue
            daily = df[["pid", "date", col]].copy()
            daily.columns = ["pid", "date", "value"]
            daily["value"] = pd.to_numeric(daily["value"], errors="coerce")
            all_daily.append(daily)

        if not all_daily:
            print("no data")
            continue

        daily_df = pd.concat(all_daily, ignore_index=True)
        daily_df = daily_df.dropna(subset=["value"])

        # Assign week number per participant
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_df = daily_df.sort_values(["pid", "date"])
        daily_df["day_num"] = daily_df.groupby("pid").cumcount()
        daily_df["week"] = daily_df["day_num"] // 7

        # Split-half: odd weeks vs even weeks
        odd = daily_df[daily_df["week"] % 2 == 1].groupby("pid")["value"].mean()
        even = daily_df[daily_df["week"] % 2 == 0].groupby("pid")["value"].mean()

        # Align
        common_pids = odd.index.intersection(even.index)
        if len(common_pids) < 30:
            print(f"too few ({len(common_pids)})")
            continue

        odd_vals = odd.loc[common_pids].values
        even_vals = even.loc[common_pids].values

        # Split-half reliability
        r, p = stats.pearsonr(odd_vals, even_vals)
        # Spearman-Brown correction
        r_sb = 2 * r / (1 + r) if (1 + r) != 0 else np.nan

        # Between-person and within-person CV
        person_means = daily_df.groupby("pid")["value"].mean()
        person_stds = daily_df.groupby("pid")["value"].std()
        grand_mean = person_means.mean()

        cv_between = person_means.std() / abs(grand_mean) if abs(grand_mean) > 0.001 else np.nan
        cv_within = (person_stds / person_means.abs().clip(lower=0.001)).mean()

        print(f"r={r:.3f}, r_SB={r_sb:.3f}, CV_btwn={cv_between:.3f}, CV_within={cv_within:.3f} (N={len(common_pids)})")

        rows.append({
            "Feature": short_name, "Modality": modality_label,
            "N_participants": len(common_pids),
            "Split_half_r": r, "Split_half_p": p,
            "Spearman_Brown_r": r_sb,
            "CV_between": cv_between, "CV_within": cv_within,
            "Signal_noise_ratio": cv_between / cv_within if cv_within > 0 else np.nan,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "sensing_reliability.csv", index=False)
    print(f"\n  Saved: {OUT / 'sensing_reliability.csv'}")

    # Summary
    if len(df_out) > 0:
        print(f"\n  Mean split-half r: {df_out['Split_half_r'].mean():.3f}")
        print(f"  Mean Spearman-Brown: {df_out['Spearman_Brown_r'].mean():.3f}")
        print(f"  Mean signal/noise ratio: {df_out['Signal_noise_ratio'].mean():.3f}")

        # Compare to BFI reliability
        bfi_alpha = 0.80  # typical BFI-44 average
        print(f"\n  BFI-44 average α: {bfi_alpha:.2f}")
        n_above = (df_out["Spearman_Brown_r"] >= bfi_alpha).sum()
        print(f"  Sensing features with reliability ≥ BFI α: {n_above}/{len(df_out)}")

    # ── Figure: Reliability comparison ──
    if len(df_out) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))

        modality_colors = {
            "Activity": "#e74c3c", "Sleep": "#3498db", "Communication": "#2ecc71",
            "Screen": "#f39c12", "Location": "#9b59b6",
        }

        x = np.arange(len(df_out))
        colors = [modality_colors.get(m, "grey") for m in df_out["Modality"]]
        bars = ax.bar(x, df_out["Spearman_Brown_r"].values, color=colors, edgecolor="white")

        # BFI reference lines
        ax.axhline(0.80, color="red", linestyle="--", alpha=0.7, label="BFI-44 avg α (0.80)")
        ax.axhline(0.65, color="orange", linestyle="--", alpha=0.7, label="BFI-10 avg α (0.65)")

        ax.set_xticks(x)
        ax.set_xticklabels(df_out["Feature"].values, fontsize=8, rotation=45, ha="right")
        ax.set_ylabel("Spearman-Brown Reliability")
        ax.set_title("Sensing Feature Reliability (Split-Half) vs Personality Questionnaire",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

        # Add modality legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=m) for m, c in modality_colors.items()]
        ax2 = ax.twinx()
        ax2.set_yticks([])
        ax2.legend(handles=legend_elements, loc="upper left", fontsize=8, title="Modality")

        plt.tight_layout()
        fig.savefig(OUT / "figure_reliability.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {OUT / 'figure_reliability.png'}")

    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Feature Ablation by Modality
# ═══════════════════════════════════════════════════════════════════════
def run_ablation():
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Feature Ablation by Modality")
    print("=" * 70)

    # ── S3 GLOBEM modality groups ──
    S3_MODALITIES = {
        "Activity": ["steps_avg", "steps_std", "active_bout_count", "sedentary_duration"],
        "Sleep": ["sleep_duration_avg", "sleep_efficiency", "sleep_onset_duration", "sleep_episode_count"],
        "Communication": ["call_incoming_count", "call_outgoing_count", "call_incoming_contacts", "call_outgoing_contacts"],
        "Screen": ["screen_unlock_count", "screen_duration_total", "screen_duration_avg"],
        "Location": ["loc_hometime", "loc_radius_gyration", "loc_unique_locations", "loc_entropy"],
    }

    # ── S2 NetHealth modality groups ──
    S2_MODALITIES = {
        "Fitbit Activity": [c for c in s2.columns if any(k in c for k in
                           ["steps_", "sedentary_", "light_active", "fairly_active",
                            "very_active", "total_active", "active_ratio", "fatburn", "cardio"])
                           and not c.endswith("_pc1")],
        "Fitbit Sleep": [c for c in s2.columns if any(k in c for k in
                        ["sleep_duration", "sleep_interruptions", "sleep_efficiency",
                         "sleep_onset", "sleep_regularity", "time_to_fall", "sleep_n_days"])
                        and not c.endswith("_pc1")],
        "Communication": [c for c in s2.columns if any(k in c for k in
                         ["call_count", "call_duration", "call_outgoing", "sms_count",
                          "sms_outgoing", "total_unique_contacts", "total_comm"])
                         and not c.endswith("_pc1")],
    }

    rows = []

    # ── S3 ──
    print("\n  --- S3 (GLOBEM) ---")
    valid_pers = [c for c in TRAITS if c in s3.columns]

    for outcome_col, outcome_label in S3_OUTCOMES.items():
        if outcome_col not in s3.columns:
            continue
        y = s3[outcome_col].values

        # Baseline: Personality only
        X_pers = s3[valid_pers].values
        res_base = kfold_r2(X_pers, y, model_name="ridge")
        r2_base = res_base["R2_mean"]

        for mod_name, mod_cols in S3_MODALITIES.items():
            valid_mod = [c for c in mod_cols if c in s3.columns]
            if not valid_mod:
                continue

            X_pers_mod = s3[valid_pers + valid_mod].values
            res = kfold_r2(X_pers_mod, y, model_name="ridge")
            dr2 = res["R2_mean"] - r2_base

            print(f"    S3 {outcome_label} + {mod_name}: ΔR²={dr2:+.4f} "
                  f"(base={r2_base:.4f}, combined={res['R2_mean']:.4f})")

            rows.append({
                "Study": "S3", "Outcome": outcome_label, "Modality": mod_name,
                "N_features": len(valid_mod), "N": res["N"],
                "R2_pers_only": r2_base, "R2_pers_plus_mod": res["R2_mean"],
                "Delta_R2": dr2,
            })

    # ── S2 ──
    print("\n  --- S2 (NetHealth) ---")
    valid_pers = [c for c in TRAITS if c in s2.columns]

    for outcome_col, outcome_label in S2_OUTCOMES.items():
        if outcome_col not in s2.columns:
            continue
        y = s2[outcome_col].values

        X_pers = s2[valid_pers].values
        res_base = kfold_r2(X_pers, y, model_name="ridge")
        r2_base = res_base["R2_mean"]

        for mod_name, mod_cols in S2_MODALITIES.items():
            valid_mod = [c for c in mod_cols if c in s2.columns]
            if not valid_mod:
                continue

            X_pers_mod = s2[valid_pers + valid_mod].values
            res = kfold_r2(X_pers_mod, y, model_name="ridge")
            dr2 = res["R2_mean"] - r2_base

            print(f"    S2 {outcome_label} + {mod_name}: ΔR²={dr2:+.4f} "
                  f"(base={r2_base:.4f}, combined={res['R2_mean']:.4f})")

            rows.append({
                "Study": "S2", "Outcome": outcome_label, "Modality": mod_name,
                "N_features": len(valid_mod), "N": res["N"],
                "R2_pers_only": r2_base, "R2_pers_plus_mod": res["R2_mean"],
                "Delta_R2": dr2,
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "modality_ablation.csv", index=False)
    print(f"\n  Saved: {OUT / 'modality_ablation.csv'}")

    # ── Figure: Ablation heatmap ──
    if len(df_out) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for idx, (study, study_label) in enumerate([("S3", "S3 GLOBEM"), ("S2", "S2 NetHealth")]):
            ax = axes[idx]
            subset = df_out[df_out["Study"] == study]
            if len(subset) == 0:
                ax.set_visible(False)
                continue

            pivot = subset.pivot_table(index="Modality", columns="Outcome",
                                       values="Delta_R2", aggfunc="mean")

            im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                          vmin=-0.05, vmax=0.05)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, fontsize=9, rotation=45, ha="right")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=9)
            ax.set_title(f"{study_label}: ΔR² by Modality", fontweight="bold")

            # Annotate
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                               fontsize=8, color="black" if abs(val) < 0.03 else "white")

            plt.colorbar(im, ax=ax, shrink=0.8, label="ΔR²")

        plt.suptitle("Modality Ablation: Incremental R² of Each Sensing Modality Over Personality",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(OUT / "figure_modality_ablation.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {OUT / 'figure_modality_ablation.png'}")

    return df_out


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    analyses = {
        "1": ("RAPIDS Raw vs PCA", run_rapids_comparison),
        "3": ("Sensing Reliability", run_reliability),
        "4": ("Modality Ablation", run_ablation),
    }

    if len(sys.argv) > 2 and sys.argv[1] == "--analysis":
        key = sys.argv[2]
        if key in analyses:
            name, fn = analyses[key]
            print(f"\nRunning Analysis {key}: {name}")
            fn()
        else:
            print(f"Unknown analysis: {key}. Available: {list(analyses.keys())}")
    else:
        # Run fast analyses first, RAPIDS last
        for key in ["3", "4", "1"]:
            name, fn = analyses[key]
            print(f"\n{'#' * 70}")
            print(f"# Running Analysis {key}: {name}")
            print(f"{'#' * 70}")
            fn()

    print("\n" + "=" * 70)
    print("All supplementary core analyses complete.")
    print(f"Results in: {OUT}")
    print("=" * 70)
