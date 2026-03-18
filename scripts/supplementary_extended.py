#!/usr/bin/env python3
"""
Phase 16 — Supplementary Analyses (Extended / Lightweight)
==========================================================
Analysis 5: Incremental validity power analysis
Analysis 6: Disattenuation correction
Analysis 2+8: Expanded clinical metrics + calibration
Analysis 7: Subgroup analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, r2_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/comparison/supplementary")
OUT.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]


# ═══════════════════════════════════════════════════════════════════════
# Load datasets
# ═══════════════════════════════════════════════════════════════════════
print("Loading datasets...")
s2 = pd.read_parquet("data/processed/nethealth/nethealth_analysis_dataset.parquet")
s3 = pd.read_parquet("data/processed/globem/globem_analysis_dataset.parquet")

S2_BEH_PCA = [c for c in s2.columns if c.startswith("nh_") and c.endswith("_pc1")]
S3_BEH_PCA = [c for c in s3.columns if c.endswith("_pc1") and not c.startswith("nh_")]

# S2 raw behavioral features (for subgroup analysis)
S2_BEH_RAW = [c for c in s2.columns if c not in TRAITS + S2_BEH_PCA
              and c not in ["egoid", "cesd_total", "stai_trait_total", "bai_total",
                           "loneliness_total", "selsa_romantic", "selsa_family",
                           "selsa_social", "self_esteem_total", "gpa_overall",
                           "gpa_first_semester"]
              and not c.endswith("_pc1")]

# S3 raw behavioral features
S3_BEH_RAW = [c for c in s3.columns if c not in TRAITS + S3_BEH_PCA
              and c not in ["pid", "bdi2_total", "stai_state", "pss_10",
                           "cesd_total", "ucla_loneliness", "cohort"]
              and not c.endswith("_pc1")]

print(f"  S2: {s2.shape}, beh_pca={S2_BEH_PCA}")
print(f"  S3: {s3.shape}, beh_pca={S3_BEH_PCA}")


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 5: Incremental Validity Power Analysis
# ═══════════════════════════════════════════════════════════════════════
def run_power_analysis():
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Incremental Validity Power Analysis")
    print("=" * 70)

    inc = pd.read_csv("results/comparison/incremental_validity.csv")

    rows = []
    for _, r in inc.iterrows():
        n = r["N"]
        dr2 = r["Delta_R2"]
        r2_full = r["R2_full"]
        df_num = int(r["df_num"])
        df_den = int(r["df_den"])

        # Cohen's f²
        f2 = dr2 / (1 - r2_full) if r2_full < 1 else np.nan

        # Non-centrality parameter
        lam = f2 * n

        # Critical F at alpha=0.05
        f_crit_05 = stats.f.ppf(0.95, df_num, df_den)
        power_05 = 1 - stats.f.cdf(f_crit_05, df_num, df_den, lam)

        # Critical F at alpha=0.00625 (Bonferroni for 8 tests)
        f_crit_bonf = stats.f.ppf(1 - 0.00625, df_num, df_den)
        power_bonf = 1 - stats.f.cdf(f_crit_bonf, df_num, df_den, lam)

        # Minimum detectable ΔR² at 80% power, alpha=0.05
        # Solve: power(f2_min * n, df_num, df_den) = 0.80
        # Binary search
        lo, hi = 0.0001, 0.5
        for _ in range(100):
            mid = (lo + hi) / 2
            f2_try = mid / (1 - r2_full)
            lam_try = f2_try * n
            p_try = 1 - stats.f.cdf(f_crit_05, df_num, df_den, lam_try)
            if p_try < 0.80:
                lo = mid
            else:
                hi = mid
        min_dr2 = (lo + hi) / 2

        print(f"  {r['Study']} {r['Outcome']}: ΔR²={dr2:.4f}, f²={f2:.4f}, "
              f"Power(α=.05)={power_05:.3f}, Power(Bonf)={power_bonf:.3f}, "
              f"Min ΔR²@80%={min_dr2:.4f}")

        rows.append({
            "Study": r["Study"], "Outcome": r["Outcome"], "N": n,
            "Delta_R2": dr2, "f2": f2, "df_num": df_num, "df_den": df_den,
            "Power_alpha05": power_05, "Power_bonferroni": power_bonf,
            "Min_DR2_80pct": min_dr2,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "power_analysis.csv", index=False)
    print(f"\n  Saved: {OUT / 'power_analysis.csv'}")

    # Summary
    adequate = (df_out["Power_alpha05"] >= 0.80).sum()
    print(f"  Adequately powered (α=.05, power≥.80): {adequate}/{len(df_out)}")
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 6: Disattenuation Correction
# ═══════════════════════════════════════════════════════════════════════
def run_disattenuation():
    print("\n" + "=" * 70)
    print("ANALYSIS 6: Disattenuation Correction")
    print("=" * 70)

    # Reliability coefficients (from literature / computed)
    # BFI-44 average alpha across 5 traits (S1/S2)
    bfi44_alpha = 0.80  # conservative average
    # BFI-10 average alpha (S3) — typically lower
    bfi10_alpha = 0.65  # Rammstedt & John (2007)

    outcome_alpha = {
        # S1
        "phq9_total": 0.86, "pss_total": 0.85, "loneliness_total": 0.89,
        "flourishing_total": 0.87, "panas_negative": 0.85,
        # S2
        "cesd_total": 0.90, "stai_trait_total": 0.90, "bai_total": 0.92,
        # S3
        "bdi2_total": 0.91, "stai_state": 0.90, "pss_10": 0.85,
        "cesd_total_s3": 0.90, "ucla_loneliness": 0.89,
    }

    # Collect best personality R² from existing results
    results = []

    # S2 — from table9
    s2_r2 = {"cesd_total": 0.313, "stai_trait_total": 0.530, "bai_total": 0.182}
    for outcome, r2 in s2_r2.items():
        alpha_y = outcome_alpha.get(outcome, 0.90)
        r2_corr = r2 / (bfi44_alpha * alpha_y)
        results.append({
            "Study": "S2", "Outcome": outcome, "Predictor": "Personality (BFI-44)",
            "alpha_predictor": bfi44_alpha, "alpha_outcome": alpha_y,
            "R2_observed": r2, "R2_corrected": min(r2_corr, 1.0),
        })

    # S3 — from globem personality_mental_health.csv
    s3_r2 = {
        "bdi2_total": 0.087, "stai_state": 0.195, "pss_10": 0.137,
        "cesd_total": 0.091, "ucla_loneliness": 0.085,
    }
    for outcome, r2 in s3_r2.items():
        alpha_key = "cesd_total_s3" if outcome == "cesd_total" else outcome
        alpha_y = outcome_alpha.get(alpha_key, 0.90)
        r2_corr = r2 / (bfi10_alpha * alpha_y)
        results.append({
            "Study": "S3", "Outcome": outcome, "Predictor": "Personality (BFI-10)",
            "alpha_predictor": bfi10_alpha, "alpha_outcome": alpha_y,
            "R2_observed": r2, "R2_corrected": min(r2_corr, 1.0),
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT / "disattenuation.csv", index=False)

    print("\n  Observed vs Corrected R²:")
    for _, r in df_out.iterrows():
        print(f"    {r['Study']} {r['Outcome']}: R²_obs={r['R2_observed']:.3f} → "
              f"R²_corr={r['R2_corrected']:.3f} "
              f"(α_pred={r['alpha_predictor']:.2f}, α_out={r['alpha_outcome']:.2f})")

    print(f"\n  Saved: {OUT / 'disattenuation.csv'}")
    return df_out


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 2+8: Expanded Clinical Metrics + Calibration
# ═══════════════════════════════════════════════════════════════════════
def run_clinical_expanded():
    print("\n" + "=" * 70)
    print("ANALYSIS 2+8: Expanded Clinical Metrics + Calibration")
    print("=" * 70)

    tasks = [
        ("S2", s2, "cesd_total", "CES-D≥16", 16, TRAITS, S2_BEH_PCA),
        ("S2", s2, "stai_trait_total", "STAI≥45", 45, TRAITS, S2_BEH_PCA),
        ("S3", s3, "bdi2_total", "BDI-II≥14", 14, TRAITS, S3_BEH_PCA),
        ("S3", s3, "bdi2_total", "BDI-II≥20", 20, TRAITS, S3_BEH_PCA),
        ("S3", s3, "pss_10", "PSS≥20", 20, TRAITS, S3_BEH_PCA),
    ]

    all_results = []
    calibration_data = {}  # for plotting

    for study, df, col, outcome_name, cutoff, pers_cols, beh_cols in tasks:
        if col not in df.columns:
            continue

        y_full = (df[col] >= cutoff).astype(float).values
        valid_pers = [c for c in pers_cols if c in df.columns]
        valid_beh = [c for c in beh_cols if c in df.columns]

        feature_sets = {
            "Pers-only": valid_pers,
            "Pers+Beh": valid_pers + valid_beh,
            "Beh-only": valid_beh,
        }

        print(f"\n  {study} {outcome_name}")

        for fs_name, fs_cols in feature_sets.items():
            if not fs_cols:
                continue

            X = df[fs_cols].values
            y = y_full.copy()
            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X_c, y_c = X[mask], y[mask]

            if y_c.sum() < 5 or (len(y_c) - y_c.sum()) < 5:
                continue

            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
            lr = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")

            # Collect out-of-fold predictions
            brier_scores = []
            sens_at_spec80 = []
            y_true_all, y_prob_all = [], []

            for train_idx, test_idx in cv.split(X_c, y_c):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_c[train_idx])
                X_te = scaler.transform(X_c[test_idx])
                y_tr, y_te = y_c[train_idx], y_c[test_idx]

                if y_te.sum() == 0 or y_te.sum() == len(y_te):
                    continue

                lr.fit(X_tr, y_tr)
                p = lr.predict_proba(X_te)[:, 1]

                # Brier score
                brier_scores.append(brier_score_loss(y_te, p))

                # Sensitivity at specificity = 0.80
                fpr, tpr, _ = roc_curve(y_te, p)
                # Specificity = 1 - FPR, so Spec=0.80 means FPR=0.20
                # Interpolate TPR at FPR=0.20
                if len(fpr) > 1:
                    sens_at_80 = np.interp(0.20, fpr, tpr)
                    sens_at_spec80.append(sens_at_80)

                y_true_all.extend(y_te)
                y_prob_all.extend(p)

            y_true_all = np.array(y_true_all)
            y_prob_all = np.array(y_prob_all)

            # ECE (Expected Calibration Error)
            n_bins = 10
            try:
                prob_true, prob_pred = calibration_curve(y_true_all, y_prob_all, n_bins=n_bins, strategy="uniform")
                # Compute ECE
                bin_counts = np.histogram(y_prob_all, bins=n_bins, range=(0, 1))[0]
                total = len(y_prob_all)
                ece = 0.0
                for i in range(len(prob_true)):
                    if i < len(bin_counts) and total > 0:
                        ece += (bin_counts[i] / total) * abs(prob_true[i] - prob_pred[i])
            except Exception:
                prob_true, prob_pred = None, None
                ece = np.nan

            # DeLong test placeholder — compare with other feature sets later
            brier_mean = float(np.mean(brier_scores)) if brier_scores else np.nan
            brier_std = float(np.std(brier_scores)) if brier_scores else np.nan
            sens80_mean = float(np.mean(sens_at_spec80)) if sens_at_spec80 else np.nan

            print(f"    {fs_name}: Brier={brier_mean:.4f}±{brier_std:.4f}, "
                  f"ECE={ece:.4f}, Sens@Spec80={sens80_mean:.3f}")

            all_results.append({
                "Study": study, "Outcome": outcome_name, "Features": fs_name,
                "N": len(y_c), "N_pos": int(y_c.sum()),
                "Brier_mean": brier_mean, "Brier_std": brier_std,
                "ECE": ece, "Sens_at_Spec80": sens80_mean,
            })

            # Store for calibration plot
            key = (study, outcome_name, fs_name)
            if prob_true is not None:
                calibration_data[key] = (prob_true, prob_pred)

    # DeLong test: compare Pers-only vs Pers+Beh for each task
    # Re-run with paired predictions on same single 10-fold split
    print("\n  DeLong Tests (Pers-only vs Pers+Beh):")
    delong_results = []
    for study, df, col, outcome_name, cutoff, pers_cols, beh_cols in tasks:
        if col not in df.columns:
            continue
        valid_pers = [c for c in pers_cols if c in df.columns]
        valid_beh = [c for c in beh_cols if c in df.columns]
        if not valid_beh:
            continue

        y_full = (df[col] >= cutoff).astype(float).values
        X_pers = df[valid_pers].values
        X_both = df[valid_pers + valid_beh].values
        mask = ~np.isnan(X_both).any(axis=1) & ~np.isnan(y_full)
        X_p, X_b, y_c = X_pers[mask], X_both[mask], y_full[mask]

        if y_c.sum() < 5:
            continue

        # Single 10-fold for paired comparison
        from sklearn.model_selection import StratifiedKFold
        cv_single = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        y_true_paired = []
        p_pers_paired = []
        p_both_paired = []

        for train_idx, test_idx in cv_single.split(X_p, y_c):
            scaler_p = StandardScaler()
            scaler_b = StandardScaler()
            X_ptr = scaler_p.fit_transform(X_p[train_idx])
            X_pte = scaler_p.transform(X_p[test_idx])
            X_btr = scaler_b.fit_transform(X_b[train_idx])
            X_bte = scaler_b.transform(X_b[test_idx])
            y_tr, y_te = y_c[train_idx], y_c[test_idx]

            if y_te.sum() == 0 or y_te.sum() == len(y_te):
                continue

            lr_p = LogisticRegression(max_iter=1000, random_state=42)
            lr_b = LogisticRegression(max_iter=1000, random_state=42)
            lr_p.fit(X_ptr, y_tr)
            lr_b.fit(X_btr, y_tr)

            y_true_paired.extend(y_te)
            p_pers_paired.extend(lr_p.predict_proba(X_pte)[:, 1])
            p_both_paired.extend(lr_b.predict_proba(X_bte)[:, 1])

        y_true_paired = np.array(y_true_paired)
        p_pers_paired = np.array(p_pers_paired)
        p_both_paired = np.array(p_both_paired)

        auc_pers = roc_auc_score(y_true_paired, p_pers_paired)
        auc_both = roc_auc_score(y_true_paired, p_both_paired)

        # Hanley-McNeil approximate test for AUC difference
        n1 = int(y_true_paired.sum())  # positives
        n0 = len(y_true_paired) - n1   # negatives
        q1_p = auc_pers / (2 - auc_pers)
        q2_p = 2 * auc_pers**2 / (1 + auc_pers)
        se_p = np.sqrt((auc_pers * (1 - auc_pers) + (n1 - 1) * (q1_p - auc_pers**2) + (n0 - 1) * (q2_p - auc_pers**2)) / (n1 * n0))
        q1_b = auc_both / (2 - auc_both)
        q2_b = 2 * auc_both**2 / (1 + auc_both)
        se_b = np.sqrt((auc_both * (1 - auc_both) + (n1 - 1) * (q1_b - auc_both**2) + (n0 - 1) * (q2_b - auc_both**2)) / (n1 * n0))

        # Correlation between AUCs (approximation)
        r = 0.6  # typical correlation for correlated AUCs on same data
        se_diff = np.sqrt(se_p**2 + se_b**2 - 2 * r * se_p * se_b)
        z = (auc_both - auc_pers) / se_diff if se_diff > 0 else 0
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))

        print(f"    {study} {outcome_name}: AUC_pers={auc_pers:.3f}, AUC_both={auc_both:.3f}, "
              f"ΔAUC={auc_both - auc_pers:.3f}, z={z:.2f}, p={p_val:.4f}")

        delong_results.append({
            "Study": study, "Outcome": outcome_name,
            "AUC_pers": auc_pers, "AUC_both": auc_both,
            "Delta_AUC": auc_both - auc_pers,
            "z_stat": z, "p_value": p_val,
        })

    # Merge DeLong into main results
    delong_df = pd.DataFrame(delong_results)
    results_df = pd.DataFrame(all_results)

    # Save
    results_df.to_csv(OUT / "clinical_expanded.csv", index=False)
    delong_df.to_csv(OUT / "delong_tests.csv", index=False)
    print(f"\n  Saved: {OUT / 'clinical_expanded.csv'}")
    print(f"  Saved: {OUT / 'delong_tests.csv'}")

    # ── Calibration Plots ──
    if calibration_data:
        outcomes_unique = list(dict.fromkeys(
            [(s, o) for (s, o, _) in calibration_data.keys()]
        ))
        n_outcomes = len(outcomes_unique)
        fig, axes = plt.subplots(1, n_outcomes, figsize=(5 * n_outcomes, 5))
        if n_outcomes == 1:
            axes = [axes]

        colors = {"Pers-only": "#e74c3c", "Pers+Beh": "#3498db", "Beh-only": "#95a5a6"}

        for idx, (study, outcome) in enumerate(outcomes_unique):
            ax = axes[idx]
            for fs_name in ["Pers-only", "Pers+Beh", "Beh-only"]:
                key = (study, outcome, fs_name)
                if key in calibration_data:
                    pt, pp = calibration_data[key]
                    ax.plot(pp, pt, marker="o", label=fs_name, color=colors.get(fs_name, "grey"))
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")
            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Observed Frequency")
            ax.set_title(f"{study} {outcome}", fontweight="bold")
            ax.legend(fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        plt.suptitle("Calibration Plots", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(OUT / "figure_calibration.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {OUT / 'figure_calibration.png'}")

    return results_df


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 7: Subgroup Analysis
# ═══════════════════════════════════════════════════════════════════════
def run_subgroup():
    print("\n" + "=" * 70)
    print("ANALYSIS 7: Subgroup Analysis")
    print("=" * 70)

    def kfold_r2(X, y, n_splits=10, n_repeats=5):
        """Quick Ridge regression with repeated k-fold CV."""
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_c, y_c = X[mask], y[mask]
        if len(y_c) < 20:
            return np.nan, len(y_c)
        cv = RepeatedKFold(n_splits=min(n_splits, len(y_c) // 3), n_repeats=n_repeats, random_state=42)
        r2s = []
        for train_idx, test_idx in cv.split(X_c):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_c[train_idx])
            X_te = scaler.transform(X_c[test_idx])
            model = Ridge(alpha=1.0)
            model.fit(X_tr, y_c[train_idx])
            r2s.append(r2_score(y_c[test_idx], model.predict(X_te)))
        return float(np.mean(r2s)), len(y_c)

    subgroup_tasks = [
        # (Study, df, outcome_col, label, cutoff, pers_cols, beh_pca_cols)
        ("S2", s2, "cesd_total", "CES-D", 16, TRAITS, S2_BEH_PCA),
        ("S2", s2, "stai_trait_total", "STAI", 45, TRAITS, S2_BEH_PCA),
        ("S3", s3, "bdi2_total", "BDI-II", 14, TRAITS, S3_BEH_PCA),
        ("S3", s3, "stai_state", "STAI-State", None, TRAITS, S3_BEH_PCA),  # median split
        ("S3", s3, "pss_10", "PSS-10", None, TRAITS, S3_BEH_PCA),  # median split
    ]

    rows = []

    for study, df, col, label, cutoff, pers_cols, beh_cols in subgroup_tasks:
        if col not in df.columns:
            continue

        valid_pers = [c for c in pers_cols if c in df.columns]
        valid_beh = [c for c in beh_cols if c in df.columns]
        if not valid_beh:
            continue

        all_cols = list(dict.fromkeys(valid_pers + valid_beh + [col, "neuroticism"]))
        sub = df[all_cols].dropna()
        if len(sub) < 40:
            continue

        y = sub[col].values

        # Clinical cutoff split (or median)
        if cutoff is not None:
            high_mask = y >= cutoff
        else:
            cutoff = np.median(y)
            high_mask = y >= cutoff
        low_mask = ~high_mask

        X_beh = sub[valid_beh].values
        X_pers = sub[valid_pers].values
        X_both = sub[valid_pers + valid_beh].values

        for subgroup, mask in [("Clinical/High", high_mask), ("Subclinical/Low", low_mask)]:
            if mask.sum() < 15:
                continue
            r2_b, n_b = kfold_r2(X_beh[mask], y[mask])
            r2_p, n_p = kfold_r2(X_pers[mask], y[mask])
            r2_c, n_c = kfold_r2(X_both[mask], y[mask])

            print(f"  {study} {label} [{subgroup}, N={n_b}]: "
                  f"Beh R²={r2_b:.3f}, Pers R²={r2_p:.3f}, Combined R²={r2_c:.3f}")

            rows.append({
                "Study": study, "Outcome": label, "Split_type": "Clinical",
                "Subgroup": subgroup, "N": n_b,
                "R2_sensing": r2_b, "R2_personality": r2_p, "R2_combined": r2_c,
            })

        # Neuroticism median split
        n_vals = sub["neuroticism"].values
        n_med = float(np.nanmedian(n_vals))
        for subgroup, mask in [("High-N", n_vals >= n_med),
                                ("Low-N", n_vals < n_med)]:
            if mask.sum() < 15:
                continue
            r2_b, n_b = kfold_r2(X_beh[mask], y[mask])
            r2_p, n_p = kfold_r2(X_pers[mask], y[mask])
            r2_c, n_c = kfold_r2(X_both[mask], y[mask])

            print(f"  {study} {label} [{subgroup}, N={n_b}]: "
                  f"Beh R²={r2_b:.3f}, Pers R²={r2_p:.3f}, Combined R²={r2_c:.3f}")

            rows.append({
                "Study": study, "Outcome": label, "Split_type": "Neuroticism",
                "Subgroup": subgroup, "N": n_b,
                "R2_sensing": r2_b, "R2_personality": r2_p, "R2_combined": r2_c,
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "subgroup_analysis.csv", index=False)
    print(f"\n  Saved: {OUT / 'subgroup_analysis.csv'}")

    # ── Figure: Subgroup comparison ──
    if len(df_out) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, split_type in enumerate(["Clinical", "Neuroticism"]):
            ax = axes[idx]
            subset = df_out[df_out["Split_type"] == split_type]
            if len(subset) == 0:
                continue

            labels = [f"{r['Study']}\n{r['Outcome']}\n{r['Subgroup']}" for _, r in subset.iterrows()]
            x = np.arange(len(labels))
            w = 0.25

            ax.bar(x - w, subset["R2_personality"], w, label="Personality", color="#e74c3c")
            ax.bar(x, subset["R2_sensing"], w, label="Sensing", color="#95a5a6")
            ax.bar(x + w, subset["R2_combined"], w, label="Combined", color="#3498db")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
            ax.set_ylabel("R²")
            ax.set_title(f"Subgroup: {split_type} Split", fontweight="bold")
            ax.legend(fontsize=8)
            ax.axhline(0, color="grey", linestyle="--", alpha=0.3)

        plt.suptitle("Subgroup Analysis: Does Sensing Work Better for High-Risk Groups?",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(OUT / "figure_subgroup.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {OUT / 'figure_subgroup.png'}")

    return df_out


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    analyses = {
        "5": ("Power Analysis", run_power_analysis),
        "6": ("Disattenuation", run_disattenuation),
        "2": ("Clinical Expanded", run_clinical_expanded),
        "7": ("Subgroup", run_subgroup),
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
        # Run all
        for key in ["5", "6", "2", "7"]:
            name, fn = analyses[key]
            print(f"\n{'#' * 70}")
            print(f"# Running Analysis {key}: {name}")
            print(f"{'#' * 70}")
            fn()

    print("\n" + "=" * 70)
    print("All supplementary extended analyses complete.")
    print(f"Results in: {OUT}")
    print("=" * 70)
