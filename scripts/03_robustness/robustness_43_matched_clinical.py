#!/usr/bin/env python3
"""
Phase 16i — Matched-N Clinical Classification (Plan B for Fig5A)
==================================================================
Re-runs the clinical classification analysis (Section 1 of clinical_utility.py)
on the **intersection sample** where BOTH personality and sensing features are
available. This addresses the N-mismatch issue in Fig5A:

In the original analysis:
  - Pers-only is fit on N=719 (S2 CES-D, full personality sample)
  - Beh-only is fit on N=365 (S2 intersection with sensing data)
  - The intersection happens to have substantially higher symptom prevalence
    (28% vs 20% for CES-D), mechanically inflating Beh-only TP/100.

This script forces both Pers-only and Beh-only to use the same intersection
mask, so prevalence and N are matched. Results are written to
results/robustness/clinical_classification_matched.csv and
results/robustness/nns_comparison_matched.csv (parallel files; the original
results/core/clinical_classification.csv is NOT modified).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

OUT = Path("results/robustness")
OUT.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]


def run_classification(X, y, feature_set_name, study_name, outcome_name,
                       n_splits=10, n_repeats=10):
    """Same logic as clinical_utility.run_classification — duplicated here for
    isolation. Returns dict of metrics, or None if too few positives/negatives.
    """
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X_c, y_c = X[mask], y[mask]

    if y_c.sum() < 5 or (len(y_c) - y_c.sum()) < 5:
        return None

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3)

    aucs_lr, aucs_rf = [], []
    sens_list, spec_list, ppv_list, npv_list, f1_list = [], [], [], [], []

    for train_idx, test_idx in cv.split(X_c, y_c):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_c[train_idx])
        X_te = scaler.transform(X_c[test_idx])
        y_tr, y_te = y_c[train_idx], y_c[test_idx]
        if y_te.sum() == 0 or y_te.sum() == len(y_te):
            continue

        lr.fit(X_tr, y_tr)
        rf.fit(X_tr, y_tr)
        p_lr = lr.predict_proba(X_te)[:, 1]
        p_rf = rf.predict_proba(X_te)[:, 1]
        a_lr = roc_auc_score(y_te, p_lr)
        a_rf = roc_auc_score(y_te, p_rf)
        aucs_lr.append(a_lr)
        aucs_rf.append(a_rf)

        best_p = p_lr if a_lr >= a_rf else p_rf
        fpr, tpr, thresholds = roc_curve(y_te, best_p)
        j_idx = np.argmax(tpr - fpr)
        y_pred = (best_p >= thresholds[j_idx]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
        sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        ppv_list.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        npv_list.append(tn / (tn + fn) if (tn + fn) > 0 else 0)
        f1_list.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0)

    auc_lr = float(np.mean(aucs_lr))
    auc_rf = float(np.mean(aucs_rf))
    best_aucs = [max(a, b) for a, b in zip(aucs_lr, aucs_rf)]
    best_auc = float(np.mean(best_aucs))
    best_auc_ci_lo = float(np.percentile(best_aucs, 2.5))
    best_auc_ci_hi = float(np.percentile(best_aucs, 97.5))
    best_model = "LR" if auc_lr >= auc_rf else "RF"
    sens = float(np.mean(sens_list))
    spec = float(np.mean(spec_list))
    ppv  = float(np.mean(ppv_list))
    npv  = float(np.mean(npv_list))
    f1   = float(np.mean(f1_list))

    return {
        "Study": study_name, "Outcome": outcome_name, "Features": feature_set_name,
        "N": len(y_c), "N_pos": int(y_c.sum()), "Prevalence": y_c.mean(),
        "AUC_LR": auc_lr, "AUC_RF": auc_rf, "Best_AUC": best_auc,
        "AUC_CI_lo": best_auc_ci_lo, "AUC_CI_hi": best_auc_ci_hi,
        "Best_Model": best_model,
        "Sensitivity": sens, "Specificity": spec, "PPV": ppv, "NPV": npv, "F1": f1,
    }


def matched_classification(df, outcome_col, cutoff, study_name, outcome_name,
                           pers_cols, beh_cols):
    """Run Pers-only, Beh-only, Pers+Beh on the SAME intersection sample
    (rows with both personality and sensing complete).
    """
    valid_pers = [c for c in pers_cols if c in df.columns]
    valid_beh  = [c for c in beh_cols  if c in df.columns]

    # Build the intersection mask: pers complete AND beh complete AND outcome present
    pers_data = df[valid_pers].values
    beh_data  = df[valid_beh].values
    y_full    = (df[outcome_col] >= cutoff).astype(float).values
    y_raw     = df[outcome_col].values

    pers_complete = ~np.isnan(pers_data).any(axis=1)
    beh_complete  = ~np.isnan(beh_data).any(axis=1)
    y_present     = ~np.isnan(y_raw)
    intersection  = pers_complete & beh_complete & y_present

    df_int = df[intersection].reset_index(drop=True)
    n_int  = intersection.sum()

    print(f"\n  {study_name} {outcome_name}", flush=True)
    print(f"    Intersection N = {n_int} (was: pers-only N would be "
          f"{pers_complete.sum()}, beh-only N would be {beh_complete.sum()})",
          flush=True)

    y = (df_int[outcome_col] >= cutoff).astype(float).values
    print(f"    Matched prevalence = {y.mean():.3f}", flush=True)

    out_rows = []

    # Pers-only on intersection
    X_pers = df_int[valid_pers].values
    res = run_classification(X_pers, y, "Pers-only", study_name, outcome_name)
    if res:
        out_rows.append(res)
        print(f"    Pers-only:  AUC={res['Best_AUC']:.3f} "
              f"Sens={res['Sensitivity']:.2f} Spec={res['Specificity']:.2f}",
              flush=True)

    # Pers+Beh on intersection
    if valid_beh:
        X_both = df_int[valid_pers + valid_beh].values
        res = run_classification(X_both, y, "Pers+Beh", study_name, outcome_name)
        if res:
            out_rows.append(res)
            print(f"    Pers+Beh:   AUC={res['Best_AUC']:.3f} "
                  f"Sens={res['Sensitivity']:.2f} Spec={res['Specificity']:.2f}",
                  flush=True)

    # Beh-only on intersection
    if valid_beh:
        X_beh = df_int[valid_beh].values
        res = run_classification(X_beh, y, "Beh-only", study_name, outcome_name)
        if res:
            out_rows.append(res)
            print(f"    Beh-only:   AUC={res['Best_AUC']:.3f} "
                  f"Sens={res['Sensitivity']:.2f} Spec={res['Specificity']:.2f}",
                  flush=True)

    return out_rows


def compute_nns_metrics(clf_df):
    """Translate classification metrics into TP/100, FP/100, NNS, etc.
    Mirrors robustness_12_nns_comparison.run_nns logic.
    """
    rows = []
    for _, r in clf_df.iterrows():
        prevalence = r["Prevalence"]
        sensitivity = r["Sensitivity"]
        specificity = r["Specificity"]
        ppv = r["PPV"]
        npv = r["NPV"]
        n = r["N"]

        n_screen = 100
        n_true_pos = n_screen * prevalence
        n_true_neg = n_screen * (1 - prevalence)

        tp_per_100 = sensitivity * n_true_pos
        fp_per_100 = (1 - specificity) * n_true_neg
        fn_per_100 = (1 - sensitivity) * n_true_pos
        tn_per_100 = specificity * n_true_neg

        nns = 1 / ppv if ppv > 0 else np.inf
        pt = prevalence
        nb = (tp_per_100 / n_screen) - (fp_per_100 / n_screen) * (pt / (1 - pt)) \
             if pt < 1 else 0

        rows.append({
            "Study": r["Study"], "Outcome": r["Outcome"], "Features": r["Features"],
            "N": int(n), "Prevalence": round(prevalence, 3),
            "Sensitivity": round(sensitivity, 3),
            "Specificity": round(specificity, 3),
            "PPV": round(ppv, 3), "NPV": round(npv, 3),
            "AUC": round(r["Best_AUC"], 3),
            "NNS": round(nns, 1),
            "TP_per_100": round(tp_per_100, 1),
            "FP_per_100": round(fp_per_100, 1),
            "FN_per_100": round(fn_per_100, 1),
            "TN_per_100": round(tn_per_100, 1),
            "Net_Benefit": round(nb, 4),
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 70, flush=True)
    print("ANALYSIS 43b: Matched-N Clinical Classification (Plan B)", flush=True)
    print("=" * 70, flush=True)

    print("\nLoading datasets...", flush=True)
    s2 = pd.read_parquet("data/processed/nethealth/nethealth_analysis_dataset.parquet")
    s3 = pd.read_parquet("data/processed/globem/globem_analysis_dataset.parquet")

    S2_BEH_PCA = [c for c in s2.columns if c.startswith("nh_") and c.endswith("_pc1")]
    S3_BEH_PCA = [c for c in s3.columns if c.endswith("_pc1") and not c.startswith("nh_")]

    tasks = [
        ("S2", s2, "cesd_total",       "CES-D≥16",   16, TRAITS, S2_BEH_PCA),
        ("S2", s2, "stai_trait_total", "STAI≥45",    45, TRAITS, S2_BEH_PCA),
        ("S3", s3, "bdi2_total",       "BDI-II≥14",  14, TRAITS, S3_BEH_PCA),
        ("S3", s3, "bdi2_total",       "BDI-II≥20",  20, TRAITS, S3_BEH_PCA),
        ("S3", s3, "pss_10",           "PSS≥20",     20, TRAITS, S3_BEH_PCA),
    ]

    all_rows = []
    for study, df, col, outcome_name, cutoff, pers_cols, beh_cols in tasks:
        if col not in df.columns:
            print(f"  Skipping {study} {outcome_name}: column {col} missing")
            continue
        rows = matched_classification(df, col, cutoff, study, outcome_name,
                                      pers_cols, beh_cols)
        all_rows.extend(rows)

    clf_df = pd.DataFrame(all_rows)
    clf_df.to_csv(OUT / "clinical_classification_matched.csv", index=False)
    print(f"\n  Saved: {OUT / 'clinical_classification_matched.csv'}", flush=True)

    nns_df = compute_nns_metrics(clf_df)
    nns_df.to_csv(OUT / "nns_comparison_matched.csv", index=False)
    print(f"  Saved: {OUT / 'nns_comparison_matched.csv'}", flush=True)

    # Head-to-head report
    print("\n" + "=" * 70, flush=True)
    print("Matched head-to-head: Personality vs Sensing TP/100", flush=True)
    print("=" * 70, flush=True)
    pers_wins = 0
    total = 0
    for (study, outcome), grp in nns_df.groupby(["Study", "Outcome"]):
        pers = grp[grp["Features"] == "Pers-only"]
        beh  = grp[grp["Features"] == "Beh-only"]
        if len(pers) == 0 or len(beh) == 0:
            continue
        tp_p = pers["TP_per_100"].values[0]
        tp_b = beh["TP_per_100"].values[0]
        delta = tp_p - tp_b
        n = pers["N"].values[0]
        prev = pers["Prevalence"].values[0]
        winner = "Pers" if tp_p > tp_b else ("Sens" if tp_b > tp_p else "tie")
        total += 1
        if tp_p > tp_b:
            pers_wins += 1
        print(f"  {study} {outcome:14s}  N={int(n)} prev={prev:.2f}  "
              f"Pers={tp_p:5.1f}  Sens={tp_b:5.1f}  Δ={delta:+5.1f}  → {winner}",
              flush=True)
    print(f"\n  Personality wins {pers_wins}/{total} cutoffs (matched-N)",
          flush=True)


if __name__ == "__main__":
    main()
