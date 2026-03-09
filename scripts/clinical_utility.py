#!/usr/bin/env python3
"""
Phase 15: Clinical Utility & Methodology
==========================================
Three sections:
  1. Clinical binary classification (AUC, sensitivity, specificity)
  2. Incremental validity (nested F-test: Pers+Beh vs Pers-only)
  3. SHAP vs traditional methods (ranking agreement)
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/comparison")
FIG = Path("results/comparison")
OUT.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]


# ═══════════════════════════════════════════════════════════════════════
# Load datasets
# ═══════════════════════════════════════════════════════════════════════
print("Loading datasets...")
s1 = pd.read_parquet("data/processed/analysis_dataset.parquet")
s2 = pd.read_parquet("data/processed/nethealth/nethealth_analysis_dataset.parquet")
s3 = pd.read_parquet("data/processed/globem/globem_analysis_dataset.parquet")

S2_BEH_PCA = [c for c in s2.columns if c.startswith("nh_") and c.endswith("_pc1")]
S3_BEH_PCA = [c for c in s3.columns if c.endswith("_pc1") and not c.startswith("nh_")]

print(f"  S1: {s1.shape}, S2: {s2.shape}, S3: {s3.shape}")
print(f"  S2 behavioral PCA: {S2_BEH_PCA}")
print(f"  S3 behavioral PCA: {S3_BEH_PCA}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: Clinical Binary Classification
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 1: Clinical Binary Classification")
print("=" * 70)


def run_classification(X, y, feature_set_name, study_name, outcome_name, n_splits=10, n_repeats=10):
    """Run repeated stratified k-fold classification, return metrics."""
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X_c, y_c = X[mask], y[mask]

    if y_c.sum() < 5 or (len(y_c) - y_c.sum()) < 5:
        return None  # too few positives/negatives

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_c)

    from sklearn.metrics import roc_curve
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3)

    # Manually loop — cross_val_predict doesn't support RepeatedStratifiedKFold (overlapping test sets)
    aucs_lr, aucs_rf = [], []
    sens_list, spec_list, ppv_list, npv_list, f1_list = [], [], [], [], []

    for train_idx, test_idx in cv.split(X_s, y_c):
        X_tr, X_te = X_s[train_idx], X_s[test_idx]
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
    best_auc = max(auc_lr, auc_rf)
    best_model = "LR" if auc_lr >= auc_rf else "RF"
    sens = float(np.mean(sens_list))
    spec = float(np.mean(spec_list))
    ppv  = float(np.mean(ppv_list))
    npv  = float(np.mean(npv_list))
    f1   = float(np.mean(f1_list))

    return {
        "Study": study_name, "Outcome": outcome_name, "Features": feature_set_name,
        "N": len(y_c), "N_pos": int(y_c.sum()), "Prevalence": y_c.mean(),
        "AUC_LR": auc_lr, "AUC_RF": auc_rf, "Best_AUC": best_auc, "Best_Model": best_model,
        "Sensitivity": sens, "Specificity": spec, "PPV": ppv, "NPV": npv, "F1": f1,
    }


# Define classification tasks
tasks = [
    # Study 2
    ("S2", s2, "cesd_total", "CES-D≥16", 16, TRAITS, S2_BEH_PCA),
    ("S2", s2, "stai_trait_total", "STAI≥45", 45, TRAITS, S2_BEH_PCA),
    # Study 3
    ("S3", s3, "bdi2_total", "BDI-II≥14", 14, TRAITS, S3_BEH_PCA),
    ("S3", s3, "bdi2_total", "BDI-II≥20", 20, TRAITS, S3_BEH_PCA),
    ("S3", s3, "pss_10", "PSS≥20", 20, TRAITS, S3_BEH_PCA),
]

clf_results = []
for study, df, col, outcome_name, cutoff, pers_cols, beh_cols in tasks:
    if col not in df.columns:
        print(f"  Skipping {study} {outcome_name}: column {col} not found")
        continue

    y = (df[col] >= cutoff).astype(float).values
    valid_pers = [c for c in pers_cols if c in df.columns]
    valid_beh = [c for c in beh_cols if c in df.columns]

    print(f"\n  {study} {outcome_name} (N={len(df)}, N+={int(np.nansum(y))})")

    # Pers-only
    X_pers = df[valid_pers].values
    res = run_classification(X_pers, y, "Pers-only", study, outcome_name)
    if res:
        clf_results.append(res)
        print(f"    Pers-only:  AUC={res['Best_AUC']:.3f} (Sens={res['Sensitivity']:.2f}, Spec={res['Specificity']:.2f})")

    # Pers+Beh
    if valid_beh:
        X_both = df[valid_pers + valid_beh].values
        res = run_classification(X_both, y, "Pers+Beh", study, outcome_name)
        if res:
            clf_results.append(res)
            print(f"    Pers+Beh:   AUC={res['Best_AUC']:.3f} (Sens={res['Sensitivity']:.2f}, Spec={res['Specificity']:.2f})")

    # Beh-only
    if valid_beh:
        X_beh = df[valid_beh].values
        res = run_classification(X_beh, y, "Beh-only", study, outcome_name)
        if res:
            clf_results.append(res)
            print(f"    Beh-only:   AUC={res['Best_AUC']:.3f} (Sens={res['Sensitivity']:.2f}, Spec={res['Specificity']:.2f})")

clf_df = pd.DataFrame(clf_results)
clf_df.to_csv(OUT / "clinical_classification.csv", index=False)

# ── Figure 18: AUC comparison ──────────────────────────────────────
if len(clf_df) > 0:
    fig, ax = plt.subplots(figsize=(14, 6))

    outcomes = clf_df["Outcome"].unique()
    feature_sets = ["Pers-only", "Pers+Beh", "Beh-only"]
    colors_fs = {"Pers-only": "#e74c3c", "Pers+Beh": "#3498db", "Beh-only": "#95a5a6"}
    x = np.arange(len(outcomes))
    width = 0.25

    for i, fs in enumerate(feature_sets):
        aucs = []
        for outcome in outcomes:
            row = clf_df[(clf_df["Outcome"] == outcome) & (clf_df["Features"] == fs)]
            aucs.append(row["Best_AUC"].values[0] if len(row) > 0 else 0)
        bars = ax.bar(x + i * width, aucs, width, label=fs, color=colors_fs[fs], edgecolor="white")
        for j, v in enumerate(aucs):
            if v > 0:
                ax.text(x[j] + i * width, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{o}" for o in outcomes], fontsize=10)
    ax.set_ylabel("AUC (Cross-Validated)", fontsize=12)
    ax.set_title("Clinical Classification: Personality vs Behavioral Features", fontsize=13, fontweight="bold")
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="Chance")
    ax.legend(fontsize=10)
    ax.set_ylim(0.4, 0.85)

    plt.tight_layout()
    fig.savefig(FIG / "figure18_clinical_classification.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {FIG / 'figure18_clinical_classification.png'}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: Incremental Validity (Nested F-test)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: Incremental Validity (Nested F-test)")
print("=" * 70)

inc_results = []

inc_tasks = [
    # Study 2
    ("S2", s2, "cesd_total", TRAITS, S2_BEH_PCA),
    ("S2", s2, "stai_trait_total", TRAITS, S2_BEH_PCA),
    ("S2", s2, "bai_total", TRAITS, S2_BEH_PCA),
    # Study 3
    ("S3", s3, "bdi2_total", TRAITS, S3_BEH_PCA),
    ("S3", s3, "stai_state", TRAITS, S3_BEH_PCA),
    ("S3", s3, "pss_10", TRAITS, S3_BEH_PCA),
    ("S3", s3, "cesd_total", TRAITS, S3_BEH_PCA),
    ("S3", s3, "ucla_loneliness", TRAITS, S3_BEH_PCA),
]

for study, df, outcome_col, pers_cols, beh_cols in inc_tasks:
    if outcome_col not in df.columns:
        continue

    valid_pers = [c for c in pers_cols if c in df.columns]
    valid_beh = [c for c in beh_cols if c in df.columns]
    if not valid_beh:
        continue

    # Complete cases
    all_cols = valid_pers + valid_beh + [outcome_col]
    sub = df[all_cols].dropna()
    if len(sub) < 20:
        continue

    y = sub[outcome_col].values
    X_pers = sm.add_constant(sub[valid_pers].values)
    X_full = sm.add_constant(sub[valid_pers + valid_beh].values)

    # Fit both models
    m1 = sm.OLS(y, X_pers).fit()
    m2 = sm.OLS(y, X_full).fit()

    r2_1 = m1.rsquared
    r2_2 = m2.rsquared
    delta_r2 = r2_2 - r2_1

    # Partial F-test
    n = len(y)
    p1 = X_pers.shape[1]  # includes constant
    p2 = X_full.shape[1]
    df_num = p2 - p1  # number of added predictors
    df_den = n - p2

    if df_den > 0 and df_num > 0:
        f_stat = ((r2_2 - r2_1) / df_num) / ((1 - r2_2) / df_den)
        p_val = 1 - stats.f.cdf(f_stat, df_num, df_den)
    else:
        f_stat, p_val = np.nan, np.nan

    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"  {study} {outcome_col}: R²_pers={r2_1:.3f}, R²_full={r2_2:.3f}, ΔR²={delta_r2:.4f}, F={f_stat:.2f}, p={p_val:.4f} {sig}")

    inc_results.append({
        "Study": study, "Outcome": outcome_col, "N": n,
        "R2_pers": r2_1, "R2_full": r2_2, "Delta_R2": delta_r2,
        "F_stat": f_stat, "p_value": p_val, "df_num": df_num, "df_den": df_den,
        "n_beh_features": len(valid_beh),
    })

inc_df = pd.DataFrame(inc_results)
inc_df.to_csv(OUT / "incremental_validity.csv", index=False)
print(f"\n  Significant incremental validity: {(inc_df['p_value'] < 0.05).sum()}/{len(inc_df)}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: SHAP vs Traditional Methods
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: SHAP vs Traditional Methods Comparison")
print("=" * 70)

shap_tasks = [
    # Study 2
    ("S2", s2, "stai_trait_total", "results/nethealth/tables/shap_personality_stai.csv"),
    ("S2", s2, "bai_total", "results/nethealth/tables/shap_personality_bai.csv"),
    # Study 3
    ("S3", s3, "bdi2_total", "results/globem/tables/shap_personality_bdiii.csv"),
    ("S3", s3, "stai_state", "results/globem/tables/shap_personality_stai.csv"),
    ("S3", s3, "pss_10", "results/globem/tables/shap_personality_pss10.csv"),
    ("S3", s3, "cesd_total", "results/globem/tables/shap_personality_cesd.csv"),
    ("S3", s3, "ucla_loneliness", "results/globem/tables/shap_personality_ucla.csv"),
]

shap_comparison = []

for study, df, outcome_col, shap_file in shap_tasks:
    if outcome_col not in df.columns or not Path(shap_file).exists():
        print(f"  Skipping {study} {outcome_col}: data or SHAP file missing")
        continue

    valid_pers = [c for c in TRAITS if c in df.columns]
    sub = df[valid_pers + [outcome_col]].dropna()
    if len(sub) < 20:
        continue

    # 1. Zero-order Pearson r (absolute)
    r_vals = {}
    for trait in valid_pers:
        r, _ = stats.pearsonr(sub[trait], sub[outcome_col])
        r_vals[trait] = abs(r)
    r_ranking = sorted(r_vals, key=r_vals.get, reverse=True)

    # 2. Standardized β (from OLS)
    X = sm.add_constant(StandardScaler().fit_transform(sub[valid_pers].values))
    y = sub[outcome_col].values
    model = sm.OLS(y, X).fit()
    beta_vals = {trait: abs(model.params[i + 1]) for i, trait in enumerate(valid_pers)}
    beta_ranking = sorted(beta_vals, key=beta_vals.get, reverse=True)

    # 3. SHAP ranking (mean across models)
    shap_df = pd.read_csv(shap_file)
    # Wide format: Model, trait1, trait2, ...
    shap_cols = [c for c in shap_df.columns if c != "Model" and c.lower() != "unnamed: 0"]
    # Map SHAP column names to our trait names
    shap_means = {}
    for trait in valid_pers:
        # Try exact match or capitalized
        candidates = [c for c in shap_cols if trait.lower() in c.lower()]
        if candidates:
            shap_means[trait] = shap_df[candidates[0]].mean()
        else:
            shap_means[trait] = 0
    shap_ranking = sorted(shap_means, key=shap_means.get, reverse=True)

    # Kendall's τ between rankings
    def rank_to_array(ranking, traits):
        return [ranking.index(t) for t in traits]

    r_ranks = rank_to_array(r_ranking, valid_pers)
    beta_ranks = rank_to_array(beta_ranking, valid_pers)
    shap_ranks = rank_to_array(shap_ranking, valid_pers)

    tau_r_beta, p_rb = stats.kendalltau(r_ranks, beta_ranks)
    tau_r_shap, p_rs = stats.kendalltau(r_ranks, shap_ranks)
    tau_beta_shap, p_bs = stats.kendalltau(beta_ranks, shap_ranks)

    print(f"\n  {study} {outcome_col} (N={len(sub)}):")
    print(f"    r ranking:    {r_ranking}")
    print(f"    β ranking:    {beta_ranking}")
    print(f"    SHAP ranking: {shap_ranking}")
    print(f"    τ(r, β)={tau_r_beta:.3f}, τ(r, SHAP)={tau_r_shap:.3f}, τ(β, SHAP)={tau_beta_shap:.3f}")

    shap_comparison.append({
        "Study": study, "Outcome": outcome_col, "N": len(sub),
        "r_rank_1": r_ranking[0], "beta_rank_1": beta_ranking[0], "shap_rank_1": shap_ranking[0],
        "tau_r_beta": tau_r_beta, "tau_r_shap": tau_r_shap, "tau_beta_shap": tau_beta_shap,
        "top1_agree": r_ranking[0] == beta_ranking[0] == shap_ranking[0],
    })

shap_comp_df = pd.DataFrame(shap_comparison)
shap_comp_df.to_csv(OUT / "shap_vs_traditional.csv", index=False)

# Summary
if len(shap_comp_df) > 0:
    print(f"\n  Top-1 agreement (all 3 methods): {shap_comp_df['top1_agree'].sum()}/{len(shap_comp_df)}")
    print(f"  Mean τ(r, SHAP): {shap_comp_df['tau_r_shap'].mean():.3f}")
    print(f"  Mean τ(β, SHAP): {shap_comp_df['tau_beta_shap'].mean():.3f}")

# ── Figure 19: SHAP vs Traditional heatmap ─────────────────────────
if len(shap_comp_df) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    tau_types = [
        ("tau_r_beta", "τ(r, β)", "Zero-order r vs Std. β"),
        ("tau_r_shap", "τ(r, SHAP)", "Zero-order r vs SHAP"),
        ("tau_beta_shap", "τ(β, SHAP)", "Std. β vs SHAP"),
    ]

    for idx, (col, label, title) in enumerate(tau_types):
        ax = axes[idx]
        vals = shap_comp_df[col].values
        labels = [f"{r['Study']}\n{r['Outcome']}" for _, r in shap_comp_df.iterrows()]

        colors_bar = ["#2ecc71" if v > 0.6 else "#f39c12" if v > 0.2 else "#e74c3c" for v in vals]
        bars = ax.barh(range(len(vals)), vals, color=colors_bar, edgecolor="white")
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(label, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(-1, 1)
        ax.axvline(0, color="grey", linestyle="--", alpha=0.5)

    plt.suptitle("Feature Ranking Agreement: SHAP vs Traditional Methods", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG / "figure19_shap_vs_traditional.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {FIG / 'figure19_shap_vs_traditional.png'}")


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Clinical classification: {OUT / 'clinical_classification.csv'}")
print(f"  Incremental validity:    {OUT / 'incremental_validity.csv'}")
print(f"  SHAP vs traditional:     {OUT / 'shap_vs_traditional.csv'}")
print(f"  Figure 18:               {FIG / 'figure18_clinical_classification.png'}")
print(f"  Figure 19:               {FIG / 'figure19_shap_vs_traditional.png'}")
