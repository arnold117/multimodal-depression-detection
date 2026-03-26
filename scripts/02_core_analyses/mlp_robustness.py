#!/usr/bin/env python3
"""
MLP Robustness Check: 3-layer MLP + Optuna Bayesian optimization
Compare with traditional ML (EN, Ridge, RF, SVR, LR) across all 3 studies.

Strategy: Optuna tunes hyperparams ONCE via inner 5-fold CV on full data,
then the best config is evaluated via outer 10×10-fold CV (same as main pipeline).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, r2_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings("ignore")

OUT = Path("results/core")
OUT.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]

# ═══════════════════════════════════════════════════════════════════════
# Load datasets
# ═══════════════════════════════════════════════════════════════════════
print("Loading datasets...")
s1 = pd.read_parquet("data/processed/analysis_dataset.parquet")
s2 = pd.read_parquet("data/processed/nethealth/nethealth_analysis_dataset.parquet")
s3 = pd.read_parquet("data/processed/globem/globem_analysis_dataset.parquet")
print(f"  S1: {s1.shape}, S2: {s2.shape}, S3: {s3.shape}")


def tune_and_evaluate_reg(X, y, study_name, n_optuna=30):
    """Tune MLP once, then evaluate with outer CV. Returns R²."""
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X_c, y_c = X[mask], y[mask]
    if len(X_c) < 20:
        return np.nan, len(X_c)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_c)

    # Phase 1: Optuna tuning via 5-fold CV on full data
    def objective(trial):
        h1 = trial.suggest_int("h1", 8, 64)
        h2 = trial.suggest_int("h2", 4, 32)
        alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
        lr_init = trial.suggest_float("lr_init", 1e-4, 1e-2, log=True)

        mlp = MLPRegressor(
            hidden_layer_sizes=(h1, h2), alpha=alpha,
            learning_rate_init=lr_init, max_iter=500,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, random_state=42,
        )
        cv_inner = 5 if len(X_s) >= 50 else 3
        scores = cross_val_score(mlp, X_s, y_c, cv=cv_inner, scoring="r2")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_optuna, show_progress_bar=False)
    bp = study.best_params

    # Phase 2: Outer CV with fixed best params
    if len(X_c) < 50:
        # LOO for small samples
        from sklearn.model_selection import LeaveOneOut
        loo = LeaveOneOut()
        preds = np.zeros(len(X_c))
        for tr, te in loo.split(X_s):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_c[tr])
            X_te = sc.transform(X_c[te])
            mlp = MLPRegressor(
                hidden_layer_sizes=(bp["h1"], bp["h2"]), alpha=bp["alpha"],
                learning_rate_init=bp["lr_init"], max_iter=500,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=20, random_state=42,
            )
            mlp.fit(X_tr, y_c[tr])
            preds[te] = mlp.predict(X_te)
        r2 = r2_score(y_c, preds)
    else:
        cv_outer = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        r2_list = []
        for tr, te in cv_outer.split(X_c):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_c[tr])
            X_te = sc.transform(X_c[te])
            mlp = MLPRegressor(
                hidden_layer_sizes=(bp["h1"], bp["h2"]), alpha=bp["alpha"],
                learning_rate_init=bp["lr_init"], max_iter=500,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=20, random_state=42,
            )
            mlp.fit(X_tr, y_c[tr])
            r2_list.append(r2_score(y_c[te], mlp.predict(X_te)))
        r2 = float(np.mean(r2_list))

    return r2, len(X_c)


def tune_and_evaluate_clf(X, y, n_optuna=30):
    """Tune MLP classifier once, then evaluate with outer CV. Returns AUC."""
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X_c, y_c = X[mask], y[mask]
    if y_c.sum() < 5 or (len(y_c) - y_c.sum()) < 5:
        return np.nan, len(X_c)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_c)

    # Phase 1: Optuna tuning
    def objective(trial):
        h1 = trial.suggest_int("h1", 8, 64)
        h2 = trial.suggest_int("h2", 4, 32)
        alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
        lr_init = trial.suggest_float("lr_init", 1e-4, 1e-2, log=True)

        mlp = MLPClassifier(
            hidden_layer_sizes=(h1, h2), alpha=alpha,
            learning_rate_init=lr_init, max_iter=500,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, random_state=42,
        )
        scores = cross_val_score(mlp, X_s, y_c, cv=5, scoring="roc_auc")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_optuna, show_progress_bar=False)
    bp = study.best_params

    # Phase 2: Outer CV
    cv_outer = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    auc_list = []
    for tr, te in cv_outer.split(X_c, y_c):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_c[tr])
        X_te = sc.transform(X_c[te])
        y_tr, y_te = y_c[tr], y_c[te]
        if y_te.sum() == 0 or y_te.sum() == len(y_te):
            continue
        mlp = MLPClassifier(
            hidden_layer_sizes=(bp["h1"], bp["h2"]), alpha=bp["alpha"],
            learning_rate_init=bp["lr_init"], max_iter=500,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, random_state=42,
        )
        mlp.fit(X_tr, y_tr)
        auc_list.append(roc_auc_score(y_te, mlp.predict_proba(X_te)[:, 1]))

    return float(np.mean(auc_list)) if auc_list else np.nan, len(X_c)


# ═══════════════════════════════════════════════════════════════════════
# PART A: Regression
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART A: MLP Regression (Personality → continuous MH)")
print("=" * 70)

reg_tasks = [
    ("S1", s1, "phq9_total", "PHQ-9"),
    ("S1", s1, "pss_total", "PSS"),
    ("S1", s1, "loneliness_total", "Loneliness"),
    ("S1", s1, "flourishing_total", "Flourishing"),
    ("S2", s2, "cesd_total", "CES-D"),
    ("S2", s2, "stai_trait_total", "STAI"),
    ("S2", s2, "bai_total", "BAI"),
    ("S3", s3, "bdi2_total", "BDI-II"),
    ("S3", s3, "stai_state", "STAI-State"),
    ("S3", s3, "pss_10", "PSS-10"),
    ("S3", s3, "cesd_total", "CESD-10"),
    ("S3", s3, "ucla_loneliness", "UCLA"),
]

reg_results = []
for study, df, col, label in reg_tasks:
    if col not in df.columns:
        print(f"  Skipping {study} {label}: column not found")
        continue
    valid_pers = [c for c in TRAITS if c in df.columns]
    X = df[valid_pers].values
    y = df[col].values

    n_trials = 15 if study == "S1" else 30
    print(f"  {study} {label}...", end=" ", flush=True)
    r2, n = tune_and_evaluate_reg(X, y, study, n_optuna=n_trials)
    print(f"R²={r2:.3f} (N={n})")
    reg_results.append({"Study": study, "Outcome": label, "N": n, "R2_MLP": r2})

reg_df = pd.DataFrame(reg_results)
reg_df.to_csv(OUT / "mlp_regression.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════
# PART B: Classification
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART B: MLP Classification (Personality → binary MH)")
print("=" * 70)

clf_tasks = [
    ("S2", s2, "cesd_total", "CES-D≥16", 16),
    ("S2", s2, "stai_trait_total", "STAI≥45", 45),
    ("S3", s3, "bdi2_total", "BDI-II≥14", 14),
    ("S3", s3, "bdi2_total", "BDI-II≥20", 20),
    ("S3", s3, "pss_10", "PSS≥20", 20),
]

clf_results = []
for study, df, col, label, cutoff in clf_tasks:
    if col not in df.columns:
        continue
    valid_pers = [c for c in TRAITS if c in df.columns]
    X = df[valid_pers].values
    y = (df[col] >= cutoff).astype(float).values

    print(f"  {study} {label}...", end=" ", flush=True)
    auc, n = tune_and_evaluate_clf(X, y)
    print(f"AUC={auc:.3f} (N={n})")
    clf_results.append({"Study": study, "Outcome": label, "N": n, "AUC_MLP": auc})

clf_df = pd.DataFrame(clf_results)
clf_df.to_csv(OUT / "mlp_classification.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════
# Summary comparison
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("COMPARISON: MLP vs Traditional Models")
print("=" * 70)

# Classification comparison
existing_clf = pd.read_csv(OUT / "clinical_classification.csv")
existing_pers = existing_clf[existing_clf["Features"] == "Pers-only"]

print("\n  Classification (Pers-only AUC):")
print(f"  {'Study':<6} {'Outcome':<12} {'LR/RF':<8} {'MLP':<8} {'Δ':<8}")
print(f"  {'-'*42}")
for _, row in clf_df.iterrows():
    match = existing_pers[
        (existing_pers["Study"] == row["Study"]) &
        (existing_pers["Outcome"] == row["Outcome"])
    ]
    if len(match) > 0:
        trad = match["Best_AUC"].values[0]
        diff = row["AUC_MLP"] - trad
        print(f"  {row['Study']:<6} {row['Outcome']:<12} {trad:<8.3f} {row['AUC_MLP']:<8.3f} {diff:+.3f}")

print(f"\n  Saved: {OUT / 'mlp_regression.csv'}")
print(f"  Saved: {OUT / 'mlp_classification.csv'}")
