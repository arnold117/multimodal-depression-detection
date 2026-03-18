#!/usr/bin/env python3
"""
Analysis 1 (FAST version): Raw RAPIDS Features vs PCA
Ridge-only, 5×5 CV — should finish in ~10-15 min
Results saved to rapids_comparison_fast.csv (won't overwrite the slow version)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import warnings, sys
warnings.filterwarnings("ignore")

OUT = Path("results/comparison/supplementary")
OUT.mkdir(parents=True, exist_ok=True)
TRAITS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]

print("Loading...", flush=True)
s3 = pd.read_parquet("data/processed/globem/globem_analysis_dataset.parquet")
rapids = pd.read_parquet("data/processed/globem/features/rapids_features.parquet")
merged = s3.merge(rapids, on="pid", how="inner")
S3_BEH_PCA = [c for c in s3.columns if c.endswith("_pc1") and not c.startswith("nh_")]

rapids_cols = [c for c in rapids.columns if c != "pid"]
valid_cols = [c for c in rapids_cols
              if np.isnan(merged[c].values).mean() <= 0.20
              and np.nanvar(merged[c].values) >= 0.01]
print(f"Features after filter: {len(valid_cols)} (from {len(rapids_cols)})", flush=True)

OUTCOMES = {"bdi2_total": "BDI-II", "stai_state": "STAI", "pss_10": "PSS-10",
            "cesd_total": "CESD", "ucla_loneliness": "UCLA"}

def kfold(X, y, preprocess_fn=None, n_splits=5, n_repeats=5, alpha=1.0):
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X_c, y_c = X[mask], y[mask]
    if len(y_c) < 30:
        return np.nan, np.nan, np.nan, len(y_c)
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    r2s = []
    for tr, te in cv.split(X_c):
        Xtr, Xte = X_c[tr].copy(), X_c[te].copy()
        # impute
        med = np.nanmedian(Xtr, axis=0)
        for j in range(Xtr.shape[1]):
            Xtr[np.isnan(Xtr[:, j]), j] = med[j]
            Xte[np.isnan(Xte[:, j]), j] = med[j]
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
        if preprocess_fn:
            Xtr, Xte = preprocess_fn(Xtr, Xte)
        m = Ridge(alpha=alpha)
        m.fit(Xtr, y_c[tr])
        r2s.append(r2_score(y_c[te], m.predict(Xte)))
    r2s = np.array(r2s)
    return float(np.mean(r2s)), float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5)), len(y_c)

def pca_fn(Xtr, Xte):
    pca = PCA(n_components=0.90, random_state=42)
    return pca.fit_transform(Xtr), pca.transform(Xte)

rows = []
valid_pers = [c for c in TRAITS if c in merged.columns]

for col, label in OUTCOMES.items():
    y = merged[col].values
    print(f"\n{label}:", flush=True)

    # (a) 5 PCA composites
    r2, lo, hi, n = kfold(merged[S3_BEH_PCA].values, y)
    print(f"  5 PCA:       R²={r2:.4f} [{lo:.4f}, {hi:.4f}]", flush=True)
    rows.append({"Outcome": label, "Approach": "5 PCA composites", "Type": "Beh", "R2": r2, "CI_lo": lo, "CI_hi": hi, "N": n})

    # (b) PCA 90% on raw
    r2, lo, hi, n = kfold(merged[valid_cols].values.copy(), y, preprocess_fn=pca_fn)
    print(f"  PCA90%:      R²={r2:.4f} [{lo:.4f}, {hi:.4f}]", flush=True)
    rows.append({"Outcome": label, "Approach": "PCA 90% variance", "Type": "Beh", "R2": r2, "CI_lo": lo, "CI_hi": hi, "N": n})

    # (c) Raw + Ridge (alpha=1)
    r2, lo, hi, n = kfold(merged[valid_cols].values.copy(), y, alpha=1.0)
    print(f"  Raw Ridge1:  R²={r2:.4f} [{lo:.4f}, {hi:.4f}]", flush=True)
    rows.append({"Outcome": label, "Approach": "Raw Ridge(1)", "Type": "Beh", "R2": r2, "CI_lo": lo, "CI_hi": hi, "N": n})

    # (d) Raw + Ridge (alpha=100, stronger reg)
    r2, lo, hi, n = kfold(merged[valid_cols].values.copy(), y, alpha=100.0)
    print(f"  Raw Ridge100:R²={r2:.4f} [{lo:.4f}, {hi:.4f}]", flush=True)
    rows.append({"Outcome": label, "Approach": "Raw Ridge(100)", "Type": "Beh", "R2": r2, "CI_lo": lo, "CI_hi": hi, "N": n})

    # (e) Personality only
    r2, lo, hi, n = kfold(merged[valid_pers].values, y)
    print(f"  Pers only:   R²={r2:.4f} [{lo:.4f}, {hi:.4f}]", flush=True)
    rows.append({"Outcome": label, "Approach": "Personality only", "Type": "Pers", "R2": r2, "CI_lo": lo, "CI_hi": hi, "N": n})

    # (f) Pers + Raw Ridge
    Xpr = np.hstack([merged[valid_pers].values, merged[valid_cols].values.copy()])
    r2, lo, hi, n = kfold(Xpr, y, alpha=10.0)
    print(f"  Pers+Raw:    R²={r2:.4f} [{lo:.4f}, {hi:.4f}]", flush=True)
    rows.append({"Outcome": label, "Approach": "Pers+Raw Ridge(10)", "Type": "P+B", "R2": r2, "CI_lo": lo, "CI_hi": hi, "N": n})

df = pd.DataFrame(rows)
df.to_csv(OUT / "rapids_comparison_fast.csv", index=False)
print(f"\nSaved: {OUT / 'rapids_comparison_fast.csv'}", flush=True)
print(df.to_string(), flush=True)
