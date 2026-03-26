#!/usr/bin/env python3
"""Apply BH-FDR correction across all Phase 16 supplementary analyses with p-values."""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.stats.multitest import multipletests

PROJECT = Path(__file__).parent.parent.parent
CORE = PROJECT / "results" / "core"
SUP = PROJECT / "results" / "robustness"
OUT = SUP / "phase16_fdr_summary.csv"

results = []

# 1. DeLong tests (5 AUC comparisons: Pers-only vs Pers+Beh)
df = pd.read_csv(CORE / "delong_tests.csv")
for _, row in df.iterrows():
    results.append({
        "Analysis": "DeLong AUC comparison",
        "Description": f"{row['Study']} {row['Outcome']}",
        "Effect": row.get("Delta_AUC", np.nan),
        "Effect_type": "Delta_AUC",
        "p_value": row["p_value"],
        "Source": "delong_tests.csv",
    })

# 2. Demographic controls (p for personality and behavior increments)
df = pd.read_csv(SUP / "full_demographics.csv")
for _, row in df.iterrows():
    results.append({
        "Analysis": "Demographic control — personality",
        "Description": f"{row['Outcome']} personality after demographics",
        "Effect": row.get("DR2_personality", np.nan),
        "Effect_type": "Delta_R2",
        "p_value": row["p_personality"],
        "Source": "full_demographics.csv",
    })
    results.append({
        "Analysis": "Demographic control — behavior",
        "Description": f"{row['Outcome']} behavior after demographics+personality",
        "Effect": row.get("DR2_behavior", np.nan),
        "Effect_type": "Delta_R2",
        "p_value": row["p_behavior"],
        "Source": "full_demographics.csv",
    })

# 3. Missing-as-signal (completeness-outcome correlations)
df = pd.read_csv(SUP / "missing_as_signal.csv")
for _, row in df.iterrows():
    results.append({
        "Analysis": "Missingness as signal",
        "Description": f"{row['Outcome']} completeness correlation",
        "Effect": row.get("r_completeness_outcome", np.nan),
        "Effect_type": "r",
        "p_value": row["p_completeness"],
        "Source": "missing_as_signal.csv",
    })

# 4. Sensing reliability (split-half p per feature)
df = pd.read_csv(SUP / "sensing_reliability.csv")
for _, row in df.iterrows():
    results.append({
        "Analysis": "Sensing reliability",
        "Description": f"{row['Feature']} split-half",
        "Effect": row.get("Split_half_r", np.nan),
        "Effect_type": "r",
        "p_value": row["Split_half_p"],
        "Source": "sensing_reliability.csv",
    })

# Build DataFrame and apply FDR
df_all = pd.DataFrame(results)
mask = df_all["p_value"].notna()
if mask.sum() > 0:
    reject, p_fdr, _, _ = multipletests(
        df_all.loc[mask, "p_value"].values, method="fdr_bh"
    )
    df_all.loc[mask, "p_fdr"] = p_fdr
    df_all.loc[mask, "sig_fdr"] = reject
else:
    df_all["p_fdr"] = np.nan
    df_all["sig_fdr"] = False

df_all.to_csv(OUT, index=False)
print(f"Saved {len(df_all)} tests to {OUT}")
print(f"FDR-significant: {df_all['sig_fdr'].sum()}/{mask.sum()}")
print("\nNote: 37/41 Phase 16 analyses report only R² without formal p-values.")
print("For those analyses, R² ≤ 0 is definitively null (no correction needed).")
