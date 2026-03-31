#!/usr/bin/env python3
"""
Phase 16h — NNS Practical Significance
========================================
Analysis 43: Translate AUC/sensitivity/specificity into clinically
interpretable numbers — "per 100 people screened, how many more true
cases does personality catch vs sensing?"

Reads existing results/core/clinical_classification.csv.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUT = Path("results/robustness")
OUT.mkdir(parents=True, exist_ok=True)


def run_nns():
    print("=" * 70, flush=True)
    print("ANALYSIS 43: NNS Practical Significance", flush=True)
    print("=" * 70, flush=True)

    # Load existing classification results
    clf = pd.read_csv("results/core/clinical_classification.csv")
    print(f"  Loaded {len(clf)} rows from clinical_classification.csv", flush=True)

    rows = []
    for _, r in clf.iterrows():
        prevalence = r["Prevalence"]
        sensitivity = r["Sensitivity"]
        specificity = r["Specificity"]
        ppv = r["PPV"]
        npv = r["NPV"]
        n = r["N"]
        n_pos = r["N_pos"]

        # Per 100 screened
        n_screen = 100
        n_true_pos = n_screen * prevalence
        n_true_neg = n_screen * (1 - prevalence)

        tp_per_100 = sensitivity * n_true_pos
        fp_per_100 = (1 - specificity) * n_true_neg
        fn_per_100 = (1 - sensitivity) * n_true_pos
        tn_per_100 = specificity * n_true_neg

        # NNS = 1 / PPV (number needed to screen to find one true case)
        nns = 1 / ppv if ppv > 0 else np.inf

        # Net benefit at threshold probability = prevalence
        pt = prevalence
        nb = (tp_per_100 / n_screen) - (fp_per_100 / n_screen) * (pt / (1 - pt)) if pt < 1 else 0

        rows.append({
            "Study": r["Study"], "Outcome": r["Outcome"], "Features": r["Features"],
            "N": n, "Prevalence": round(prevalence, 3),
            "Sensitivity": round(sensitivity, 3), "Specificity": round(specificity, 3),
            "PPV": round(ppv, 3), "NPV": round(npv, 3),
            "AUC": round(r["Best_AUC"], 3),
            "NNS": round(nns, 1),
            "TP_per_100": round(tp_per_100, 1),
            "FP_per_100": round(fp_per_100, 1),
            "FN_per_100": round(fn_per_100, 1),
            "TN_per_100": round(tn_per_100, 1),
            "Net_Benefit": round(nb, 4),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "nns_comparison.csv", index=False)
    print(f"  Saved: {OUT / 'nns_comparison.csv'}", flush=True)

    # Compute head-to-head deltas
    print("\n  Head-to-head (per 100 screened):", flush=True)
    for (study, outcome), grp in df_out.groupby(["Study", "Outcome"]):
        pers = grp[grp["Features"] == "Pers-only"]
        beh = grp[grp["Features"] == "Beh-only"]
        if len(pers) == 0 or len(beh) == 0:
            continue
        tp_pers = pers["TP_per_100"].values[0]
        tp_beh = beh["TP_per_100"].values[0]
        delta = tp_pers - tp_beh
        print(f"    {study} {outcome}: Personality catches {delta:+.1f} more true cases per 100",
              flush=True)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: TP per 100 screened
    ax = axes[0]
    feature_order = ["Pers-only", "Pers+Beh", "Beh-only"]
    colors = {"Pers-only": "#e74c3c", "Pers+Beh": "#3498db", "Beh-only": "#95a5a6"}
    outcomes = df_out["Outcome"].unique()
    x = np.arange(len(outcomes))
    width = 0.25

    for i, fs in enumerate(feature_order):
        vals = []
        for outcome in outcomes:
            row = df_out[(df_out["Outcome"] == outcome) & (df_out["Features"] == fs)]
            vals.append(row["TP_per_100"].values[0] if len(row) > 0 else 0)
        ax.bar(x + i * width, vals, width, label=fs,
               color=colors.get(fs, "#333"), edgecolor="white")

    ax.set_xticks(x + width)
    ax.set_xticklabels(outcomes, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("True Positives per 100 Screened")
    ax.set_title("(A) Screening Yield", fontweight="bold")
    ax.legend(fontsize=9)

    # Panel B: NNS comparison
    ax = axes[1]
    for i, fs in enumerate(feature_order):
        vals = []
        for outcome in outcomes:
            row = df_out[(df_out["Outcome"] == outcome) & (df_out["Features"] == fs)]
            nns_val = row["NNS"].values[0] if len(row) > 0 else 0
            vals.append(min(nns_val, 20))
        ax.bar(x + i * width, vals, width, label=fs,
               color=colors.get(fs, "#333"), edgecolor="white")

    ax.set_xticks(x + width)
    ax.set_xticklabels(outcomes, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Number Needed to Screen (NNS)")
    ax.set_title("(B) Screening Efficiency (lower = better)", fontweight="bold")
    ax.legend(fontsize=9)

    plt.suptitle("Practical Clinical Significance: Personality vs Sensing",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT / "figure_nns.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT / 'figure_nns.png'}", flush=True)

    return df_out


if __name__ == "__main__":
    result = run_nns()
    print("\n✓ Analysis 43 complete.", flush=True)
    print(result.to_string(index=False), flush=True)
