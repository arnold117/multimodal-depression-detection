#!/usr/bin/env python3
"""
Phase 16h — Sensing Temporal Reliability Decay
================================================
Analysis 44: Compute ICC(3,k) for sensing features across non-overlapping
time windows of increasing size (7, 14, 30, 45, 60 days). Shows that
sensing features have poor temporal stability, explaining their failure
as mental health predictors.

Reference: BFI-44 test-retest r ≈ 0.85 (Rammstedt & John, 2007)
           BFI-10 test-retest r ≈ 0.75 (Rammstedt & John, 2007)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pingouin as pg
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/robustness")
OUT.mkdir(parents=True, exist_ok=True)

COHORTS = ["INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4"]
RAW_DIR = Path("data/raw/globem")

FEATURE_MAP = [
    ("steps.csv", "avgsumsteps:14dhist", "Steps"),
    ("sleep.csv", "avgdurationasleepmain:14dhist", "Sleep Duration"),
    ("sleep.csv", "avgefficiencymain:14dhist", "Sleep Efficiency"),
    ("screen.csv", "rapids_sumdurationunlock:14dhist", "Screen Time"),
    ("call.csv", "incoming_count:14dhist", "Calls (Incoming)"),
    ("location.csv", "barnett_hometime:14dhist", "Home Time"),
]

WINDOW_SIZES = [7, 14, 30, 45, 60]

BFI_44_RETEST = 0.85
BFI_10_RETEST = 0.75


def find_col(columns, substring):
    matches = [c for c in columns if substring in c]
    return min(matches, key=len) if matches else None


def load_daily_feature(modality_file, col_substr, short_name):
    """Load one daily feature across all cohorts."""
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
        return pd.concat(all_daily, ignore_index=True)
    return None


def compute_icc_by_windows(daily_df, window_days, min_persons=30):
    """
    Split each person's daily data into non-overlapping windows of
    `window_days` length. Compute within-window mean, then ICC(3,k)
    across windows (treating windows as 'raters' and persons as 'targets').
    """
    daily_df = daily_df.sort_values(["pid", "date"]).copy()
    daily_df["day_num"] = daily_df.groupby("pid").cumcount()
    daily_df["window"] = daily_df["day_num"] // window_days

    agg = daily_df.groupby(["pid", "window"])["value"].mean().reset_index()

    counts = agg.groupby("pid")["window"].count()
    valid_pids = counts[counts >= 2].index
    agg = agg[agg["pid"].isin(valid_pids)]

    n_persons = agg["pid"].nunique()
    n_windows_median = agg.groupby("pid")["window"].count().median()

    if n_persons < min_persons:
        return {
            "ICC": np.nan, "ICC_CI_lo": np.nan, "ICC_CI_hi": np.nan,
            "N_persons": n_persons, "N_windows_median": n_windows_median,
        }

    max_windows = int(agg.groupby("pid")["window"].count().min())
    max_windows = min(max_windows, 6)

    balanced_rows = []
    for pid, grp in agg.groupby("pid"):
        grp_sorted = grp.sort_values("window")
        balanced_rows.append(grp_sorted.head(max_windows))
    balanced = pd.concat(balanced_rows, ignore_index=True)

    balanced["rater"] = balanced.groupby("pid").cumcount()

    try:
        icc_result = pg.intraclass_corr(
            data=balanced, targets="pid", raters="rater", ratings="value"
        )
        row_3k = icc_result[icc_result["Type"] == "ICC(C,k)"]
        if len(row_3k) > 0:
            icc_val = row_3k["ICC"].values[0]
            ci = row_3k["CI95"].values[0]
            return {
                "ICC": float(icc_val),
                "ICC_CI_lo": float(ci[0]),
                "ICC_CI_hi": float(ci[1]),
                "N_persons": n_persons,
                "N_windows_median": float(n_windows_median),
            }
    except Exception as e:
        print(f"    ICC computation failed: {e}", flush=True)

    return {
        "ICC": np.nan, "ICC_CI_lo": np.nan, "ICC_CI_hi": np.nan,
        "N_persons": n_persons, "N_windows_median": float(n_windows_median),
    }


def run_temporal_reliability():
    print("=" * 70, flush=True)
    print("ANALYSIS 44: Sensing Temporal Reliability Decay", flush=True)
    print("=" * 70, flush=True)

    rows = []

    for modality_file, col_substr, feat_label in FEATURE_MAP:
        print(f"\n  Feature: {feat_label}", flush=True)
        daily_df = load_daily_feature(modality_file, col_substr, feat_label)
        if daily_df is None:
            print(f"    No data found, skipping", flush=True)
            continue

        daily_df = daily_df.dropna(subset=["value"])
        n_total = daily_df["pid"].nunique()
        print(f"    Total: {len(daily_df)} obs, {n_total} persons", flush=True)

        for window_days in WINDOW_SIZES:
            result = compute_icc_by_windows(daily_df, window_days)
            icc_str = f"{result['ICC']:.3f}" if not np.isnan(result['ICC']) else "NaN"
            ci_lo = f"{result['ICC_CI_lo']:.3f}" if not np.isnan(result['ICC_CI_lo']) else "NaN"
            ci_hi = f"{result['ICC_CI_hi']:.3f}" if not np.isnan(result['ICC_CI_hi']) else "NaN"
            print(f"    {window_days:3d}-day windows: ICC={icc_str} "
                  f"[{ci_lo}, {ci_hi}] "
                  f"(N={result['N_persons']})", flush=True)

            rows.append({
                "Feature": feat_label,
                "Window_days": window_days,
                **result,
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "temporal_reliability.csv", index=False)
    print(f"\n  Saved: {OUT / 'temporal_reliability.csv'}", flush=True)

    # Figure: ICC decay curves
    fig, ax = plt.subplots(figsize=(10, 6))

    features = df_out["Feature"].unique()
    cmap = plt.cm.Set2
    colors = {f: cmap(i / len(features)) for i, f in enumerate(features)}

    for feat in features:
        sub = df_out[df_out["Feature"] == feat].sort_values("Window_days")
        ax.plot(sub["Window_days"], sub["ICC"], "o-", color=colors[feat],
                label=feat, markersize=6, linewidth=2)
        ax.fill_between(sub["Window_days"], sub["ICC_CI_lo"], sub["ICC_CI_hi"],
                        alpha=0.15, color=colors[feat])

    ax.axhline(BFI_44_RETEST, color="#e74c3c", linestyle="--", linewidth=2,
               label=f"BFI-44 test-retest (r={BFI_44_RETEST})")
    ax.axhline(BFI_10_RETEST, color="#e74c3c", linestyle=":", linewidth=2,
               label=f"BFI-10 test-retest (r={BFI_10_RETEST})")

    ax.set_xlabel("Window Size (days)", fontsize=12)
    ax.set_ylabel("ICC(3,k)", fontsize=12)
    ax.set_title("Sensing Feature Temporal Reliability vs Personality Test-Retest",
                 fontweight="bold", fontsize=13)
    ax.set_xticks(WINDOW_SIZES)
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT / "figure_temporal_reliability.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT / 'figure_temporal_reliability.png'}", flush=True)

    return df_out


if __name__ == "__main__":
    result = run_temporal_reliability()
    print("\n✓ Analysis 44 complete.", flush=True)
    print(result.to_string(index=False), flush=True)
