#!/usr/bin/env python3
"""
Phase 14b: GLOBEM Longitudinal Analyses
========================================
Two analyses using GLOBEM's repeated/pre-post data:

1. Weekly PHQ-4 Trajectory Analysis (W2-W4, ~550 ppl, ~10 weeks)
   - Does Neuroticism predict worsening depression trajectory (slope)?
   - Linear mixed model: PHQ4 ~ week * Neuroticism + (1+week|pid)

2. Pre→Post Change Prediction (all 4 cohorts)
   - Does personality predict within-semester change (ΔPost-Pre)?
   - Outcomes: STAI, PSS-10, CESD-10, UCLA Loneliness
   - Multiple regression: Δoutcome ~ Big Five + cohort

These address the "cross-sectional limitation" reviewer concern.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/globem/tables")
FIG = Path("results/globem/figures")
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]


# ═══════════════════════════════════════════════════════════════════════
# PART 1: Weekly PHQ-4 Trajectory
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 1: Weekly PHQ-4 Trajectory Analysis")
print("=" * 70)

# Load weekly PHQ-4 (W2-W4) and personality
weekly_frames = []
pers_frames = []

for w in [2, 3, 4]:
    dep = pd.read_csv(f"data/raw/globem/INS-W_{w}/SurveyData/dep_weekly.csv")
    dep = dep[["pid", "date", "phq4"]].dropna(subset=["phq4"])
    dep["cohort"] = w

    pre = pd.read_csv(f"data/raw/globem/INS-W_{w}/SurveyData/pre.csv")
    pers_cols = ["pid"] + [c for c in pre.columns if "BFI10" in c]
    pers = pre[pers_cols].copy()
    pers.columns = ["pid"] + TRAITS
    pers["cohort"] = w

    weekly_frames.append(dep)
    pers_frames.append(pers)

weekly = pd.concat(weekly_frames, ignore_index=True)
pers_all = pd.concat(pers_frames, ignore_index=True).drop_duplicates("pid")

# Compute week number (0-indexed from each person's first observation)
weekly["date"] = pd.to_datetime(weekly["date"])
weekly = weekly.sort_values(["pid", "date"])
first_date = weekly.groupby("pid")["date"].transform("min")
weekly["week"] = ((weekly["date"] - first_date).dt.days / 7).round().astype(int)

# Merge personality
weekly = weekly.merge(pers_all, on="pid", how="inner", suffixes=("", "_pers"))
# Drop cohort duplicate
if "cohort_pers" in weekly.columns:
    weekly = weekly.drop(columns=["cohort_pers"])

# Filter: need ≥ 3 timepoints per person
tp_counts = weekly.groupby("pid").size()
valid_pids = tp_counts[tp_counts >= 3].index
weekly = weekly[weekly["pid"].isin(valid_pids)]

print(f"  N = {weekly['pid'].nunique()} participants, {len(weekly)} observations")
print(f"  Timepoints per person: mean={weekly.groupby('pid').size().mean():.1f}")

# Approach: OLS per-person slopes, then correlate with personality
# (More robust than mixed model with convergence issues)
person_slopes = []
for pid, grp in weekly.groupby("pid"):
    if len(grp) < 3:
        continue
    slope, intercept, r, p, se = stats.linregress(grp["week"], grp["phq4"])
    row = {"pid": pid, "phq4_slope": slope, "phq4_intercept": intercept,
           "phq4_mean": grp["phq4"].mean()}
    person_slopes.append(row)

slopes_df = pd.DataFrame(person_slopes).merge(pers_all, on="pid", how="inner")
print(f"  Persons with slopes: {len(slopes_df)}")
print(f"  PHQ-4 slope: mean={slopes_df['phq4_slope'].mean():.4f}, "
      f"SD={slopes_df['phq4_slope'].std():.4f}")

# Correlate each trait with slope and mean
print("\n  Trait → PHQ-4 slope (worsening trajectory):")
slope_results = []
for trait in TRAITS:
    mask = slopes_df[[trait, "phq4_slope"]].dropna().index
    r, p = stats.pearsonr(slopes_df.loc[mask, trait], slopes_df.loc[mask, "phq4_slope"])
    n = len(mask)
    slope_results.append({"Trait": trait, "Outcome": "phq4_slope", "r": r, "p": p, "N": n})
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"    {trait:20s}: r={r:+.3f} (p={p:.4f}) {sig}")

print("\n  Trait → PHQ-4 mean level:")
for trait in TRAITS:
    mask = slopes_df[[trait, "phq4_mean"]].dropna().index
    r, p = stats.pearsonr(slopes_df.loc[mask, trait], slopes_df.loc[mask, "phq4_mean"])
    slope_results.append({"Trait": trait, "Outcome": "phq4_mean", "r": r, "p": p, "N": len(mask)})
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"    {trait:20s}: r={r:+.3f} (p={p:.4f}) {sig}")

slope_df_out = pd.DataFrame(slope_results)
slope_df_out.to_csv(OUT / "phq4_trajectory.csv", index=False)


# ── Trajectory figure ───────────────────────────────────────────────
# Split by N tertiles
n_col = "neuroticism"
slopes_df["N_tertile"] = pd.qcut(slopes_df[n_col], 3, labels=["Low N", "Mid N", "High N"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Mean trajectory by N tertile
ax = axes[0]
colors_t = {"Low N": "#2ecc71", "Mid N": "#f39c12", "High N": "#e74c3c"}
for tert in ["Low N", "Mid N", "High N"]:
    pids = slopes_df[slopes_df["N_tertile"] == tert]["pid"]
    sub = weekly[weekly["pid"].isin(pids)]
    means = sub.groupby("week")["phq4"].agg(["mean", "sem"]).reset_index()
    means = means[means["week"] <= 10]  # cap at 10 weeks
    ax.plot(means["week"], means["mean"], "o-", color=colors_t[tert], label=tert, linewidth=2)
    ax.fill_between(means["week"], means["mean"] - means["sem"],
                    means["mean"] + means["sem"], alpha=0.15, color=colors_t[tert])

ax.set_xlabel("Week", fontsize=12)
ax.set_ylabel("PHQ-4 Score", fontsize=12)
ax.set_title("A. Weekly PHQ-4 by Neuroticism Tertile", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)

# Panel B: Scatter N vs slope
ax = axes[1]
ax.scatter(slopes_df[n_col], slopes_df["phq4_slope"], alpha=0.3, s=20, color="#e74c3c")
# Regression line
mask = slopes_df[[n_col, "phq4_slope"]].dropna().index
z = np.polyfit(slopes_df.loc[mask, n_col], slopes_df.loc[mask, "phq4_slope"], 1)
x_line = np.linspace(slopes_df[n_col].min(), slopes_df[n_col].max(), 100)
ax.plot(x_line, np.polyval(z, x_line), "-", color="#c0392b", linewidth=2)
r_ns, p_ns = stats.pearsonr(slopes_df.loc[mask, n_col], slopes_df.loc[mask, "phq4_slope"])
ax.set_xlabel("Neuroticism", fontsize=12)
ax.set_ylabel("PHQ-4 Weekly Slope", fontsize=12)
ax.set_title(f"B. Neuroticism → PHQ-4 Trajectory (r={r_ns:+.3f}, p={p_ns:.3f})",
             fontsize=13, fontweight="bold")
ax.axhline(0, color="grey", linestyle="--", alpha=0.5)

plt.tight_layout()
fig.savefig(FIG / "figure16_phq4_trajectory.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {FIG / 'figure16_phq4_trajectory.png'}")


# ═══════════════════════════════════════════════════════════════════════
# PART 2: Pre→Post Change Prediction
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: Pre→Post Change Prediction")
print("=" * 70)

outcome_maps = {
    "STAI": ("STAIS_PRE", "STAIS_POST"),
    "PSS-10": ("PSS_10items_PRE", "PSS_10items_POST"),
    "CESD-10": ("CESD_10items_PRE", "CESD_10items_POST"),
    "UCLA": ("UCLA_10items_PRE", "UCLA_10items_POST"),
}

all_change = []

for w in [1, 2, 3, 4]:
    pre = pd.read_csv(f"data/raw/globem/INS-W_{w}/SurveyData/pre.csv")
    post = pd.read_csv(f"data/raw/globem/INS-W_{w}/SurveyData/post.csv")

    merged = pre.merge(post, on="pid", how="inner", suffixes=("_pre", "_post"))
    merged["cohort"] = w

    # Personality (GLOBEM uses "extroversion" spelling)
    bfi_map = {
        "extraversion": "BFI10_extroversion_PRE",
        "agreeableness": "BFI10_agreeableness_PRE",
        "conscientiousness": "BFI10_conscientiousness_PRE",
        "neuroticism": "BFI10_neuroticism_PRE",
        "openness": "BFI10_openness_PRE",
    }
    for trait, col in bfi_map.items():
        merged[trait] = merged[col]

    # Change scores
    for outcome, (pre_col, post_col) in outcome_maps.items():
        if pre_col in merged.columns and post_col in merged.columns:
            merged[f"delta_{outcome}"] = merged[post_col] - merged[pre_col]

    delta_cols = [f"delta_{o}" for o in outcome_maps if f"delta_{o}" in merged.columns]
    all_change.append(merged[["pid", "cohort"] + TRAITS + delta_cols].copy())

change_df = pd.concat(all_change, ignore_index=True)

# Drop rows without personality
change_df = change_df.dropna(subset=TRAITS)
print(f"  N = {len(change_df)} participants with personality + pre/post data")

# For each outcome, correlate Big Five with change score
change_results = []
for outcome in outcome_maps:
    delta_col = f"delta_{outcome}"
    if delta_col not in change_df.columns:
        continue

    valid = change_df.dropna(subset=[delta_col])
    n = len(valid)
    mean_delta = valid[delta_col].mean()
    sd_delta = valid[delta_col].std()

    print(f"\n  {outcome}: N={n}, Δ mean={mean_delta:+.2f}, SD={sd_delta:.2f}")

    for trait in TRAITS:
        r, p = stats.pearsonr(valid[trait], valid[delta_col])
        change_results.append({
            "Outcome": outcome, "Trait": trait, "r": r, "p": p, "N": n,
            "delta_mean": mean_delta, "delta_sd": sd_delta,
        })
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {trait:20s}: r={r:+.3f} (p={p:.4f}) {sig}")

    # Multiple regression: Δ ~ Big Five + cohort
    from sklearn.linear_model import LinearRegression
    X = valid[TRAITS + ["cohort"]].values
    y = valid[delta_col].values
    mask_complete = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    if mask_complete.sum() > 10:
        X_c, y_c = X[mask_complete], y[mask_complete]
        reg = LinearRegression().fit(X_c, y_c)
        r2 = reg.score(X_c, y_c)
        print(f"    Multiple R² (Big Five + cohort → Δ{outcome}): {r2:.3f}")

change_out = pd.DataFrame(change_results)
change_out.to_csv(OUT / "pre_post_change.csv", index=False)


# ── Change prediction figure ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, outcome in enumerate(outcome_maps):
    ax = axes[idx // 2][idx % 2]
    delta_col = f"delta_{outcome}"
    if delta_col not in change_df.columns:
        continue

    valid = change_df.dropna(subset=[delta_col])

    # Bar chart of trait correlations with change
    rs_plot = []
    ps_plot = []
    for trait in TRAITS:
        r, p = stats.pearsonr(valid[trait], valid[delta_col])
        rs_plot.append(r)
        ps_plot.append(p)

    colors_bar = ["#e74c3c" if p < 0.05 else "#bdc3c7" for p in ps_plot]
    bars = ax.bar(range(len(TRAITS)), rs_plot, color=colors_bar, edgecolor="white")

    # Add significance markers
    for i, (r_val, p_val) in enumerate(zip(rs_plot, ps_plot)):
        if p_val < 0.001:
            marker = "***"
        elif p_val < 0.01:
            marker = "**"
        elif p_val < 0.05:
            marker = "*"
        else:
            marker = ""
        if marker:
            ax.text(i, r_val + 0.01 * np.sign(r_val), marker,
                    ha="center", va="bottom" if r_val > 0 else "top", fontsize=12)

    ax.set_xticks(range(len(TRAITS)))
    ax.set_xticklabels(["E", "A", "C", "N", "O"], fontsize=11)
    ax.set_ylabel("r with Δ(Post-Pre)", fontsize=10)
    ax.set_title(f"Δ{outcome}", fontsize=12, fontweight="bold")
    ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
    ax.set_ylim(-0.25, 0.25)

fig.suptitle("Personality → Within-Semester Change (Post – Pre)", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG / "figure17_pre_post_change.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {FIG / 'figure17_pre_post_change.png'}")

print("\n  Saved: results/globem/tables/phq4_trajectory.csv")
print("  Saved: results/globem/tables/pre_post_change.csv")
