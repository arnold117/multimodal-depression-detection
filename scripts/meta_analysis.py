#!/usr/bin/env python3
"""
Phase 14a: Mini Random-Effects Meta-Analysis
=============================================
Pool Neuroticism → Mental Health correlations across 3 studies.

Constructs pooled:
  - N → Depression  (PHQ-9 / CES-D / BDI-II)
  - N → Anxiety     (PSS / STAI / STAI)
  - N → Loneliness  (UCLA-like across 3 studies)
  - C → GPA         (2 studies only)

Method: Hunter-Schmidt random-effects (r-to-z, inverse-variance weights).
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/comparison")
OUT.mkdir(parents=True, exist_ok=True)


# ── Helper functions ────────────────────────────────────────────────────
def r_to_z(r):
    return np.arctanh(r)

def z_to_r(z):
    return np.tanh(z)

def meta_random_effects(rs, ns):
    """
    Random-effects meta-analysis using Fisher z transformation.
    Returns pooled_r, ci_lo, ci_hi, Q, I2, p.
    """
    k = len(rs)
    zs = np.array([r_to_z(r) for r in rs])
    ws = np.array([n - 3 for n in ns])  # inverse-variance weight for z

    # Fixed-effect pooled z
    z_fe = np.sum(ws * zs) / np.sum(ws)

    # Cochran's Q
    Q = np.sum(ws * (zs - z_fe) ** 2)
    df = k - 1
    p_Q = 1 - stats.chi2.cdf(Q, df) if df > 0 else 1.0

    # Between-study variance (tau^2, DerSimonian-Laird)
    C = np.sum(ws) - np.sum(ws ** 2) / np.sum(ws)
    tau2 = max(0, (Q - df) / C) if C > 0 else 0

    # Random-effects weights
    ws_re = 1.0 / (1.0 / ws + tau2)
    z_re = np.sum(ws_re * zs) / np.sum(ws_re)
    se_re = 1.0 / np.sqrt(np.sum(ws_re))

    # 95% CI
    z_lo = z_re - 1.96 * se_re
    z_hi = z_re + 1.96 * se_re

    # I^2
    I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0

    # Test of pooled effect
    p_effect = 2 * (1 - stats.norm.cdf(abs(z_re / se_re)))

    return {
        "pooled_r": z_to_r(z_re),
        "ci_lo": z_to_r(z_lo),
        "ci_hi": z_to_r(z_hi),
        "Q": Q,
        "I2": I2,
        "p_heterogeneity": p_Q,
        "p_effect": p_effect,
        "tau2": tau2,
        "k": k,
        "total_N": sum(ns),
    }


# ── Data: r values from three_study_mh_correlations.csv ─────────────
corr = pd.read_csv("results/comparison/three_study_mh_correlations.csv")

# Define pooling groups
pools = {
    "N → Depression": {
        "trait": "neuroticism",
        "construct": "Depression",
    },
    "N → Anxiety/Stress": {
        "trait": "neuroticism",
        "construct": "State Anxiety",
    },
    "N → Loneliness": {
        "trait": "neuroticism",
        "construct": "Loneliness",
    },
    "E → Loneliness": {
        "trait": "extraversion",
        "construct": "Loneliness",
    },
    "C → Depression": {
        "trait": "conscientiousness",
        "construct": "Depression",
    },
}

# Also add C → GPA from Study 1 and Study 2 (not in MH correlations file)
# Study 1: r=0.552, N=28; Study 2: r=0.263, N=220
gpa_pool = {
    "C → GPA": {
        "rs": [0.552, 0.263],
        "ns": [28, 220],
    }
}

# Also add N → Perceived Stress (S1 and S3 only, S2 has no PSS)
# Pull from the correlation file
stress_row = corr[(corr["Construct"] == "Perceived Stress") & (corr["Trait"] == "neuroticism")]


# ── Run meta-analyses ───────────────────────────────────────────────
results = []

for label, spec in pools.items():
    row = corr[(corr["Construct"] == spec["construct"]) & (corr["Trait"] == spec["trait"])]
    if len(row) == 0:
        continue
    row = row.iloc[0]

    rs, ns = [], []
    for prefix in ["S1", "S2", "S3"]:
        r_val = row.get(f"{prefix}_r")
        n_val = row.get(f"{prefix}_N")
        if pd.notna(r_val) and pd.notna(n_val):
            rs.append(float(r_val))
            ns.append(int(float(n_val)))

    if len(rs) < 2:
        continue

    meta = meta_random_effects(rs, ns)
    meta["Label"] = label
    meta["studies"] = ", ".join(
        [f"S{i+1}(r={rs[i]:.3f},N={ns[i]})" for i in range(len(rs))]
    )
    results.append(meta)

# Also pool N → Perceived Stress (S1 + S3)
if len(stress_row) > 0:
    sr = stress_row.iloc[0]
    rs_stress, ns_stress = [], []
    for prefix in ["S1", "S3"]:
        r_val = sr.get(f"{prefix}_r")
        n_val = sr.get(f"{prefix}_N")
        if pd.notna(r_val) and pd.notna(n_val):
            rs_stress.append(float(r_val))
            ns_stress.append(int(float(n_val)))
    if len(rs_stress) >= 2:
        meta = meta_random_effects(rs_stress, ns_stress)
        meta["Label"] = "N → Perceived Stress"
        meta["studies"] = ", ".join(
            [f"S{[1,3][i]}(r={rs_stress[i]:.3f},N={ns_stress[i]})" for i in range(len(rs_stress))]
        )
        results.append(meta)

# C → GPA (hardcoded, not in MH correlation file)
for label, spec in gpa_pool.items():
    meta = meta_random_effects(spec["rs"], spec["ns"])
    meta["Label"] = label
    meta["studies"] = ", ".join(
        [f"S{i+1}(r={spec['rs'][i]:.3f},N={spec['ns'][i]})" for i in range(len(spec["rs"]))]
    )
    results.append(meta)

df = pd.DataFrame(results)
cols = ["Label", "k", "total_N", "pooled_r", "ci_lo", "ci_hi", "p_effect",
        "Q", "I2", "p_heterogeneity", "tau2", "studies"]
df = df[cols]
df.to_csv(OUT / "meta_analysis.csv", index=False)

print("\n" + "=" * 70)
print("MINI META-ANALYSIS: Random-Effects Pooled Correlations")
print("=" * 70)
for _, row in df.iterrows():
    sig = "***" if row["p_effect"] < 0.001 else "**" if row["p_effect"] < 0.01 else "*" if row["p_effect"] < 0.05 else "n.s."
    het = f"I²={row['I2']:.1f}%" + (" (heterogeneous)" if row["p_heterogeneity"] < 0.05 else " (homogeneous)")
    print(f"\n  {row['Label']} (k={row['k']}, N={row['total_N']})")
    print(f"    Pooled r = {row['pooled_r']:.3f} [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}] {sig}")
    print(f"    {het}")


# ── Forest plot ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Sort by pooled_r
df_plot = df.sort_values("pooled_r", ascending=True).reset_index(drop=True)

y_positions = range(len(df_plot))
colors = []
for _, row in df_plot.iterrows():
    if "N →" in row["Label"]:
        colors.append("#e74c3c")  # red for neuroticism
    elif "C →" in row["Label"]:
        colors.append("#2ecc71")  # green for conscientiousness
    else:
        colors.append("#3498db")  # blue for extraversion

for i, (_, row) in enumerate(df_plot.iterrows()):
    ax.errorbar(
        row["pooled_r"], i,
        xerr=[[row["pooled_r"] - row["ci_lo"]], [row["ci_hi"] - row["pooled_r"]]],
        fmt="D", color=colors[i], markersize=8, capsize=5, linewidth=2,
    )
    sig = "***" if row["p_effect"] < 0.001 else "**" if row["p_effect"] < 0.01 else "*" if row["p_effect"] < 0.05 else ""
    ax.annotate(
        f'r={row["pooled_r"]:.3f}{sig}  I²={row["I2"]:.0f}%',
        xy=(row["ci_hi"] + 0.02, i), va="center", fontsize=9,
    )

ax.set_yticks(list(y_positions))
ax.set_yticklabels(df_plot["Label"], fontsize=11)
ax.axvline(0, color="grey", linestyle="--", alpha=0.5)
ax.set_xlabel("Pooled Correlation (r) with 95% CI", fontsize=12)
ax.set_title("Random-Effects Meta-Analysis Across 3 Universities", fontsize=13, fontweight="bold")

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="D", color="#e74c3c", linestyle="", markersize=8, label="Neuroticism"),
    Line2D([0], [0], marker="D", color="#2ecc71", linestyle="", markersize=8, label="Conscientiousness"),
    Line2D([0], [0], marker="D", color="#3498db", linestyle="", markersize=8, label="Extraversion"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

plt.tight_layout()
fig.savefig(OUT / "meta_analysis_forest.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUT / 'meta_analysis.csv'}")
print(f"Saved: {OUT / 'meta_analysis_forest.png'}")
