#!/usr/bin/env python3
"""
Paper Figures — Publication-Ready Figures for JMIR Mental Health
================================================================
Generates 10 figures (8 main + 2 supplement) with consistent styling.

Usage:
    python scripts/05_paper_materials/paper_figures.py           # all figures
    python scripts/05_paper_materials/paper_figures.py fig2 fig5  # specific figures
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════
CORE = Path("results/core")
ROB = Path("results/robustness")
BY_STUDY = Path("results/by_study")
OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# Color Palette
# ═══════════════════════════════════════════════════════════════════════
C_PERS = "#2980b9"       # personality — blue
C_SENS = "#e74c3c"       # sensing — red
C_COMB = "#1abc9c"       # combined — teal
C_GREY = "#95a5a6"       # neutral/behavior-only — grey
C_DL = "#8e44ad"         # deep learning / MOMENT — purple
C_GB = "#f39c12"         # GradientBoosting — orange
C_BG = "#fafafa"         # figure background


def setup_style():
    """Configure matplotlib for publication figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": C_BG,
        "figure.facecolor": "white",
    })


# ═══════════════════════════════════════════════════════════════════════
# FIG 1: Study Design Overview
# ═══════════════════════════════════════════════════════════════════════
def fig1_study_design():
    """Schematic overview of three studies."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    studies = [
        {"name": "Study 1: StudentLife", "uni": "Dartmouth, 2013", "n": "N = 28",
         "pers": "BFI-44 (5 min)", "sensing": "Smartphone\n13 modalities, 87 features\n10 weeks",
         "outcomes": "PHQ-9, PSS,\nLoneliness, GPA"},
        {"name": "Study 2: NetHealth", "uni": "Notre Dame, 2015–19", "n": "N = 722",
         "pers": "BFI-44 (5 min)", "sensing": "Fitbit + Communication\n28 features\n4 years",
         "outcomes": "CES-D, STAI, BAI,\nLoneliness, Self-Esteem, GPA"},
        {"name": "Study 3: GLOBEM", "uni": "U. Washington, 2018–21", "n": "N = 809",
         "pers": "BFI-10 (1 min)", "sensing": "Fitbit + Phone + GPS\n19–2,597 features\n1 year",
         "outcomes": "BDI-II, STAI, PSS,\nCESD, UCLA Loneliness"},
    ]

    for i, s in enumerate(studies):
        cx = 1.7 + i * 3.0
        ax.text(cx, 5.6, s["name"], ha="center", va="center", fontsize=10,
                fontweight="bold", color="#2c3e50")
        ax.text(cx, 5.2, s["uni"], ha="center", va="center", fontsize=8, color="#7f8c8d")
        ax.text(cx, 4.85, s["n"], ha="center", va="center", fontsize=9,
                fontweight="bold", color="#2c3e50")

        rect_p = mpatches.FancyBboxPatch((cx - 1.2, 3.8), 2.4, 0.7,
                                          boxstyle="round,pad=0.1",
                                          facecolor=C_PERS, alpha=0.15, edgecolor=C_PERS, lw=1.5)
        ax.add_patch(rect_p)
        ax.text(cx, 4.15, s["pers"], ha="center", va="center", fontsize=8, color=C_PERS,
                fontweight="bold")

        rect_s = mpatches.FancyBboxPatch((cx - 1.2, 2.3), 2.4, 1.2,
                                          boxstyle="round,pad=0.1",
                                          facecolor=C_SENS, alpha=0.10, edgecolor=C_SENS, lw=1.5)
        ax.add_patch(rect_s)
        ax.text(cx, 2.9, s["sensing"], ha="center", va="center", fontsize=7.5,
                color=C_SENS, fontweight="bold")

        ax.annotate("", xy=(cx, 1.6), xytext=(cx, 2.2),
                    arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5))

        rect_o = mpatches.FancyBboxPatch((cx - 1.2, 0.4), 2.4, 1.1,
                                          boxstyle="round,pad=0.1",
                                          facecolor="#ecf0f1", edgecolor="#bdc3c7", lw=1)
        ax.add_patch(rect_o)
        ax.text(cx, 0.95, s["outcomes"], ha="center", va="center", fontsize=7.5,
                color="#2c3e50")

    ax.text(5, 5.95, "Can Passive Sensing Replace Questionnaires?", ha="center",
            fontsize=13, fontweight="bold", color="#2c3e50")

    fig.savefig(OUT / "fig1_study_design.png")
    plt.close()
    print("  Saved fig1_study_design.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# FIG 2: Grand Synthesis — Personality vs Sensing R²
# ═══════════════════════════════════════════════════════════════════════
def fig2_grand_synthesis():
    """Horizontal bar chart: personality vs sensing R² across 15 outcomes."""
    df = pd.read_csv(CORE / "grand_synthesis.csv")

    fig, ax = plt.subplots(figsize=(7.5, 6))

    df = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(df))
    h = 0.35

    ax.barh(y + h/2, df["R2_personality"], h, color=C_PERS, label="Personality",
            edgecolor="white", zorder=3)
    ax.barh(y - h/2, df["R2_sensing"], h, color=C_SENS, label="Sensing",
            edgecolor="white", zorder=3)

    ax.errorbar(df["R2_personality"], y + h/2,
                xerr=[df["R2_personality"] - df["R2_pers_ci_lo"],
                      df["R2_pers_ci_hi"] - df["R2_personality"]],
                fmt="none", color=C_PERS, capsize=2, lw=0.8, zorder=4)
    ax.errorbar(df["R2_sensing"], y - h/2,
                xerr=[df["R2_sensing"] - df["R2_sens_ci_lo"],
                      df["R2_sens_ci_hi"] - df["R2_sensing"]],
                fmt="none", color=C_SENS, capsize=2, lw=0.8, zorder=4)

    labels = [f"{row.Study} — {row.Outcome}" for _, row in df.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)

    study_boundaries = []
    prev_study = None
    for i, row in df.iterrows():
        if prev_study and row.Study != prev_study:
            study_boundaries.append(i - 0.5)
        prev_study = row.Study
    for b in study_boundaries:
        ax.axhline(b, color="#bdc3c7", lw=0.8, ls="--", zorder=1)

    ax.axvline(0, color="#2c3e50", lw=1, zorder=2)

    n_wins = df["Pers_wins"].sum()
    ax.text(0.02, 0.02, f"Personality wins {n_wins}/{len(df)} ({100*n_wins//len(df)}%)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=11,
            fontweight="bold", color=C_PERS,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_PERS, alpha=0.9))

    ax.set_xlim(-1.5, 1.0)  # clip extreme S1 CIs for readability
    ax.set_xlabel("Cross-Validated R²")
    ax.set_title("Personality vs Passive Sensing: 15 Outcomes Across 3 Studies")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="x", alpha=0.3)

    fig.savefig(OUT / "fig2_grand_synthesis.png")
    plt.close()
    print("  Saved fig2_grand_synthesis.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# FIG 3: Meta-Analysis Forest Plot
# ═══════════════════════════════════════════════════════════════════════
def fig3_meta_forest():
    """Forest plot of meta-analytic personality–outcome correlations."""
    df = pd.read_csv(CORE / "meta_analysis.csv")

    fig, ax = plt.subplots(figsize=(7.5, 4))
    y = np.arange(len(df))

    for col in df.columns:
        if "ci_lo" in col.lower() or "ci_hi" in col.lower():
            df[col] = pd.to_numeric(df[col], errors="coerce")

    r_col = [c for c in df.columns if "pooled" in c.lower() or c == "r" or c == "pooled_r"]
    r_col = r_col[0] if r_col else "pooled_r"
    lo_col = [c for c in df.columns if "ci_lo" in c.lower()]
    lo_col = lo_col[0] if lo_col else "CI_lo"
    hi_col = [c for c in df.columns if "ci_hi" in c.lower()]
    hi_col = hi_col[0] if hi_col else "CI_hi"
    label_col = [c for c in df.columns if "label" in c.lower() or c == "Label"]
    label_col = label_col[0] if label_col else df.columns[0]

    r_vals = pd.to_numeric(df[r_col], errors="coerce")
    ci_lo = pd.to_numeric(df[lo_col], errors="coerce")
    ci_hi = pd.to_numeric(df[hi_col], errors="coerce")

    colors = [C_PERS if r > 0 else C_SENS for r in r_vals]

    ax.scatter(r_vals, y, color=colors, s=80, zorder=3, edgecolors="white", lw=0.5)
    ax.hlines(y, ci_lo, ci_hi, colors=colors, lw=2, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(df[label_col], fontsize=9)
    ax.axvline(0, color="#2c3e50", lw=1, ls="--")
    ax.set_xlabel("Pooled r [95% CI]")
    ax.set_title("Meta-Analytic Personality–Outcome Correlations Across 3 Studies")
    ax.grid(axis="x", alpha=0.3)

    fig.savefig(OUT / "fig3_meta_forest.png")
    plt.close()
    print("  Saved fig3_meta_forest.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# FIG 4: SHAP Heatmap
# ═══════════════════════════════════════════════════════════════════════
def fig4_shap_heatmap():
    """Heatmap of SHAP importance: traits (rows) × study-outcomes (cols)."""
    traits = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
    trait_labels = ["E", "A", "C", "N", "O"]

    shap_files = {
        "S2-STAI": BY_STUDY / "s2/tables/shap_personality_stai.csv",
        "S2-BAI": BY_STUDY / "s2/tables/shap_personality_bai.csv",
        "S2-GPA": BY_STUDY / "s2/tables/shap_personality_gpa.csv",
        "S3-BDI": BY_STUDY / "s3/tables/shap_personality_bdiii.csv",
        "S3-CESD": BY_STUDY / "s3/tables/shap_personality_cesd.csv",
        "S3-PSS": BY_STUDY / "s3/tables/shap_personality_pss10.csv",
        "S3-STAI": BY_STUDY / "s3/tables/shap_personality_stai.csv",
        "S3-UCLA": BY_STUDY / "s3/tables/shap_personality_ucla.csv",
    }

    data = {}
    for label, fpath in shap_files.items():
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        ridge_row = df[df["Model"].str.contains("Ridge", case=False)]
        if len(ridge_row) == 0:
            ridge_row = df.iloc[:1]
        vals = []
        for t in traits:
            v = pd.to_numeric(ridge_row[t].values[0], errors="coerce") if t in ridge_row.columns else 0
            vals.append(v)
        data[label] = vals

    mat = pd.DataFrame(data, index=trait_labels).T

    fig, ax = plt.subplots(figsize=(5, 4.5))
    annot_mat = mat.copy().astype(str)
    for i in range(mat.shape[0]):
        ranks = mat.iloc[i].rank(ascending=False).astype(int)
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            star = " *" if ranks.iloc[j] == 1 else ""
            annot_mat.iloc[i, j] = f"{val:.1f}{star}"

    sns.heatmap(mat, annot=annot_mat, fmt="", cmap="YlOrRd", ax=ax,
                linewidths=0.5, linecolor="white", cbar_kws={"label": "Mean |SHAP|"})

    ax.set_title("SHAP Feature Importance: Neuroticism Dominates")
    ax.set_ylabel("")
    ax.set_xlabel("")

    fig.savefig(OUT / "fig4_shap_heatmap.png")
    plt.close()
    print("  Saved fig4_shap_heatmap.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# FIG 5: Deep Learning Baseline
# ═══════════════════════════════════════════════════════════════════════
def fig5_deep_learning():
    """Two-row grouped bar chart: 5 models × 8 outcomes."""
    df = pd.read_csv(ROB / "deep_learning_comparison.csv")

    model_order = ["Personality (Ridge)", "Sensing PCA (Ridge)",
                   "GradientBoosting+Optuna", "1D-CNN", "MOMENT → Ridge"]
    colors = {
        "Personality (Ridge)": C_PERS,
        "Sensing PCA (Ridge)": C_GREY,
        "GradientBoosting+Optuna": C_GB,
        "1D-CNN": C_SENS,
        "MOMENT → Ridge": C_DL,
    }

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7), sharex=False)

    for ax, study, title in zip(axes, ["S3", "S2"],
                                 ["Study 3: GLOBEM (N≈700)", "Study 2: NetHealth (N≈500)"]):
        sub = df[df["Study"] == study]
        if len(sub) == 0:
            continue

        outcomes = sub["Outcome"].unique()
        x = np.arange(len(outcomes))
        width = 0.15

        for i, model in enumerate(model_order):
            vals = []
            for outcome in outcomes:
                row = sub[(sub["Outcome"] == outcome) & (sub["Model"] == model)]
                vals.append(row["R2_mean"].values[0] if len(row) > 0 else 0)
            ax.bar(x + i * width, vals, width, label=model,
                   color=colors.get(model, "#333"), edgecolor="white", zorder=3)

        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(outcomes, fontsize=9)
        ax.set_ylabel("Cross-Validated R²")
        ax.set_title(title, fontsize=11)
        ax.axhline(0, color="#2c3e50", lw=1, zorder=2)
        ax.set_ylim(-0.25, None)
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(fontsize=7.5, ncol=3, loc="upper right",
                   frameon=True, fancybox=True)

    axes[1].annotate("MOMENT R² = −1.0 to −1.7\n(off scale, bars truncated)",
                     xy=(0.97, 0.05), xycoords="axes fraction",
                     ha="right", va="bottom", fontsize=8, color=C_DL,
                     fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_DL, alpha=0.9))

    plt.suptitle("Deep Learning Cannot Rescue Passive Sensing", fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT / "fig5_deep_learning.png")
    plt.close()
    print("  Saved fig5_deep_learning.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# FIG 6: Dose-Response
# ═══════════════════════════════════════════════════════════════════════
def fig6_dose_response():
    """Line plot: R² vs days of sensing data, with personality reference band."""
    df = pd.read_csv(ROB / "dose_response.csv")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    outcomes = df["Outcome"].unique()
    cmap = plt.cm.tab10

    pers_vals = df.groupby("Outcome")["R2_personality"].mean()
    pers_lo = pers_vals.min()
    pers_hi = pers_vals.max()
    ax.axhspan(pers_lo, pers_hi, color=C_PERS, alpha=0.12, zorder=1)
    ax.axhline(pers_vals.mean(), color=C_PERS, ls="--", lw=1.5, zorder=2,
               label=f"Personality R² ({pers_lo:.2f}–{pers_hi:.2f})")

    for i, outcome in enumerate(outcomes):
        sub = df[df["Outcome"] == outcome].sort_values("N_days")
        ax.plot(sub["N_days"], sub["R2_behavior"], "o-", color=cmap(i),
                label=f"{outcome} (sensing)", markersize=5, lw=1.5, zorder=3)

    ax.axhline(0, color="#2c3e50", lw=0.8, ls=":", zorder=1)
    ax.set_xlabel("Days of Sensing Data")
    ax.set_ylabel("Cross-Validated R²")
    ax.set_title("More Sensing Data Does Not Improve Prediction")
    ax.set_xticks(df["N_days"].unique())
    ax.legend(fontsize=8, loc="center right", frameon=True)
    ax.grid(alpha=0.3)

    fig.savefig(OUT / "fig6_dose_response.png")
    plt.close()
    print("  Saved fig6_dose_response.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# FIG 7: Temporal Reliability (ICC Decay)
# ═══════════════════════════════════════════════════════════════════════
def fig7_temporal_reliability():
    """ICC decay curves with personality reference band."""
    df = pd.read_csv(ROB / "temporal_reliability.csv")

    fig, ax = plt.subplots(figsize=(7.5, 5))

    features = df["Feature"].unique()
    cmap = plt.cm.Set2

    # Stagger label y-positions to avoid overlap at 60-day endpoint
    # Order in data: Steps, Sleep Duration, Sleep Efficiency, Screen Time, Calls (Incoming), Home Time
    # 60-day ICC values: Steps~0.873, SleepDur~0.745, SleepEff~0.928, Screen~0.879, Calls~0.876, Home~0.727
    label_offsets = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
    label_y_nudge = [0.020, 0, 0, -0.020, 0.008, 0]

    for i, feat in enumerate(features):
        sub = df[df["Feature"] == feat].sort_values("Window_days")
        color = cmap(i / len(features))
        ax.plot(sub["Window_days"], sub["ICC"], "o-", color=color,
                markersize=6, lw=2, zorder=3)
        ax.fill_between(sub["Window_days"], sub["ICC_CI_lo"], sub["ICC_CI_hi"],
                        alpha=0.12, color=color)
        last = sub.iloc[-1]
        nudge = label_y_nudge[i] if i < len(label_y_nudge) else 0
        ax.text(last["Window_days"] + label_offsets[min(i, len(label_offsets)-1)],
                last["ICC"] + nudge, feat,
                fontsize=7, color=color, va="center")

    ax.axhspan(0.75, 0.85, color=C_PERS, alpha=0.15, zorder=1)
    ax.axhline(0.85, color=C_PERS, ls="--", lw=1.5, zorder=2)
    ax.axhline(0.75, color=C_PERS, ls=":", lw=1.5, zorder=2)
    ax.text(5, 0.86, "BFI-44 retest r = .85", fontsize=8, color=C_PERS, va="bottom")
    ax.text(5, 0.74, "BFI-10 retest r = .75", fontsize=8, color=C_PERS, va="top")

    ax.annotate("Sensing features are STABLE\nbut predict nothing",
                xy=(45, 0.55), fontsize=10, fontweight="bold", color="#2c3e50",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.4", fc="#fff9c4", ec="#f9a825", alpha=0.9))

    ax.set_xlabel("Window Size (days)")
    ax.set_ylabel("ICC(3,k)")
    ax.set_title("Sensing Temporal Reliability vs Personality Test-Retest")
    ax.set_xticks([7, 14, 30, 45, 60])
    ax.set_xlim(3, 85)
    ax.set_ylim(0.55, 1.02)
    ax.grid(alpha=0.3)

    fig.savefig(OUT / "fig7_temporal_reliability.png")
    plt.close()
    print("  Saved fig7_temporal_reliability.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# FIG 8: NNS Clinical Significance
# ═══════════════════════════════════════════════════════════════════════
def fig8_nns():
    """Grouped bar chart: TP per 100 screened by feature set."""
    df = pd.read_csv(ROB / "nns_comparison.csv")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    feature_order = ["Pers-only", "Pers+Beh", "Beh-only"]
    colors = {"Pers-only": C_PERS, "Pers+Beh": C_COMB, "Beh-only": C_GREY}
    outcomes = df["Outcome"].unique()
    x = np.arange(len(outcomes))
    width = 0.25

    for i, fs in enumerate(feature_order):
        vals = []
        for outcome in outcomes:
            row = df[(df["Outcome"] == outcome) & (df["Features"] == fs)]
            vals.append(row["TP_per_100"].values[0] if len(row) > 0 else 0)
        ax.bar(x + i * width, vals, width, label=fs,
               color=colors.get(fs, "#333"), edgecolor="white", zorder=3)
        for xi, v in zip(x + i * width, vals):
            if v > 0:
                ax.text(xi, v + 0.3, f"{v:.0f}", ha="center", va="bottom",
                        fontsize=7, color=colors.get(fs, "#333"))

    ax.set_xticks(x + width)
    ax.set_xticklabels(outcomes, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("True Positives per 100 Screened")
    ax.set_title("Screening Yield: Personality Catches More True Cases")
    ax.legend(fontsize=9, frameon=True)
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(OUT / "fig8_nns.png")
    plt.close()
    print("  Saved fig8_nns.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# FIG S1: Cost-Effectiveness (Supplement)
# ═══════════════════════════════════════════════════════════════════════
def figS1_cost():
    """Scatter: R² vs time investment, with cost annotations."""
    df = pd.read_csv(CORE / "cost_effectiveness.csv")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    # Manual y-offsets to avoid label overlap
    label_offsets_map = {
        "Full Big Five (5 min)": (8, -18),
        "Neuroticism score (1 min)": (8, -15),
        "Pers + Sensing PCA": (5, 8),
        "Pers + Comm (S2 best)": (5, -15),
    }
    for _, row in df.iterrows():
        is_pers = "BFI" in row["Approach"] or "Neuroticism" in row["Approach"] or "personality" in row["Approach"].lower()
        color = C_PERS if is_pers else C_SENS
        marker = "o" if is_pers else "s"
        ax.scatter(row["Time_min"], row["R2"], color=color, marker=marker,
                   s=100, zorder=3, edgecolors="white", lw=0.5)
        default_offset = (8, 5) if row["R2"] > 0 else (8, -12)
        offset = label_offsets_map.get(row["Approach"], default_offset)
        ax.annotate(row["Approach"], (row["Time_min"], row["R2"]),
                    textcoords="offset points", xytext=offset,
                    fontsize=7, color=color)

    ax.axhline(0, color="#2c3e50", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("Assessment Time (minutes, log scale)")
    ax.set_ylabel("Cross-Validated R²")
    ax.set_title("Cost-Effectiveness: Brief Questionnaire vs Weeks of Sensing")
    ax.grid(alpha=0.3)

    ax.scatter([], [], color=C_PERS, marker="o", s=60, label="Questionnaire")
    ax.scatter([], [], color=C_SENS, marker="s", s=60, label="Passive Sensing")
    ax.legend(frameon=True)

    fig.savefig(OUT / "figS1_cost_effectiveness.png")
    plt.close()
    print("  Saved figS1_cost_effectiveness.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# FIG S2: Idiographic R² Distribution (Supplement)
# ═══════════════════════════════════════════════════════════════════════
def figS2_idiographic():
    """Histogram of person-specific R² values."""
    df = pd.read_csv(ROB / "idiographic_models.csv")

    fig, ax = plt.subplots(figsize=(7.5, 4))

    r2 = df["R2_person"].dropna()
    r2_clipped = r2.clip(-2, 1)

    ax.hist(r2_clipped, bins=40, color=C_GREY, edgecolor="white", alpha=0.8, zorder=3)

    ax.axvline(0, color="#2c3e50", lw=1.5, ls="--", label="R² = 0", zorder=4)
    ax.axvline(0.3, color=C_PERS, lw=1.5, ls="--", label="R² = 0.3", zorder=4)

    n_above_0 = (r2 > 0).sum()
    n_above_03 = (r2 > 0.3).sum()
    n_total = len(r2)
    ax.text(0.97, 0.95,
            f"{n_above_03}/{n_total} ({100*n_above_03/n_total:.0f}%) with R² > 0.3\n"
            f"{n_above_0}/{n_total} ({100*n_above_0/n_total:.0f}%) with R² > 0",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_PERS, alpha=0.9))

    ax.set_xlabel("Person-Specific R² (Idiographic Models)")
    ax.set_ylabel("Count")
    ax.set_title("Individual Differences in Sensing Predictive Value")
    ax.legend(fontsize=9, frameon=True)
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(OUT / "figS2_idiographic.png")
    plt.close()
    print("  Saved figS2_idiographic.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
ALL_FIGS = {
    "fig1": fig1_study_design,
    "fig2": fig2_grand_synthesis,
    "fig3": fig3_meta_forest,
    "fig4": fig4_shap_heatmap,
    "fig5": fig5_deep_learning,
    "fig6": fig6_dose_response,
    "fig7": fig7_temporal_reliability,
    "fig8": fig8_nns,
    "figS1": figS1_cost,
    "figS2": figS2_idiographic,
}

if __name__ == "__main__":
    setup_style()

    targets = sys.argv[1:] if len(sys.argv) > 1 else list(ALL_FIGS.keys())

    print(f"Generating {len(targets)} figures...", flush=True)
    for name in targets:
        if name in ALL_FIGS:
            print(f"\n  [{name}]", flush=True)
            ALL_FIGS[name]()
        else:
            print(f"  Unknown figure: {name}", flush=True)

    print(f"\n✓ Done. Figures saved to {OUT}/", flush=True)
