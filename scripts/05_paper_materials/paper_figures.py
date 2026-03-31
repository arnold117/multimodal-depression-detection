#!/usr/bin/env python3
"""
Paper Figures v2 — Publication-Ready Figures for JMIR Mental Health
====================================================================
12 figures (9 main + 2 supplement) with Nature/Lancet-grade styling.

Style A (Nature panel): Multi-panel data figures with (a)(b) labels
Style B (Infographic): Visual diagrams with icons, gradients, hierarchy
Style C (Global): Tableau-inspired palette, refined typography, grid alignment

Usage:
    python scripts/05_paper_materials/paper_figures.py           # all
    python scripts/05_paper_materials/paper_figures.py fig3      # specific
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
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
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
# Color Palette — Tableau-inspired, muted, professional
# ═══════════════════════════════════════════════════════════════════════
PAL = {
    "pers":     "#4E79A7",  # steel blue — personality
    "pers_lt":  "#A0CBE8",  # light blue
    "sens":     "#E15759",  # muted red — sensing
    "sens_lt":  "#FFBCBC",  # light red
    "comb":     "#59A14F",  # green — combined
    "comb_lt":  "#B6D7A8",  # light green
    "grey":     "#BAB0AC",  # warm grey
    "grey_dk":  "#79706E",  # dark grey
    "purple":   "#B07AA1",  # muted purple — DL/MOMENT
    "orange":   "#F28E2B",  # warm orange — GradientBoosting
    "teal":     "#76B7B2",  # teal
    "gold":     "#EDC948",  # gold accent
    "bg":       "#FAFAFA",  # figure background
    "text":     "#2D2D2D",  # near-black text
    "text_lt":  "#6B6B6B",  # secondary text
    "grid":     "#E8E8E8",  # grid lines
}


def setup_style():
    """Configure matplotlib for publication — Nature/Lancet grade."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelcolor": PAL["text"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color": PAL["grey_dk"],
        "ytick.color": PAL["grey_dk"],
        "legend.fontsize": 8,
        "legend.framealpha": 0.95,
        "legend.edgecolor": PAL["grid"],
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.6,
        "axes.edgecolor": PAL["grey"],
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "grid.color": PAL["grid"],
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.8,
    })


def panel_label(ax, label, x=-0.08, y=1.06):
    """Add Nature-style panel label (a), (b), etc."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=PAL["text"], va="top", ha="left")


# ═══════════════════════════════════════════════════════════════════════
# FIG 1: Study Design (Infographic style)
# ═══════════════════════════════════════════════════════════════════════
def fig1_study_design():
    fig = plt.figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Subtle gradient background
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", cmap=LinearSegmentedColormap.from_list(
        "", ["#F8F9FA", "#EDF2F7"]), extent=[0, 10, 0, 7], zorder=0)

    studies = [
        {"name": "Study 1", "sub": "StudentLife", "uni": "Dartmouth, 2013",
         "n": "N = 28", "pers": "BFI-44\n5 minutes",
         "sensing": "Smartphone\n13 modalities\n87 features\n10 weeks",
         "outcomes": "PHQ-9, PSS\nLoneliness, GPA"},
        {"name": "Study 2", "sub": "NetHealth", "uni": "Notre Dame, 2015–19",
         "n": "N = 722", "pers": "BFI-44\n5 minutes",
         "sensing": "Fitbit + Comm\n28 features\n4 years",
         "outcomes": "CES-D, STAI, BAI\nLoneliness, SE, GPA"},
        {"name": "Study 3", "sub": "GLOBEM", "uni": "U. Washington, 2018–21",
         "n": "N = 809", "pers": "BFI-10\n1 minute",
         "sensing": "Fitbit + Phone + GPS\n19–2,597 features\n1 year",
         "outcomes": "BDI-II, STAI, PSS\nCESD, UCLA"},
    ]

    # Title
    ax.text(5, 6.65, "Can Passive Sensing Replace Questionnaires?",
            ha="center", fontsize=15, fontweight="bold", color=PAL["text"])
    ax.text(5, 6.25, "Three-Study Head-to-Head Comparison (N = 1,559)",
            ha="center", fontsize=10, color=PAL["text_lt"])

    for i, s in enumerate(studies):
        cx = 1.7 + i * 3.0

        # Study card background
        card = FancyBboxPatch((cx - 1.35, 0.15), 2.7, 5.7,
                               boxstyle="round,pad=0.15", facecolor="white",
                               edgecolor=PAL["grid"], lw=0.8, alpha=0.9, zorder=1)
        ax.add_patch(card)

        # Header
        ax.text(cx, 5.55, s["name"], ha="center", fontsize=11,
                fontweight="bold", color=PAL["text"], zorder=2)
        ax.text(cx, 5.2, s["sub"], ha="center", fontsize=9,
                color=PAL["text_lt"], style="italic", zorder=2)
        ax.text(cx, 4.85, s["uni"], ha="center", fontsize=7.5,
                color=PAL["text_lt"], zorder=2)

        # N badge
        badge = FancyBboxPatch((cx - 0.55, 4.35), 1.1, 0.35,
                                boxstyle="round,pad=0.08", facecolor=PAL["pers"],
                                edgecolor="none", alpha=0.9, zorder=2)
        ax.add_patch(badge)
        ax.text(cx, 4.52, s["n"], ha="center", va="center", fontsize=9,
                fontweight="bold", color="white", zorder=3)

        # Personality box
        p_box = FancyBboxPatch((cx - 1.15, 3.35), 2.3, 0.8,
                                boxstyle="round,pad=0.08", facecolor=PAL["pers_lt"],
                                edgecolor=PAL["pers"], lw=0.8, alpha=0.6, zorder=2)
        ax.add_patch(p_box)
        ax.text(cx - 1.0, 3.95, "QUESTIONNAIRE", fontsize=6, fontweight="bold",
                color=PAL["pers"], zorder=3)
        ax.text(cx, 3.65, s["pers"], ha="center", fontsize=8,
                color=PAL["text"], zorder=3)

        # Sensing box
        s_box = FancyBboxPatch((cx - 1.15, 1.85), 2.3, 1.3,
                                boxstyle="round,pad=0.08", facecolor=PAL["sens_lt"],
                                edgecolor=PAL["sens"], lw=0.8, alpha=0.4, zorder=2)
        ax.add_patch(s_box)
        ax.text(cx - 1.0, 2.95, "PASSIVE SENSING", fontsize=6, fontweight="bold",
                color=PAL["sens"], zorder=3)
        ax.text(cx, 2.45, s["sensing"], ha="center", fontsize=7.5,
                color=PAL["text"], zorder=3)

        # Arrow
        ax.annotate("", xy=(cx, 1.1), xytext=(cx, 1.75),
                    arrowprops=dict(arrowstyle="-|>", color=PAL["grey_dk"],
                                    lw=1.2, mutation_scale=12), zorder=3)

        # Outcomes
        ax.text(cx, 0.85, "OUTCOMES", ha="center", fontsize=6,
                fontweight="bold", color=PAL["grey_dk"], zorder=3)
        ax.text(cx, 0.5, s["outcomes"], ha="center", fontsize=7.5,
                color=PAL["text"], zorder=3)

    fig.savefig(OUT / "fig1_study_design.png")
    plt.close()
    print("  fig1_study_design.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# FIG 2: Analysis Pipeline (Infographic style)
# ═══════════════════════════════════════════════════════════════════════
def fig2_pipeline():
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    steps = [
        ("3 Datasets\nN = 1,559", PAL["teal"]),
        ("Feature\nExtraction\n87–2,597 features", PAL["gold"]),
        ("ML Models\nRidge, GB, CNN\nMOMENT", PAL["orange"]),
        ("Head-to-Head\nPersonality\nvs Sensing", PAL["pers"]),
        ("44 Robustness\nChecks", PAL["sens"]),
    ]

    for i, (label, color) in enumerate(steps):
        cx = 1.0 + i * 2.0
        box = FancyBboxPatch((cx - 0.75, 0.6), 1.5, 1.6,
                              boxstyle="round,pad=0.12", facecolor=color,
                              edgecolor="white", lw=2, alpha=0.85, zorder=2)
        ax.add_patch(box)
        ax.text(cx, 1.4, label, ha="center", va="center", fontsize=8,
                fontweight="bold", color="white", zorder=3)

        # Step number
        ax.text(cx, 2.45, f"Step {i+1}", ha="center", fontsize=7,
                color=PAL["text_lt"], fontweight="bold", zorder=3)

        # Arrow between steps
        if i < len(steps) - 1:
            ax.annotate("", xy=(cx + 1.0, 1.4), xytext=(cx + 0.8, 1.4),
                        arrowprops=dict(arrowstyle="-|>", color=PAL["grey_dk"],
                                        lw=1.5, mutation_scale=14), zorder=3)

    # Title
    ax.text(5, 2.85, "Analysis Pipeline", ha="center", fontsize=12,
            fontweight="bold", color=PAL["text"])

    fig.savefig(OUT / "fig2_pipeline.png")
    plt.close()
    print("  fig2_pipeline.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# FIG 3: Core Finding — (a) Grand Synthesis + (b) Meta Forest
# ═══════════════════════════════════════════════════════════════════════
def fig3_core():
    fig = plt.figure(figsize=(7.5, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1], hspace=0.35)

    # — Panel (a): Grand Synthesis —
    ax_a = fig.add_subplot(gs[0])
    panel_label(ax_a, "a")

    df = pd.read_csv(CORE / "grand_synthesis.csv")
    df = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(df))
    h = 0.35

    ax_a.barh(y + h/2, df["R2_personality"], h, color=PAL["pers"],
              edgecolor="white", lw=0.3, label="Personality", zorder=3)
    ax_a.barh(y - h/2, df["R2_sensing"], h, color=PAL["sens"],
              edgecolor="white", lw=0.3, label="Sensing", zorder=3)

    ax_a.errorbar(df["R2_personality"], y + h/2,
                  xerr=[df["R2_personality"] - df["R2_pers_ci_lo"],
                        df["R2_pers_ci_hi"] - df["R2_personality"]],
                  fmt="none", color=PAL["pers"], capsize=2, lw=0.6, zorder=4)
    ax_a.errorbar(df["R2_sensing"], y - h/2,
                  xerr=[df["R2_sensing"] - df["R2_sens_ci_lo"],
                        df["R2_sens_ci_hi"] - df["R2_sensing"]],
                  fmt="none", color=PAL["sens"], capsize=2, lw=0.6, zorder=4)

    labels = [f"{row.Study} — {row.Outcome}" for _, row in df.iterrows()]
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels, fontsize=8)

    prev = None
    for i, row in df.iterrows():
        if prev and row.Study != prev:
            ax_a.axhline(i - 0.5, color=PAL["grid"], lw=0.6, ls="--", zorder=1)
        prev = row.Study

    ax_a.axvline(0, color=PAL["text"], lw=0.8, zorder=2)
    ax_a.set_xlim(-1.5, 1.0)

    n_wins = df["Pers_wins"].sum()
    ax_a.text(0.02, 0.02, f"Personality wins {n_wins}/{len(df)} ({100*n_wins//len(df)}%)",
              transform=ax_a.transAxes, ha="left", va="bottom", fontsize=10,
              fontweight="bold", color=PAL["pers"],
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PAL["pers"], alpha=0.9))

    ax_a.set_xlabel("Cross-Validated R²")
    ax_a.set_title("Personality vs Sensing Across 3 Studies and 15 Outcomes", fontsize=10)
    ax_a.legend(loc="upper right", frameon=True)
    ax_a.grid(axis="x", alpha=0.3)

    # — Panel (b): Meta Forest —
    ax_b = fig.add_subplot(gs[1])
    panel_label(ax_b, "b")

    meta = pd.read_csv(CORE / "meta_analysis.csv")
    y_m = np.arange(len(meta))
    r_vals = pd.to_numeric(meta["pooled_r"], errors="coerce")
    ci_lo = pd.to_numeric(meta["ci_lo"], errors="coerce")
    ci_hi = pd.to_numeric(meta["ci_hi"], errors="coerce")
    colors_m = [PAL["pers"] if r > 0 else PAL["sens"] for r in r_vals]

    ax_b.scatter(r_vals, y_m, color=colors_m, s=70, zorder=3, edgecolors="white", lw=0.5)
    ax_b.hlines(y_m, ci_lo, ci_hi, colors=colors_m, lw=2, zorder=2)

    ax_b.set_yticks(y_m)
    labels_m = [l.replace("→", "$\\rightarrow$") for l in meta["Label"]]
    ax_b.set_yticklabels(labels_m, fontsize=8)
    ax_b.axvline(0, color=PAL["text"], lw=0.8, ls="--")
    ax_b.set_xlabel("Pooled r [95% CI]")
    ax_b.set_title("Meta-Analytic Personality–Outcome Correlations", fontsize=10)
    ax_b.grid(axis="x", alpha=0.3)

    fig.savefig(OUT / "fig3_core.png")
    plt.close()
    print("  fig3_core.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# FIG 4: Mechanism — (a) SHAP Heatmap + (b) DL Baseline
# ═══════════════════════════════════════════════════════════════════════
def fig4_mechanism():
    fig = plt.figure(figsize=(7.5, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2], hspace=0.35)

    # — Panel (a): SHAP —
    ax_a = fig.add_subplot(gs[0])
    panel_label(ax_a, "a")

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
        ridge = df[df["Model"].str.contains("Ridge", case=False)]
        if len(ridge) == 0:
            ridge = df.iloc[:1]
        vals = [pd.to_numeric(ridge[t].values[0], errors="coerce") if t in ridge.columns else 0
                for t in traits]
        data[label] = vals

    mat = pd.DataFrame(data, index=trait_labels).T

    annot = mat.copy().astype(str)
    for i in range(mat.shape[0]):
        ranks = mat.iloc[i].rank(ascending=False).astype(int)
        for j in range(mat.shape[1]):
            v = mat.iloc[i, j]
            star = " *" if ranks.iloc[j] == 1 else ""
            annot.iloc[i, j] = f"{v:.1f}{star}"

    cmap = LinearSegmentedColormap.from_list("", ["#FFF5EB", "#FD8D3C", "#A63603"])
    sns.heatmap(mat, annot=annot, fmt="", cmap=cmap, ax=ax_a,
                linewidths=0.8, linecolor="white", cbar_kws={"label": "Mean |SHAP|", "shrink": 0.8})
    ax_a.set_title("SHAP Feature Importance (* = rank #1)", fontsize=10)

    # — Panel (b): DL Baseline —
    ax_b = fig.add_subplot(gs[1])
    panel_label(ax_b, "b")

    dl = pd.read_csv(ROB / "deep_learning_comparison.csv")
    model_order = ["Personality (Ridge)", "Sensing PCA (Ridge)",
                   "GradientBoosting+Optuna", "1D-CNN", "MOMENT → Ridge"]
    model_colors = {
        "Personality (Ridge)": PAL["pers"],
        "Sensing PCA (Ridge)": PAL["grey"],
        "GradientBoosting+Optuna": PAL["orange"],
        "1D-CNN": PAL["sens"],
        "MOMENT → Ridge": PAL["purple"],
    }
    model_short = {
        "Personality (Ridge)": "Personality",
        "Sensing PCA (Ridge)": "Sensing PCA",
        "GradientBoosting+Optuna": "GB+Optuna",
        "1D-CNN": "1D-CNN",
        "MOMENT → Ridge": "MOMENT",
    }

    # Combine S3 + S2 outcomes
    all_outcomes = []
    for study in ["S3", "S2"]:
        sub = dl[dl["Study"] == study]
        for o in sub["Outcome"].unique():
            all_outcomes.append(f"{study}\n{o}")

    x = np.arange(len(all_outcomes))
    width = 0.15

    for i, model in enumerate(model_order):
        vals = []
        for study in ["S3", "S2"]:
            sub = dl[dl["Study"] == study]
            for o in sub["Outcome"].unique():
                row = sub[(sub["Outcome"] == o) & (sub["Model"] == model)]
                vals.append(row["R2_mean"].values[0] if len(row) > 0 else 0)
        ax_b.bar(x + i * width, vals, width, label=model_short[model],
                 color=model_colors[model], edgecolor="white", lw=0.3, zorder=3)

    ax_b.set_xticks(x + width * 2)
    ax_b.set_xticklabels(all_outcomes, fontsize=7)
    ax_b.set_ylabel("Cross-Validated R²")
    ax_b.axhline(0, color=PAL["text"], lw=0.8, zorder=2)
    ax_b.set_ylim(-0.25, None)
    ax_b.grid(axis="y", alpha=0.3)
    ax_b.set_title("Deep Learning Cannot Rescue Passive Sensing", fontsize=10)
    ax_b.legend(fontsize=7, ncol=3, loc="upper right", frameon=True)

    ax_b.annotate("MOMENT R² = −1.0 to −1.7 (off scale)",
                  xy=(0.97, 0.03), xycoords="axes fraction",
                  ha="right", va="bottom", fontsize=7, color=PAL["purple"],
                  fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PAL["purple"], alpha=0.9))

    fig.savefig(OUT / "fig4_mechanism.png")
    plt.close()
    print("  fig4_mechanism.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# FIG 5: Not data, not reliability — (a) Dose-Response + (b) ICC
# ═══════════════════════════════════════════════════════════════════════
def fig5_robustness():
    fig = plt.figure(figsize=(7.5, 7))
    gs = gridspec.GridSpec(2, 1, hspace=0.35)

    # — Panel (a): Dose-Response —
    ax_a = fig.add_subplot(gs[0])
    panel_label(ax_a, "a")

    dr = pd.read_csv(ROB / "dose_response.csv")
    outcomes = dr["Outcome"].unique()
    cmap_lines = plt.cm.Set2

    pers_vals = dr.groupby("Outcome")["R2_personality"].mean()
    ax_a.axhspan(pers_vals.min(), pers_vals.max(), color=PAL["pers"], alpha=0.08, zorder=1)
    ax_a.axhline(pers_vals.mean(), color=PAL["pers"], ls="--", lw=1.2, zorder=2,
                 label=f"Personality R² ({pers_vals.min():.2f}–{pers_vals.max():.2f})")

    for i, o in enumerate(outcomes):
        sub = dr[dr["Outcome"] == o].sort_values("N_days")
        ax_a.plot(sub["N_days"], sub["R2_behavior"], "o-", color=cmap_lines(i),
                  label=f"{o}", markersize=4, lw=1.5, zorder=3)

    ax_a.axhline(0, color=PAL["text"], lw=0.5, ls=":", zorder=1)
    ax_a.set_xlabel("Days of Sensing Data")
    ax_a.set_ylabel("Cross-Validated R²")
    ax_a.set_title("More Data Does Not Help: Sensing R² Flat Across Duration", fontsize=10)
    ax_a.set_xticks(dr["N_days"].unique())
    ax_a.legend(fontsize=7, ncol=3, frameon=True)
    ax_a.grid(alpha=0.3)

    # — Panel (b): Temporal ICC —
    ax_b = fig.add_subplot(gs[1])
    panel_label(ax_b, "b")

    icc = pd.read_csv(ROB / "temporal_reliability.csv")
    features = icc["Feature"].unique()
    cmap_icc = plt.cm.Set2

    y_nudges = {"Steps": 0.025, "Sleep Duration": 0, "Sleep Efficiency": 0,
                "Screen Time": -0.025, "Calls (Incoming)": 0.005, "Home Time": 0}

    for i, feat in enumerate(features):
        sub = icc[icc["Feature"] == feat].sort_values("Window_days")
        color = cmap_icc(i / len(features))
        ax_b.plot(sub["Window_days"], sub["ICC"], "o-", color=color,
                  markersize=5, lw=1.8, zorder=3)
        ax_b.fill_between(sub["Window_days"], sub["ICC_CI_lo"], sub["ICC_CI_hi"],
                          alpha=0.10, color=color)
        last = sub.iloc[-1]
        nudge = y_nudges.get(feat, 0)
        ax_b.text(last["Window_days"] + 1.5, last["ICC"] + nudge, feat,
                  fontsize=7, color=color, va="center")

    ax_b.axhspan(0.75, 0.85, color=PAL["pers"], alpha=0.10, zorder=1)
    ax_b.axhline(0.85, color=PAL["pers"], ls="--", lw=1.2, zorder=2)
    ax_b.axhline(0.75, color=PAL["pers"], ls=":", lw=1.2, zorder=2)
    ax_b.text(5, 0.86, "BFI-44 retest r = .85", fontsize=7, color=PAL["pers"], va="bottom")
    ax_b.text(5, 0.74, "BFI-10 retest r = .75", fontsize=7, color=PAL["pers"], va="top")

    ax_b.annotate("Sensing is STABLE but predicts nothing",
                  xy=(35, 0.58), fontsize=9, fontweight="bold", color=PAL["text"],
                  ha="center",
                  bbox=dict(boxstyle="round,pad=0.4", fc=PAL["gold"], ec=PAL["orange"],
                            alpha=0.4))

    ax_b.set_xlabel("Window Size (days)")
    ax_b.set_ylabel("ICC(3,k)")
    ax_b.set_title("Sensing Temporal Reliability vs Personality Test-Retest", fontsize=10)
    ax_b.set_xticks([7, 14, 30, 45, 60])
    ax_b.set_xlim(3, 85)
    ax_b.set_ylim(0.55, 1.02)
    ax_b.grid(alpha=0.3)

    fig.savefig(OUT / "fig5_robustness.png")
    plt.close()
    print("  fig5_robustness.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# FIG 6: Clinical — (a) NNS + (b) Cost-Effectiveness
# ═══════════════════════════════════════════════════════════════════════
def fig6_clinical():
    fig = plt.figure(figsize=(7.5, 7))
    gs = gridspec.GridSpec(2, 1, hspace=0.35)

    # — Panel (a): NNS —
    ax_a = fig.add_subplot(gs[0])
    panel_label(ax_a, "a")

    nns = pd.read_csv(ROB / "nns_comparison.csv")
    feat_order = ["Pers-only", "Pers+Beh", "Beh-only"]
    feat_colors = {"Pers-only": PAL["pers"], "Pers+Beh": PAL["comb"], "Beh-only": PAL["grey"]}
    outcomes = nns["Outcome"].unique()
    x = np.arange(len(outcomes))
    w = 0.25

    for i, fs in enumerate(feat_order):
        vals = []
        for o in outcomes:
            row = nns[(nns["Outcome"] == o) & (nns["Features"] == fs)]
            vals.append(row["TP_per_100"].values[0] if len(row) > 0 else 0)
        ax_a.bar(x + i * w, vals, w, label=fs, color=feat_colors[fs],
                 edgecolor="white", lw=0.3, zorder=3)
        for xi, v in zip(x + i * w, vals):
            if v > 0:
                ax_a.text(xi, v + 0.3, f"{v:.0f}", ha="center", va="bottom",
                          fontsize=6.5, color=feat_colors[fs], fontweight="bold")

    ax_a.set_xticks(x + w)
    ax_a.set_xticklabels(outcomes, fontsize=8, rotation=12, ha="right")
    ax_a.set_ylabel("True Positives per 100 Screened")
    ax_a.set_title("Screening Yield by Feature Set", fontsize=10)
    ax_a.legend(fontsize=8, frameon=True)
    ax_a.grid(axis="y", alpha=0.3)

    # — Panel (b): Cost —
    ax_b = fig.add_subplot(gs[1])
    panel_label(ax_b, "b")

    cost = pd.read_csv(CORE / "cost_effectiveness.csv")
    offsets = {
        "Full Big Five (5 min)": (8, -18),
        "Neuroticism score (1 min)": (8, -15),
        "Pers + Sensing PCA": (5, 8),
        "Pers + Comm (S2 best)": (5, -15),
    }
    for _, row in cost.iterrows():
        is_p = "BFI" in row["Approach"] or "Neuroticism" in row["Approach"] or "personality" in row["Approach"].lower()
        c = PAL["pers"] if is_p else PAL["sens"]
        m = "o" if is_p else "s"
        ax_b.scatter(row["Time_min"], row["R2"], color=c, marker=m,
                     s=80, zorder=3, edgecolors="white", lw=0.5)
        default = (8, 5) if row["R2"] > 0 else (8, -12)
        off = offsets.get(row["Approach"], default)
        ax_b.annotate(row["Approach"], (row["Time_min"], row["R2"]),
                      textcoords="offset points", xytext=off, fontsize=6.5, color=c)

    ax_b.axhline(0, color=PAL["text"], lw=0.5, ls=":")
    ax_b.set_xscale("log")
    ax_b.set_xlabel("Assessment Time (minutes, log scale)")
    ax_b.set_ylabel("Cross-Validated R²")
    ax_b.set_title("10 Seconds and $0 vs Weeks and $100+", fontsize=10)
    ax_b.grid(alpha=0.3)
    ax_b.scatter([], [], color=PAL["pers"], marker="o", s=50, label="Questionnaire")
    ax_b.scatter([], [], color=PAL["sens"], marker="s", s=50, label="Passive Sensing")
    ax_b.legend(frameon=True, fontsize=8)

    fig.savefig(OUT / "fig6_clinical.png")
    plt.close()
    print("  fig6_clinical.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# FIG 7: Raw Data Example — (a) Daily sensing ts + (b) BFI questionnaire
# ═══════════════════════════════════════════════════════════════════════
def fig7_raw_data():
    fig = plt.figure(figsize=(7.5, 6))
    gs = gridspec.GridSpec(2, 1, hspace=0.4)

    # — Panel (a): One person's daily sensing —
    ax_a = fig.add_subplot(gs[0])
    panel_label(ax_a, "a")

    # Load one person's steps data
    steps = pd.read_csv("data/raw/globem/INS-W_1/FeatureData/steps.csv", low_memory=False)
    col = [c for c in steps.columns if "avgsumsteps" in c][0]
    pid = steps["pid"].unique()[0]
    person = steps[steps["pid"] == pid].copy()
    person["date"] = pd.to_datetime(person["date"])
    person[col] = pd.to_numeric(person[col], errors="coerce")
    person = person.dropna(subset=[col]).sort_values("date").head(90)

    days = np.arange(len(person))
    vals = person[col].values

    ax_a.fill_between(days, 0, vals, color=PAL["sens_lt"], alpha=0.5, zorder=2)
    ax_a.plot(days, vals, color=PAL["sens"], lw=1.0, zorder=3)
    ax_a.set_xlabel("Day of Study")
    ax_a.set_ylabel("Average Daily Steps")
    ax_a.set_title(f"90 Days of Continuous Passive Sensing (One Participant)", fontsize=10)
    ax_a.grid(alpha=0.3)

    ax_a.annotate("Weeks of data collection\nResult: R² ≈ 0",
                  xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
                  fontsize=8, fontweight="bold", color=PAL["sens"],
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PAL["sens"], alpha=0.9))

    # — Panel (b): BFI questionnaire visual —
    ax_b = fig.add_subplot(gs[1])
    panel_label(ax_b, "b")
    ax_b.axis("off")

    # Simulate a BFI-10 questionnaire look
    items = [
        ("I see myself as someone who...", None),
        ("1. ...is reserved", "E (R)"),
        ("2. ...is generally trusting", "A"),
        ("3. ...tends to be lazy", "C (R)"),
        ("4. ...is relaxed, handles stress well", "N (R)"),
        ("5. ...has few artistic interests", "O (R)"),
        ("6. ...is outgoing, sociable", "E"),
        ("7. ...tends to find fault with others", "A (R)"),
        ("8. ...does a thorough job", "C"),
        ("9. ...gets nervous easily", "N"),
        ("10. ...has an active imagination", "O"),
    ]

    # Background card
    card = FancyBboxPatch((0.05, 0.02), 0.9, 0.96, boxstyle="round,pad=0.02",
                           facecolor="#F7F9FC", edgecolor=PAL["pers"], lw=1.5,
                           transform=ax_b.transAxes, zorder=1)
    ax_b.add_patch(card)

    ax_b.text(0.5, 0.95, "Big Five Inventory — 10 Items (BFI-10)", fontsize=10,
              fontweight="bold", color=PAL["pers"], ha="center", va="top",
              transform=ax_b.transAxes, zorder=2)
    ax_b.text(0.5, 0.88, "Completion time: ~1 minute  |  Cost: $0  |  R² = 0.09–0.52",
              fontsize=8, color=PAL["text_lt"], ha="center", va="top",
              transform=ax_b.transAxes, zorder=2)

    for i, (text, trait) in enumerate(items):
        y_pos = 0.80 - i * 0.07
        if trait is None:
            ax_b.text(0.1, y_pos, text, fontsize=8, style="italic",
                      color=PAL["text"], transform=ax_b.transAxes, zorder=2)
        else:
            ax_b.text(0.1, y_pos, text, fontsize=7.5, color=PAL["text"],
                      transform=ax_b.transAxes, zorder=2)
            ax_b.text(0.88, y_pos, trait, fontsize=7, color=PAL["pers"],
                      fontweight="bold", transform=ax_b.transAxes, zorder=2, ha="right")
            # Likert dots
            for j in range(5):
                ax_b.plot(0.62 + j * 0.04, y_pos, "o", color=PAL["grid"],
                          markersize=4, transform=ax_b.transAxes, zorder=2)

    fig.savefig(OUT / "fig7_raw_data.png")
    plt.close()
    print("  fig7_raw_data.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# FIG 8: Cross-Study — (a) Correlation matrices + (b) Replication
# ═══════════════════════════════════════════════════════════════════════
def fig8_cross_study():
    fig = plt.figure(figsize=(7.5, 7))
    gs = gridspec.GridSpec(2, 1, hspace=0.4)

    # — Panel (a): Correlation heatmap per study —
    ax_a = fig.add_subplot(gs[0])
    panel_label(ax_a, "a")

    corr = pd.read_csv(CORE / "three_study_mh_correlations.csv")
    traits = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
    trait_short = ["E", "A", "C", "N", "O"]
    constructs = corr["Construct"].unique()

    # Build matrix: rows = constructs, cols = trait × study
    cols = []
    data_rows = []
    for _, row in corr.iterrows():
        r_row = []
        for s_col in ["S1_r", "S2_r", "S3_r"]:
            r_row.append(pd.to_numeric(row[s_col], errors="coerce"))
        data_rows.append(r_row)

    # Reshape: each row is one trait-construct pair, columns are S1/S2/S3
    labels = [f"{row['Trait'][:3].title()}→{row['Construct'][:4]}" for _, row in corr.iterrows()]
    mat = pd.DataFrame(data_rows, index=labels, columns=["S1", "S2", "S3"])

    # Select top associations (|r| > 0.15 in at least one study)
    mask = mat.abs().max(axis=1) > 0.15
    mat_filt = mat[mask]

    cmap = LinearSegmentedColormap.from_list("", [PAL["sens"], "white", PAL["pers"]])
    sns.heatmap(mat_filt, annot=True, fmt=".2f", cmap=cmap, center=0,
                ax=ax_a, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "r", "shrink": 0.8}, vmin=-0.6, vmax=0.6)
    ax_a.set_title("Personality–Outcome Correlations: Consistent Across Studies", fontsize=10)

    # — Panel (b): Replication summary —
    ax_b = fig.add_subplot(gs[1])
    panel_label(ax_b, "b")

    rep = pd.read_csv(CORE / "three_study_replication_summary.csv")
    n_consistent = rep["direction_consistent"].sum()
    n_total = len(rep)

    # Bar chart of consistent vs inconsistent
    findings = rep["Finding"].values
    consistent = rep["direction_consistent"].astype(int).values
    y_r = np.arange(len(findings))
    colors_r = [PAL["comb"] if c else PAL["sens"] for c in consistent]

    ax_b.barh(y_r, consistent, color=colors_r, edgecolor="white", lw=0.3, height=0.6, zorder=3)
    ax_b.set_yticks(y_r)
    ax_b.set_yticklabels(findings, fontsize=7)
    ax_b.set_xlabel("Direction Consistent")
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["No", "Yes"])
    ax_b.set_title(f"Replication: {n_consistent}/{n_total} Findings Consistent Across Studies",
                   fontsize=10)
    ax_b.grid(axis="x", alpha=0.3)

    fig.savefig(OUT / "fig8_cross_study.png")
    plt.close()
    print("  fig8_cross_study.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# FIG 9: Methods — (a) Calibration + (b) Modality Ablation
# ═══════════════════════════════════════════════════════════════════════
def fig9_methods():
    fig = plt.figure(figsize=(7.5, 7))
    gs = gridspec.GridSpec(2, 1, hspace=0.35)

    # — Panel (a): Calibration (Brier + ECE) —
    ax_a = fig.add_subplot(gs[0])
    panel_label(ax_a, "a")

    cal = pd.read_csv(ROB / "clinical_expanded.csv")
    feat_colors = {"Pers-only": PAL["pers"], "Pers+Beh": PAL["comb"], "Beh-only": PAL["grey"]}

    outcomes_cal = cal["Outcome"].unique()
    x = np.arange(len(outcomes_cal))
    w = 0.25

    for i, fs in enumerate(["Pers-only", "Pers+Beh", "Beh-only"]):
        vals = []
        for o in outcomes_cal:
            row = cal[(cal["Outcome"] == o) & (cal["Features"] == fs)]
            vals.append(row["Brier_mean"].values[0] if len(row) > 0 else np.nan)
        ax_a.bar(x + i * w, vals, w, label=fs, color=feat_colors.get(fs, PAL["grey"]),
                 edgecolor="white", lw=0.3, zorder=3)

    ax_a.set_xticks(x + w)
    ax_a.set_xticklabels(outcomes_cal, fontsize=7, rotation=15, ha="right")
    ax_a.set_ylabel("Brier Score (lower = better)")
    ax_a.set_title("Model Calibration: Personality Models Better Calibrated", fontsize=10)
    ax_a.legend(fontsize=7, frameon=True)
    ax_a.grid(axis="y", alpha=0.3)

    # — Panel (b): Modality Ablation —
    ax_b = fig.add_subplot(gs[1])
    panel_label(ax_b, "b")

    abl = pd.read_csv(ROB / "modality_ablation.csv")
    # Average Delta_R2 by modality across outcomes
    avg = abl.groupby("Modality")["Delta_R2"].mean().sort_values(ascending=True)

    colors_abl = [PAL["comb"] if v > 0 else PAL["sens"] for v in avg.values]
    ax_b.barh(np.arange(len(avg)), avg.values, color=colors_abl,
              edgecolor="white", lw=0.3, height=0.6, zorder=3)
    ax_b.set_yticks(np.arange(len(avg)))
    ax_b.set_yticklabels(avg.index, fontsize=8)
    ax_b.axvline(0, color=PAL["text"], lw=0.8, zorder=2)
    ax_b.set_xlabel("ΔR² (adding modality to personality)")
    ax_b.set_title("No Sensing Modality Adds Meaningful Value", fontsize=10)
    ax_b.grid(axis="x", alpha=0.3)

    fig.savefig(OUT / "fig9_methods.png")
    plt.close()
    print("  fig9_methods.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# FIG S1: Idiographic (Supplement)
# ═══════════════════════════════════════════════════════════════════════
def figS1_idiographic():
    fig, ax = plt.subplots(figsize=(7.5, 4))

    df = pd.read_csv(ROB / "idiographic_models.csv")
    r2 = df["R2_person"].dropna().clip(-2, 1)

    ax.hist(r2, bins=40, color=PAL["grey"], edgecolor="white", alpha=0.85, zorder=3)
    ax.axvline(0, color=PAL["text"], lw=1.5, ls="--", label="R² = 0", zorder=4)
    ax.axvline(0.3, color=PAL["pers"], lw=1.5, ls="--", label="R² = 0.3", zorder=4)

    n0 = (df["R2_person"] > 0).sum()
    n03 = (df["R2_person"] > 0.3).sum()
    n = len(df["R2_person"].dropna())
    ax.text(0.97, 0.95,
            f"{n03}/{n} ({100*n03/n:.0f}%) with R² > 0.3\n"
            f"{n0}/{n} ({100*n0/n:.0f}%) with R² > 0",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PAL["pers"], alpha=0.9))

    ax.set_xlabel("Person-Specific R² (Idiographic Models)")
    ax.set_ylabel("Count")
    ax.set_title("Sensing Works for Some Individuals", fontsize=10)
    ax.legend(fontsize=8, frameon=True)
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(OUT / "figS1_idiographic.png")
    plt.close()
    print("  figS1_idiographic.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# FIG S2: Demographics (Supplement)
# ═══════════════════════════════════════════════════════════════════════
def figS2_demographics():
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.5), sharey=True)

    study_files = [
        ("S1 (N=28)", BY_STUDY / "s1/tables/descriptive_stats.csv"),
        ("S2 (N=722)", BY_STUDY / "s2/tables/table5_descriptive.csv"),
        ("S3 (N=809)", BY_STUDY / "s3/tables/table10_descriptive.csv"),
    ]
    traits = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
    trait_short = ["E", "A", "C", "N", "O"]

    for ax, (title, fpath) in zip(axes, study_files):
        df = pd.read_csv(fpath)
        # Get trait means and SDs
        means = []
        sds = []
        for t in traits:
            row = df[df.iloc[:, 0].str.lower() == t]
            if len(row) > 0:
                means.append(float(row["Mean"].values[0]))
                sds.append(float(row["SD"].values[0]))
            else:
                means.append(np.nan)
                sds.append(np.nan)

        x = np.arange(len(trait_short))
        ax.bar(x, means, color=PAL["pers"], edgecolor="white", lw=0.3, alpha=0.8, zorder=3)
        ax.errorbar(x, means, yerr=sds, fmt="none", color=PAL["text"], capsize=3, lw=0.8, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(trait_short, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 5)

    axes[0].set_ylabel("Mean (± SD)")
    fig.suptitle("Big Five Trait Distributions Across Studies", fontsize=11, fontweight="bold")
    plt.tight_layout()

    fig.savefig(OUT / "figS2_demographics.png")
    plt.close()
    print("  figS2_demographics.png", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
ALL_FIGS = {
    "fig1": fig1_study_design,
    "fig2": fig2_pipeline,
    "fig3": fig3_core,
    "fig4": fig4_mechanism,
    "fig5": fig5_robustness,
    "fig6": fig6_clinical,
    "fig7": fig7_raw_data,
    "fig8": fig8_cross_study,
    "fig9": fig9_methods,
    "figS1": figS1_idiographic,
    "figS2": figS2_demographics,
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
            print(f"  Unknown: {name}", flush=True)
    print(f"\n✓ Done. {OUT}/", flush=True)
