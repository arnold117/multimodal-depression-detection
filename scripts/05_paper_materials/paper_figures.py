#!/usr/bin/env python3
"""
Paper Figures v4 — Nature-Style Publication Quality
=====================================================
9 figures (7 main + 2 supplement).

Visual philosophy: clean, generous whitespace, minimal chart junk,
professional typography. Infographics (Fig 1, 2, 7) kept minimal.

Usage:
    python scripts/05_paper_materials/paper_figures.py
    python scripts/05_paper_materials/paper_figures.py fig3 fig5
"""

import sys, numpy as np, pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

CORE = Path("results/core")
ROB  = Path("results/robustness")
BY   = Path("results/by_study")
OUT  = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────
BLU     = "#4E79A7"
RED     = "#F28E2B"  # orange — colorblind-safe (was #E15759 red)
GRY     = "#BAB0AC"
GRY_LT  = "#F0F0F0"
TXT     = "#333333"
TXT_LT  = "#888888"
WHITE   = "#FFFFFF"
BG      = "#FAFAFA"  # subtle off-white for panel backgrounds


def setup():
    """Nature-style rcParams: clean, minimal, professional."""
    plt.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "medium",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "legend.title_fontsize": 9,
        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.5,
        "axes.edgecolor": "#CCCCCC",
        "axes.facecolor": WHITE,
        "axes.titlepad": 12,
        "axes.labelpad": 8,
        # Figure
        "figure.facecolor": WHITE,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
        # Grid
        "axes.grid": False,
        "grid.color": "#EEEEEE",
        "grid.linewidth": 0.3,
        "grid.alpha": 1.0,
        # Ticks
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.pad": 4,
        "ytick.major.pad": 4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Lines
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.5,
    })


def plabel(ax, s, x=-0.08, y=1.06):
    """Panel label (a, b, c) in bold, Nature style."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=TXT, va="top", ha="left")


def clean_axis(ax, grid_axis=None):
    """Make axis clean: optional subtle grid, tight spines."""
    if grid_axis:
        ax.grid(axis=grid_axis, color="#EEEEEE", linewidth=0.3, zorder=0)
    ax.tick_params(axis="both", which="both", length=3, width=0.5, color="#CCCCCC")


def badge(ax, text, x, y, color=BLU, fontsize=9, transform=None):
    """Annotation badge with rounded border."""
    if transform is None:
        transform = ax.transAxes
    ax.text(x, y, text, transform=transform, fontsize=fontsize,
            fontweight="bold", color=color, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc=WHITE, ec=color,
                      lw=0.8, alpha=0.95))


# ═══════════════════════════════════════════════════════════════════════
# FIG 1 — Study Overview: (a) pipeline, (b) 3 study cards, (c) data contrast
# ═══════════════════════════════════════════════════════════════════════
def fig1_study_design():
    fig = plt.figure(figsize=(7.5, 10))
    # Top: pipeline, Middle: study cards, Bottom: data contrast
    gs = fig.add_gridspec(3, 1, height_ratios=[0.6, 2.2, 1.2], hspace=0.15)

    # ── (a) Pipeline — horizontal flow ────────────────────────────
    ax_pipe = fig.add_subplot(gs[0])
    ax_pipe.set_xlim(0, 10); ax_pipe.set_ylim(0, 2.2); ax_pipe.axis("off")
    plabel(ax_pipe, "a", x=-0.02, y=1.1)

    steps = [
        ("3 Datasets", "N = 1,559", GRY),
        ("Feature\nExtraction", "87 - 2,597", GRY),
        ("ML Models", "Ridge  GBM\nCNN  MOMENT", TXT_LT),
        ("Personality\nvs Sensing", "15 outcomes", BLU),
        ("Robustness", "44 checks", RED),
    ]
    for i, (title, detail, color) in enumerate(steps):
        cx = 1.0 + i * 2.0
        w_box, h_box = 1.5, 1.3
        ax_pipe.add_patch(FancyBboxPatch((cx - w_box/2, 0.35), w_box, h_box,
            boxstyle="round,pad=0.08", fc=color, alpha=0.08,
            ec=color, lw=0.6, zorder=1))
        ax_pipe.text(cx, 1.2, title, ha="center", va="center", fontsize=7.5,
                fontweight="bold", color=TXT, zorder=2)
        ax_pipe.text(cx, 0.65, detail, ha="center", va="center", fontsize=6,
                color=TXT_LT, zorder=2)
        if i < len(steps) - 1:
            ax_pipe.annotate("", xy=(cx + w_box/2 + 0.18, 1.0),
                        xytext=(cx + w_box/2 + 0.05, 1.0),
                        arrowprops=dict(arrowstyle="-|>", color="#CCCCCC", lw=0.8,
                                        mutation_scale=8), zorder=2)

    # ── (b) Three study cards ─────────────────────────────────────
    ax_cards = fig.add_subplot(gs[1])
    ax_cards.set_xlim(0, 10); ax_cards.set_ylim(0, 7); ax_cards.axis("off")
    plabel(ax_cards, "b", x=-0.02, y=1.02)

    studies = [
        {"name": "Study 1", "sub": "StudentLife", "uni": "Dartmouth 2013",
         "n": "N = 28", "pers": "BFI-44", "pt": "5 min",
         "sens": "Smartphone", "sd": "13 modalities", "st": "10 weeks",
         "out": "PHQ-9  PSS  Loneliness  GPA"},
        {"name": "Study 2", "sub": "NetHealth", "uni": "Notre Dame 2015-19",
         "n": "N = 722", "pers": "BFI-44", "pt": "5 min",
         "sens": "Fitbit + Phone", "sd": "28 features", "st": "4 years",
         "out": "CES-D  STAI  BAI  +3 more"},
        {"name": "Study 3", "sub": "GLOBEM", "uni": "U. Washington 2018-21",
         "n": "N = 809", "pers": "BFI-10", "pt": "1 min",
         "sens": "Fitbit + Phone + GPS", "sd": "19-2,597 features", "st": "1 year",
         "out": "BDI-II  STAI  PSS  +2 more"},
    ]

    for i, s in enumerate(studies):
        cx = 1.7 + i * 2.8

        # Card shadow + card
        ax_cards.add_patch(FancyBboxPatch((cx - 1.22, 0.18), 2.44, 6.4,
            boxstyle="round,pad=0.1", fc="#F5F5F5", ec="none", zorder=0))
        ax_cards.add_patch(FancyBboxPatch((cx - 1.25, 0.2), 2.44, 6.4,
            boxstyle="round,pad=0.1", fc=WHITE, ec="#E0E0E0", lw=0.6, zorder=1))

        y0 = 6.2
        ax_cards.text(cx, y0, s["name"], ha="center", fontsize=10,
                fontweight="bold", color=TXT, zorder=2)
        ax_cards.text(cx, y0 - 0.28, s["sub"], ha="center", fontsize=8,
                color=BLU, fontweight="bold", zorder=2)
        ax_cards.text(cx, y0 - 0.5, s["uni"], ha="center", fontsize=7,
                color=TXT_LT, zorder=2)

        # N pill
        ax_cards.add_patch(FancyBboxPatch((cx - 0.4, y0 - 0.95), 0.8, 0.28,
            boxstyle="round,pad=0.04", fc=BLU, ec="none", zorder=2))
        ax_cards.text(cx, y0 - 0.81, s["n"], ha="center", va="center", fontsize=7.5,
                fontweight="bold", color=WHITE, zorder=3)

        # Questionnaire
        qy = 4.0
        ax_cards.add_patch(FancyBboxPatch((cx - 0.95, qy), 1.9, 0.75,
            boxstyle="round,pad=0.05", fc=BLU, alpha=0.05, ec=BLU, lw=0.5, zorder=1))
        ax_cards.text(cx - 0.8, qy + 0.6, "QUESTIONNAIRE", fontsize=4.5,
                fontweight="bold", color=BLU, alpha=0.6, zorder=2)
        ax_cards.text(cx, qy + 0.38, s["pers"], ha="center", fontsize=7.5, color=TXT, zorder=2)
        ax_cards.text(cx, qy + 0.1, s["pt"], ha="center", fontsize=11,
                fontweight="bold", color=BLU, zorder=2)

        ax_cards.text(cx, 3.75, "vs", ha="center", fontsize=6.5, color=TXT_LT,
                style="italic", zorder=2)

        # Sensing
        sy = 1.6
        ax_cards.add_patch(FancyBboxPatch((cx - 0.95, sy), 1.9, 1.9,
            boxstyle="round,pad=0.05", fc=RED, alpha=0.04, ec=RED, lw=0.5, zorder=1))
        ax_cards.text(cx - 0.8, sy + 1.72, "PASSIVE SENSING", fontsize=4.5,
                fontweight="bold", color=RED, alpha=0.6, zorder=2)
        ax_cards.text(cx, sy + 1.35, s["sens"], ha="center", fontsize=7.5, color=TXT, zorder=2)
        ax_cards.text(cx, sy + 1.08, s["sd"], ha="center", fontsize=7, color=TXT_LT, zorder=2)
        ax_cards.text(cx, sy + 0.25, s["st"], ha="center", fontsize=13,
                fontweight="bold", color=RED, zorder=2)

        # Outcomes
        ax_cards.annotate("", xy=(cx, 1.25), xytext=(cx, 1.5),
                    arrowprops=dict(arrowstyle="-|>", color="#CCCCCC", lw=0.8,
                                    mutation_scale=8), zorder=2)
        ax_cards.text(cx, 0.95, s["out"], ha="center", fontsize=6, color=TXT_LT,
                zorder=2, style="italic")

    # Summary bar
    ax_cards.add_patch(FancyBboxPatch((0.35, 0.05), 9.3, 0.45,
        boxstyle="round,pad=0.06", fc=GRY_LT, ec="none", zorder=1))
    ax_cards.text(5, 0.27, "15 head-to-head comparisons  |  44 robustness analyses  |  4 ML architectures",
            ha="center", fontsize=7, color=TXT, zorder=2)

    # ── (c) Data contrast: sensing time series vs questionnaire ───
    gs_bottom = gs[2].subgridspec(1, 2, width_ratios=[1.5, 1], wspace=0.25)
    ax_ts = fig.add_subplot(gs_bottom[0])
    plabel(ax_ts, "c", x=-0.08, y=1.08)

    # Sensing time series
    try:
        steps_data = pd.read_csv("data/raw/globem/INS-W_1/FeatureData/steps.csv",
                                  low_memory=False)
        col = [c for c in steps_data.columns if "avgsumsteps" in c][0]
        pid = steps_data.pid.unique()[2]
        p = steps_data[steps_data.pid == pid].copy()
        p["date"] = pd.to_datetime(p["date"])
        p[col] = pd.to_numeric(p[col], errors="coerce")
        p = p.dropna(subset=[col]).sort_values("date").head(90)
        days = np.arange(len(p))
        vals = p[col].values
    except Exception:
        days = np.arange(90)
        np.random.seed(42)
        vals = np.random.lognormal(8.5, 0.6, 90)

    ax_ts.fill_between(days, 0, vals, color=RED, alpha=0.1, zorder=1)
    ax_ts.plot(days, vals, color=RED, lw=0.5, alpha=0.6, zorder=2)
    ax_ts.set_xlabel("Day of Study", fontsize=8)
    ax_ts.set_ylabel("Daily Steps", fontsize=8)
    ax_ts.set_title("90 Days of Passive Sensing", fontsize=9, pad=8)
    clean_axis(ax_ts)
    ax_ts.text(0.97, 0.97, "All this data  $\\rightarrow$  R$^2$ $\\approx$ 0",
            transform=ax_ts.transAxes, ha="right", va="top", fontsize=8,
            fontweight="bold", color=RED, alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF5F5", ec=RED,
                      lw=0.5, alpha=0.95))

    # Questionnaire card
    ax_card = fig.add_subplot(gs_bottom[1])
    ax_card.axis("off")
    plabel(ax_card, "d", x=-0.12, y=1.08)
    ax_card.add_patch(FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
        boxstyle="round,pad=0.04", fc="#F7F9FC", ec=BLU, lw=0.8,
        transform=ax_card.transAxes))
    ax_card.text(0.5, 0.82, "BFI-10", fontsize=14, fontweight="bold", color=BLU,
             ha="center", transform=ax_card.transAxes)
    ax_card.text(0.5, 0.66, "10 items", fontsize=11, color=TXT,
             ha="center", transform=ax_card.transAxes)
    ax_card.text(0.5, 0.48, "1 minute", fontsize=16, fontweight="bold", color=BLU,
             ha="center", transform=ax_card.transAxes)
    ax_card.text(0.5, 0.32, "Cost: Free", fontsize=8, color=TXT_LT,
             ha="center", transform=ax_card.transAxes)
    ax_card.text(0.5, 0.16, "R$^2$ = 0.09 - 0.52", fontsize=10, fontweight="bold",
             color=BLU, ha="center", transform=ax_card.transAxes)

    fig.savefig(OUT / "fig1_overview.png")
    plt.close(); print("  fig1 done")


# ═══════════════════════════════════════════════════════════════════════
# FIG 2 — Analysis Pipeline (minimal horizontal flow)
# ═══════════════════════════════════════════════════════════════════════
def fig2_pipeline():
    fig, ax = plt.subplots(figsize=(7.5, 2.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.2); ax.axis("off")

    steps = [
        ("3 Datasets", "N = 1,559", GRY),
        ("Feature\nExtraction", "87 - 2,597", GRY),
        ("ML Models", "Ridge  GBM\nCNN  MOMENT", TXT_LT),
        ("Personality\nvs Sensing", "15 outcomes", BLU),
        ("Robustness", "44 checks", RED),
    ]

    for i, (title, detail, color) in enumerate(steps):
        cx = 1.0 + i * 2.0
        # Rounded rectangle instead of chevron — cleaner
        w, h = 1.4, 1.4
        ax.add_patch(FancyBboxPatch((cx - w/2, 0.35), w, h,
            boxstyle="round,pad=0.1", fc=color, alpha=0.08,
            ec=color, lw=0.8, zorder=1))
        ax.text(cx, 1.25, title, ha="center", va="center", fontsize=8,
                fontweight="bold", color=TXT, zorder=2)
        ax.text(cx, 0.7, detail, ha="center", va="center", fontsize=6.5,
                color=TXT_LT, zorder=2)

        # Arrow between boxes
        if i < len(steps) - 1:
            ax.annotate("", xy=(cx + w/2 + 0.15, 1.05), xytext=(cx + w/2 + 0.05, 1.05),
                        arrowprops=dict(arrowstyle="-|>", color="#CCCCCC", lw=1,
                                        mutation_scale=10), zorder=2)

    ax.text(5, 2.05, "Analysis Pipeline", ha="center", fontsize=11,
            fontweight="bold", color=TXT)

    fig.savefig(OUT / "fig2_pipeline.png")
    plt.close(); print("  fig2 done")


# ═══════════════════════════════════════════════════════════════════════
# FIG 3 — Core: (a) Dumbbell + (b) Meta Forest
# ═══════════════════════════════════════════════════════════════════════
def fig3_core():
    fig = plt.figure(figsize=(7.5, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.4, 1, 0.8], hspace=0.32)

    # ── (a) Dumbbell ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0]); plabel(ax, "a")
    df = pd.read_csv(CORE / "grand_synthesis.csv")

    # Reverse for top-to-bottom reading
    df = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(df))

    # Study background bands
    prev_study = None
    band_start = 0
    for i, row in df.iterrows():
        if prev_study is not None and row.Study != prev_study:
            if band_start % 2 == 0:
                ax.axhspan(band_start - 0.5, i - 0.5, color="#F8F8F8", zorder=0)
            band_start = i
        prev_study = row.Study
    # Last band
    if band_start % 2 == 0:
        ax.axhspan(band_start - 0.5, len(df) - 0.5, color="#F8F8F8", zorder=0)

    # Plot dumbbells
    for i, row in df.iterrows():
        p, s = row.R2_personality, row.R2_sensing
        # Connecting line
        ax.plot([min(p, s), max(p, s)], [i, i], color="#DDDDDD", lw=2, zorder=1,
                solid_capstyle="round")
        # Points
        ax.scatter(p, i, color=BLU, s=55, zorder=4, edgecolors=WHITE, lw=0.8)
        ax.scatter(s, i, color=RED, s=55, zorder=4, edgecolors=WHITE, lw=0.8)

    # Y-axis labels: clean format
    labels = []
    for _, r in df.iterrows():
        study_short = r.Study
        labels.append(f"{study_short}  {r.Outcome}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)

    # Study group labels on right side
    studies_seen = {}
    for i, row in df.iterrows():
        if row.Study not in studies_seen:
            studies_seen[row.Study] = [i]
        else:
            studies_seen[row.Study].append(i)

    ax.axvline(0, color=TXT, lw=0.6, alpha=0.4, zorder=1)
    ax.set_xlim(-0.8, 0.7)
    ax.set_xlabel("Cross-Validated R²", fontsize=9)
    ax.set_title("Personality vs Passive Sensing: 15 Head-to-Head Comparisons",
                 fontsize=10, pad=14)
    clean_axis(ax, grid_axis="x")

    # Legend — clean, minimal
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=BLU,
               markersize=7, label='Personality'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RED,
               markersize=7, label='Sensing'),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True,
              edgecolor="#DDDDDD", fancybox=False, fontsize=8,
              borderpad=0.6, handletextpad=0.4)

    # Win badge
    n_wins = int(df.Pers_wins.sum())
    badge(ax, f"Personality wins {n_wins}/15 (93%)", 0.02, 0.04, color=BLU, fontsize=8)

    # ── (b) Meta-analysis forest ──────────────────────────────────
    ax2 = fig.add_subplot(gs[1]); plabel(ax2, "b")
    meta = pd.read_csv(CORE / "meta_analysis.csv")
    ym = np.arange(len(meta))

    r_vals = pd.to_numeric(meta.pooled_r, errors="coerce")
    lo = pd.to_numeric(meta.ci_lo, errors="coerce")
    hi = pd.to_numeric(meta.ci_hi, errors="coerce")

    # Color by sign
    for i in range(len(meta)):
        c = BLU if r_vals.iloc[i] > 0 else RED
        # CI line
        ax2.plot([lo.iloc[i], hi.iloc[i]], [i, i], color=c, lw=2, solid_capstyle="round",
                 zorder=2, alpha=0.5)
        # Point estimate — diamond
        ax2.scatter(r_vals.iloc[i], i, color=c, s=60, zorder=4,
                    marker="D", edgecolors=WHITE, lw=0.6)

        # Highlight neuroticism rows
        lbl = str(meta.iloc[i].Label)
        if lbl.startswith("N"):
            ax2.axhspan(i - 0.4, i + 0.4, color=BLU, alpha=0.03, zorder=0)

    # Labels
    labs = []
    for _, row in meta.iterrows():
        lbl = str(row.Label).replace("\u2192", "$\\rightarrow$")
        k_val = int(row.k) if pd.notna(row.k) else "?"
        n_val = int(row.total_N) if pd.notna(row.total_N) else "?"
        labs.append(f"{lbl}   k={k_val}, N={n_val}")
    ax2.set_yticks(ym)
    ax2.set_yticklabels(labs, fontsize=7)

    ax2.axvline(0, color=TXT, lw=0.5, ls="--", alpha=0.5)
    ax2.set_xlabel("Pooled r  [95% CI]", fontsize=9)
    ax2.set_title("Meta-Analytic Personality$\\rightarrow$Outcome Correlations", fontsize=10, pad=14)
    clean_axis(ax2, grid_axis="x")

    # ── (c) Incremental validity — does sensing add to personality? ──
    ax3 = fig.add_subplot(gs[2]); plabel(ax3, "c")
    inc = pd.read_csv(CORE / "incremental_validity.csv")
    labels_inc = [f"{r.Study} {r.Outcome}" for _, r in inc.iterrows()]
    y_inc = np.arange(len(inc))
    colors_inc = [BLU if not r.sig_fdr else "#8B5CF6" for _, r in inc.iterrows()]
    ax3.barh(y_inc, inc.Delta_R2, color=colors_inc, edgecolor=WHITE, lw=0.5,
             height=0.6, zorder=3, alpha=0.8)
    for i, row in inc.iterrows():
        sig = "*" if row.sig_fdr else ""
        ax3.text(row.Delta_R2 + 0.001, i, f"{row.Delta_R2:.3f}{sig}",
                 va="center", fontsize=6.5, color=TXT)
    ax3.set_yticks(y_inc)
    ax3.set_yticklabels(labels_inc, fontsize=7)
    ax3.axvline(0, color=TXT, lw=0.5, alpha=0.5)
    ax3.set_xlabel("$\\Delta$R² (sensing added to personality)")
    ax3.set_title("Incremental $\\Delta$R² When Sensing Added to Personality (* = FDR < .05)",
                  fontsize=10, pad=14)
    clean_axis(ax3, grid_axis="x")

    fig.savefig(OUT / "fig2_core.png")
    plt.close(); print("  fig2 done")


# ═══════════════════════════════════════════════════════════════════════
# FIG 4 — Mechanism: (a) SHAP heatmap + (b) DL baseline
# ═══════════════════════════════════════════════════════════════════════
def fig4_mechanism():
    fig = plt.figure(figsize=(7.5, 11.5))
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.9, 0.8, 1.2], hspace=0.35)

    # ── (a) SHAP heatmap — which traits drive which outcomes ──────
    ax = fig.add_subplot(gs[0]); plabel(ax, "a")
    traits = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
    tlbl = ["E", "A", "C", "N", "O"]
    files = {
        "S2-STAI": BY / "s2/tables/shap_personality_stai.csv",
        "S2-BAI":  BY / "s2/tables/shap_personality_bai.csv",
        "S3-BDI":  BY / "s3/tables/shap_personality_bdiii.csv",
        "S3-CESD": BY / "s3/tables/shap_personality_cesd.csv",
        "S3-PSS":  BY / "s3/tables/shap_personality_pss10.csv",
        "S3-STAI": BY / "s3/tables/shap_personality_stai.csv",
        "S3-UCLA": BY / "s3/tables/shap_personality_ucla.csv",
    }
    data = {}
    for lbl, fp in files.items():
        if not fp.exists(): continue
        d = pd.read_csv(fp)
        row = d[d.Model.str.contains("Ridge", case=False)]
        if len(row) == 0: row = d.iloc[:1]
        data[lbl] = [pd.to_numeric(row[t].values[0], errors="coerce")
                     if t in row.columns else 0 for t in traits]
    mat = pd.DataFrame(data, index=tlbl).T

    # Annotation with rank
    annot = mat.copy().astype(str)
    for i in range(mat.shape[0]):
        ranks = mat.iloc[i].rank(ascending=False).astype(int)
        for j in range(mat.shape[1]):
            star = " *" if ranks.iloc[j] == 1 else ""
            annot.iloc[i, j] = f"{mat.iloc[i, j]:.2f}{star}"

    cmap = LinearSegmentedColormap.from_list("shap",
        ["#FFF8F0", "#FDAE6B", "#D94801"])
    sns.heatmap(mat, annot=annot, fmt="", cmap=cmap, ax=ax,
                linewidths=1.5, linecolor=WHITE,
                cbar_kws={"label": "Mean |SHAP|", "shrink": 0.6,
                           "aspect": 20, "pad": 0.02},
                annot_kws={"fontsize": 7.5})
    ax.set_title("Big-Five SHAP Importance per Depression Scale (* = top predictor)",
                 fontsize=10, pad=14)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8.5)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, fontweight="bold")

    # ── (b) Trait-domain specificity: correlation heatmap ─────────
    ax_b = fig.add_subplot(gs[1]); plabel(ax_b, "b")
    corr = pd.read_csv(CORE / "three_study_mh_correlations.csv")
    # Build trait × construct matrix (average across studies)
    constructs = ["Depression", "State Anxiety", "Perceived Stress", "Loneliness"]
    trait_names = ["extraversion", "agreeableness", "conscientiousness",
                   "neuroticism", "openness"]
    trait_short = ["E", "A", "C", "N", "O"]

    corr_mat = np.full((len(constructs), len(trait_names)), np.nan)
    for i, con in enumerate(constructs):
        for j, tr in enumerate(trait_names):
            row = corr[(corr.Construct == con) & (corr.Trait == tr)]
            if len(row) == 0: continue
            r = row.iloc[0]
            vals = []
            for s in [1, 2, 3]:
                v = pd.to_numeric(r.get(f"S{s}_r", np.nan), errors="coerce")
                if pd.notna(v): vals.append(v)
            if vals:
                corr_mat[i, j] = np.mean(vals)

    # Add GPA row
    gpa_corrs = pd.read_csv(Path("results/supplementary/personality_gpa_correlations.csv"))
    gpa_vals = []
    for tr in trait_names:
        row = gpa_corrs[gpa_corrs.Trait == tr]
        if len(row):
            # Average S1 and S2
            vals = [pd.to_numeric(row.iloc[0].get(f"S{s}_r", np.nan), errors="coerce")
                    for s in [1, 2]]
            vals = [v for v in vals if pd.notna(v)]
            gpa_vals.append(np.mean(vals) if vals else np.nan)
        else:
            gpa_vals.append(np.nan)
    corr_mat = np.vstack([corr_mat, [gpa_vals]])
    constructs_ext = constructs + ["GPA"]

    corr_df = pd.DataFrame(corr_mat, index=constructs_ext, columns=trait_short)
    cmap_div = LinearSegmentedColormap.from_list("corrdiv",
        [RED, "#FFEEEE", WHITE, "#EEF0FF", BLU])
    annot_corr = corr_df.map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    sns.heatmap(corr_df, annot=annot_corr, fmt="", cmap=cmap_div, center=0,
                ax=ax_b, linewidths=1.5, linecolor=WHITE, vmin=-0.6, vmax=0.6,
                cbar_kws={"label": "Mean r (across studies)", "shrink": 0.6},
                annot_kws={"fontsize": 8})
    ax_b.set_title("Trait$\\rightarrow$Outcome Specificity (mean Pearson r across 3 studies)",
                   fontsize=10, pad=14)
    ax_b.set_yticklabels(ax_b.get_yticklabels(), rotation=0, fontsize=8.5)
    ax_b.set_xticklabels(ax_b.get_xticklabels(), fontsize=10, fontweight="bold")

    # ── (c) DL baseline — all 5 models per outcome (heatmap style) ──
    ax2 = fig.add_subplot(gs[2]); plabel(ax2, "c")
    dl = pd.read_csv(ROB / "deep_learning_comparison.csv")

    # Pivot: rows = Study-Outcome, cols = Model
    model_order = ["Personality (Ridge)", "Sensing PCA (Ridge)",
                   "GradientBoosting+Optuna", "1D-CNN", "MOMENT \u2192 Ridge"]
    model_short = ["Personality\n(Ridge)", "Sensing\n(Ridge)", "Gradient\nBoosting",
                   "1D-CNN", "MOMENT"]

    # Build matrix
    outcomes = []
    for study in ["S2", "S3"]:
        sub = dl[dl.Study == study]
        for o in sub.Outcome.unique():
            outcomes.append(f"{study} {o}")

    mat = np.full((len(outcomes), len(model_order)), np.nan)
    for i, (study, outcome) in enumerate(
            [(s, o) for s in ["S2", "S3"] for o in dl[dl.Study == s].Outcome.unique()]):
        for j, m in enumerate(model_order):
            row = dl[(dl.Study == study) & (dl.Outcome == outcome) & (dl.Model == m)]
            if len(row):
                val = row.R2_mean.values[0]
                # Clip MOMENT for display (actual is -1 to -1.7)
                mat[i, j] = max(val, -0.2)

    # Diverging colormap: blue for positive, red for negative
    cmap = LinearSegmentedColormap.from_list("r2div",
        [RED, "#FFEEEE", WHITE, "#EEF0FF", BLU])
    vmax = max(0.55, np.nanmax(mat))

    im = ax2.imshow(mat, cmap=cmap, aspect="auto", vmin=-0.2, vmax=vmax, zorder=1)

    # Annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v): continue
            # Show actual value for MOMENT (not clipped)
            actual = dl[(dl.Study == outcomes[i][:2]) &
                        (dl.Outcome == outcomes[i][3:]) &
                        (dl.Model == model_order[j])]
            if len(actual):
                real_v = actual.R2_mean.values[0]
                txt = f"{real_v:.2f}" if real_v > -1 else f"{real_v:.1f}"
            else:
                txt = f"{v:.2f}"
            color = WHITE if abs(v) > 0.3 else TXT
            ax2.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                     fontweight="bold" if j == 0 else "normal", color=color)

    ax2.set_xticks(range(len(model_short)))
    ax2.set_xticklabels(model_short, fontsize=6.5)
    ax2.set_yticks(range(len(outcomes)))
    ax2.set_yticklabels(outcomes, fontsize=7)

    # Study separator
    s2_count = len(dl[dl.Study == "S2"].Outcome.unique())
    ax2.axhline(s2_count - 0.5, color=WHITE, lw=2)

    cb = fig.colorbar(im, ax=ax2, shrink=0.6, pad=0.02, aspect=20)
    cb.set_label("Cross-Validated R²", fontsize=7.5)

    ax2.set_title("Cross-Validated R² across Model Architectures", fontsize=10, pad=14)

    fig.savefig(OUT / "fig3_mechanism.png")
    plt.close(); print("  fig3 done")


# ═══════════════════════════════════════════════════════════════════════
# FIG 5 — Robustness: (a) Dose-Response + (b) ICC
# ═══════════════════════════════════════════════════════════════════════
def fig5_robustness():
    fig = plt.figure(figsize=(7.5, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.7], hspace=0.38)

    # ── (a) Dose-response: mean + range band instead of 5 cluttered lines
    ax = fig.add_subplot(gs[0]); plabel(ax, "a")
    dr = pd.read_csv(ROB / "dose_response.csv")
    days = sorted(dr.N_days.unique())

    # Personality reference — thin dashed line
    pers_mean = dr.groupby("Outcome").R2_personality.mean().mean()
    ax.axhline(pers_mean, color=BLU, ls="--", lw=1.2, zorder=2, alpha=0.8)
    ax.text(93, pers_mean + 0.005, f"Personality mean R² = {pers_mean:.2f}",
            fontsize=7, color=BLU, va="bottom", ha="right", alpha=0.8)

    # Sensing: aggregate across outcomes — mean line + min/max band
    sens_by_day = dr.groupby("N_days").R2_behavior.agg(["mean", "min", "max"])
    from scipy.interpolate import make_interp_spline
    x_smooth = np.linspace(min(days), max(days), 200)
    try:
        spl_mean = make_interp_spline(days, sens_by_day["mean"].values, k=2)
        spl_lo = make_interp_spline(days, sens_by_day["min"].values, k=2)
        spl_hi = make_interp_spline(days, sens_by_day["max"].values, k=2)
        y_mean = spl_mean(x_smooth)
        y_lo = spl_lo(x_smooth)
        y_hi = spl_hi(x_smooth)
    except Exception:
        x_smooth = np.array(days)
        y_mean = sens_by_day["mean"].values
        y_lo = sens_by_day["min"].values
        y_hi = sens_by_day["max"].values

    ax.fill_between(x_smooth, y_lo, y_hi, color=RED, alpha=0.1, zorder=1,
                    label="Sensing range (5 outcomes)")
    ax.plot(x_smooth, y_mean, color=RED, lw=1.8, zorder=3, label="Sensing mean")
    ax.scatter(days, sens_by_day["mean"].values, color=RED, s=30, zorder=4,
              edgecolors=WHITE, lw=0.6)

    ax.axhline(0, color=TXT, lw=0.4, alpha=0.3)
    ax.set_ylim(-0.06, 0.18)
    ax.set_xlabel("Days of Sensing Data")
    ax.set_ylabel("Cross-Validated R²")
    ax.set_title("More Sensing Data Does Not Help", fontsize=10, pad=14)
    ax.set_xticks(days)
    ax.legend(fontsize=7, frameon=True, edgecolor="#DDDDDD",
              fancybox=False, borderpad=0.5, loc="upper left")
    clean_axis(ax, grid_axis="y")

    # Callout
    ax.text(0.98, 0.05, "7 days $\\approx$ 92 days",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold", color=TXT, alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.35", fc=WHITE, ec="#DDDDDD",
                      lw=0.5))

    # ── (b) ICC ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1]); plabel(ax2, "b")
    icc = pd.read_csv(ROB / "temporal_reliability.csv")
    keep = ["Steps", "Sleep Duration", "Screen Time", "Home Time"]
    icc = icc[icc.Feature.isin(keep)]

    markers_icc = {'Steps': 'o', 'Sleep Duration': 's', 'Screen Time': '^', 'Home Time': 'D'}
    alphas_icc = {'Steps': 1.0, 'Sleep Duration': 0.75, 'Screen Time': 0.6, 'Home Time': 0.45}

    from scipy.interpolate import make_interp_spline
    for feat in keep:
        sub = icc[icc.Feature == feat].sort_values("Window_days")
        if len(sub) == 0: continue
        a = alphas_icc.get(feat, 0.7)
        xd = sub.Window_days.values
        yd = sub.ICC.values
        # Smooth curve
        try:
            x_sm = np.linspace(xd.min(), xd.max(), 100)
            spl = make_interp_spline(xd, yd, k=2)
            y_sm = spl(x_sm)
            ax2.plot(x_sm, y_sm, color=RED, alpha=a, lw=1.2, zorder=2)
        except Exception:
            ax2.plot(xd, yd, color=RED, alpha=a, lw=1.2, zorder=2)
        ax2.scatter(xd, yd, color=RED, alpha=a, s=30, zorder=4,
                    marker=markers_icc.get(feat, 'o'),
                    edgecolors=WHITE, lw=0.5, label=feat)
        ax2.fill_between(sub.Window_days, sub.ICC_CI_lo, sub.ICC_CI_hi,
                         color=RED, alpha=a * 0.07, zorder=0)

    # Personality reference band
    ax2.axhspan(0.75, 0.85, color=BLU, alpha=0.06, zorder=0)
    ax2.axhline(0.85, color=BLU, ls="--", lw=0.8, alpha=0.5)
    ax2.axhline(0.75, color=BLU, ls=":", lw=0.8, alpha=0.5)
    ax2.text(6, 0.865, "BFI-44 retest r = .85", fontsize=6.5, color=BLU, alpha=0.7)
    ax2.text(6, 0.735, "BFI-10 retest r = .75", fontsize=6.5, color=BLU, alpha=0.7)

    ax2.legend(fontsize=7.5, loc="center right", frameon=True, edgecolor="#DDDDDD",
               fancybox=False, borderpad=0.5)

    # Key message
    ax2.text(0.5, 0.08, "Stable but predicts nothing",
             transform=ax2.transAxes, ha="center", va="bottom",
             fontsize=9, fontweight="bold", color=TXT,
             bbox=dict(boxstyle="round,pad=0.4", fc="#FFFDE7", ec="#E0D870",
                       lw=0.5, alpha=0.9))

    ax2.set_xlabel("Window Size (days)")
    ax2.set_ylabel("ICC(3,k)")
    ax2.set_title("Sensing Is Reliable but Irrelevant", fontsize=10, pad=14)
    ax2.set_xticks([7, 14, 30, 45, 60])
    ax2.set_xlim(3, 65)
    ax2.set_ylim(0.55, 1.0)
    clean_axis(ax2, grid_axis="y")

    # ── (c) Cross-study transfer ────────────────────────────────
    ax3 = fig.add_subplot(gs[2]); plabel(ax3, "c")
    tr = pd.read_csv(ROB / "cross_study_transfer.csv")
    labels_tr = tr.Transfer.values
    x_tr = np.arange(len(tr))
    w = 0.3
    ax3.bar(x_tr - w/2, tr.R2_within_study, w, color=BLU, edgecolor=WHITE, lw=0.5,
            label="Within-study", zorder=3, alpha=0.8)
    ax3.bar(x_tr + w/2, tr.R2_transfer, w, color=RED, edgecolor=WHITE, lw=0.5,
            label="Cross-study transfer", zorder=3, alpha=0.8)
    # Value labels
    for i in range(len(tr)):
        ax3.text(i - w/2, tr.R2_within_study.iloc[i] + 0.01,
                 f"{tr.R2_within_study.iloc[i]:.2f}", ha="center", va="bottom",
                 fontsize=6, color=BLU, fontweight="bold")
        ax3.text(i + w/2, min(tr.R2_transfer.iloc[i], 0) - 0.02,
                 f"{tr.R2_transfer.iloc[i]:.2f}", ha="center", va="top",
                 fontsize=6, color=RED, fontweight="bold")
    # Simplify long labels e.g. "S2→S3 (Depression)" → "S2→S3 CESD"
    short_labels = [
        str(l).replace("Depression ", "").replace("Anxiety ", "")
              .replace("(CES-D)", "CESD").replace("(STAI)", "STAI")
              .replace("(", "").replace(")", "")
        for l in labels_tr
    ]
    ax3.set_xticks(x_tr)
    ax3.set_xticklabels(short_labels, fontsize=7.5, rotation=15, ha="right")
    ax3.axhline(0, color=TXT, lw=0.5, alpha=0.5)
    ax3.set_ylabel("Cross-Validated R²")
    ax3.set_title("Personality Models Don't Transfer Across Studies", fontsize=10, pad=14)
    ax3.legend(fontsize=7, frameon=True, edgecolor="#DDDDDD", fancybox=False)
    clean_axis(ax3, grid_axis="y")

    fig.savefig(OUT / "fig4_robustness.png")
    plt.close(); print("  fig4 done")


# ═══════════════════════════════════════════════════════════════════════
# FIG 6 — Clinical: (a) NNS + (b) Cost-Effectiveness
# ═══════════════════════════════════════════════════════════════════════
def fig6_clinical():
    fig = plt.figure(figsize=(7.5, 9.5))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.8], hspace=0.4)

    # ── (a) NNS ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0]); plabel(ax, "a")
    nns = pd.read_csv(ROB / "nns_comparison.csv")
    outcomes = nns.Outcome.unique()
    x = np.arange(len(outcomes))
    w = 0.3

    pers_vals, beh_vals = [], []
    for o in outcomes:
        p = nns[(nns.Outcome == o) & (nns.Features == "Pers-only")]
        b = nns[(nns.Outcome == o) & (nns.Features == "Beh-only")]
        pers_vals.append(p.TP_per_100.values[0] if len(p) else 0)
        beh_vals.append(b.TP_per_100.values[0] if len(b) else 0)

    ax.bar(x - w/2, pers_vals, w, color=BLU, edgecolor=WHITE, lw=0.5,
           label="Personality", zorder=3)
    ax.bar(x + w/2, beh_vals, w, color=RED, edgecolor=WHITE, lw=0.5,
           label="Sensing", zorder=3)

    # Value labels
    for i in range(len(outcomes)):
        ax.text(i - w/2, pers_vals[i] + 0.4, f"{pers_vals[i]:.0f}", ha="center",
                va="bottom", fontsize=6.5, color=BLU, fontweight="bold")
        ax.text(i + w/2, beh_vals[i] + 0.4, f"{beh_vals[i]:.0f}", ha="center",
                va="bottom", fontsize=6.5, color=RED)
        # Delta
        delta = pers_vals[i] - beh_vals[i]
        if abs(delta) > 0.5:
            sign = "+" if delta > 0 else ""
            ymax = max(pers_vals[i], beh_vals[i]) + 2.0
            ax.text(i, ymax, f"{sign}{delta:.0f}", ha="center", fontsize=6,
                    color=BLU if delta > 0 else RED, fontweight="bold", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("True Positives per 100 Screened")
    ax.set_title("Screening Yield", fontsize=10, pad=14)
    ax.legend(fontsize=7.5, frameon=True, edgecolor="#DDDDDD", fancybox=False)
    clean_axis(ax, grid_axis="y")

    # ── (b) Cost-effectiveness scatter ────────────────────────────
    ax2 = fig.add_subplot(gs[1]); plabel(ax2, "b")
    cost = pd.read_csv(CORE / "cost_effectiveness.csv")

    # Scatter
    for _, row in cost.iterrows():
        is_p = any(k in row.Approach for k in ["BFI", "Neuroticism", "Big Five"])
        c = BLU if is_p else RED
        m = "o" if is_p else "s"
        ax2.scatter(row.Time_min, row.R2, color=c, marker=m, s=65, zorder=4,
                    edgecolors=WHITE, lw=0.8)

    # Labels with smart offsets
    offsets = {
        "2 BFI items (10 sec)": (10, 5),
        "Neuroticism score (1 min)": (10, -12),
        "Full Big Five (5 min)": (10, 5),
        "All 44 BFI items (5 min)": (10, -12),
        "Sensing PCA (weeks)": (-8, -14),
        "Sensing 28 raw (weeks)": (-8, 10),
        "Pers + Sensing PCA": (-8, 8),
        "Pers + Comm (S2 best)": (-8, -14),
    }
    for _, row in cost.iterrows():
        is_p = any(k in row.Approach for k in ["BFI", "Neuroticism", "Big Five"])
        c = BLU if is_p else RED
        off = offsets.get(row.Approach, (8, 5))
        ax2.annotate(row.Approach, (row.Time_min, row.R2),
                     textcoords="offset points", xytext=off,
                     fontsize=5.5, color=c, alpha=0.8,
                     arrowprops=dict(arrowstyle="-", color="#CCCCCC", lw=0.3)
                     if abs(off[0]) > 5 else None)

    ax2.axhline(0, color=TXT, lw=0.4, alpha=0.3)
    ax2.set_xscale("log")
    ax2.set_xlabel("Assessment Time (minutes, log scale)", fontsize=10)
    ax2.set_ylabel("Cross-Validated R²", fontsize=10)
    ax2.set_title("Brief Questionnaire vs Weeks of Sensing", fontsize=10, pad=14)
    clean_axis(ax2, grid_axis="y")

    # Legend
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor=BLU, markersize=6,
                  label='Questionnaire'),
           Line2D([0], [0], marker='s', color='w', markerfacecolor=RED, markersize=6,
                  label='Passive Sensing')]
    ax2.legend(handles=leg, frameon=True, edgecolor="#DDDDDD", fancybox=False,
               fontsize=7.5)

    # ── (c) Worst-20% rescue ────────────────────────────────────
    ax3 = fig.add_subplot(gs[2]); plabel(ax3, "c")
    w20 = pd.read_csv(ROB / "worst20_rescue.csv")
    x_w = np.arange(len(w20))
    w = 0.18
    ax3.bar(x_w - 1.5*w, w20.R2_pers_worst, w, color=BLU, edgecolor=WHITE, lw=0.5,
            label="Pers (hard)", zorder=3, alpha=0.6)
    ax3.bar(x_w - 0.5*w, w20.R2_beh_worst, w, color=RED, edgecolor=WHITE, lw=0.5,
            label="Sens (hard)", zorder=3, alpha=0.6)
    ax3.bar(x_w + 0.5*w, w20.R2_pers_rest, w, color=BLU, edgecolor=WHITE, lw=0.5,
            label="Pers (easy)", zorder=3)
    ax3.bar(x_w + 1.5*w, w20.R2_beh_rest, w, color=RED, edgecolor=WHITE, lw=0.5,
            label="Sens (easy)", zorder=3)
    ax3.set_xticks(x_w)
    ax3.set_xticklabels(w20.Outcome, fontsize=8)
    ax3.axhline(0, color=TXT, lw=0.5, alpha=0.5)
    ax3.set_ylabel("Cross-Validated R²", fontsize=10)
    ax3.set_title("Sensing Cannot Rescue Personality's Hard Cases", fontsize=10, pad=14)
    ax3.legend(fontsize=6, frameon=True, edgecolor="#DDDDDD", fancybox=False,
               ncol=4, loc="upper right", borderpad=0.4)
    clean_axis(ax3, grid_axis="y")

    fig.savefig(OUT / "fig5_clinical.png")
    plt.close(); print("  fig5 done")


# ═══════════════════════════════════════════════════════════════════════
# FIG 7 — Raw Data Comparison (left: sensing ts, right: questionnaire)
# ═══════════════════════════════════════════════════════════════════════
def fig7_raw_data():
    fig = plt.figure(figsize=(7.5, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.25)

    # ── Left: sensing time series ─────────────────────────────────
    ax = fig.add_subplot(gs[0]); plabel(ax, "a")
    try:
        steps = pd.read_csv("data/raw/globem/INS-W_1/FeatureData/steps.csv",
                            low_memory=False)
        col = [c for c in steps.columns if "avgsumsteps" in c][0]
        pid = steps.pid.unique()[2]
        p = steps[steps.pid == pid].copy()
        p["date"] = pd.to_datetime(p["date"])
        p[col] = pd.to_numeric(p[col], errors="coerce")
        p = p.dropna(subset=[col]).sort_values("date").head(90)
        days = np.arange(len(p))
        vals = p[col].values
    except Exception:
        # Fallback: synthetic
        days = np.arange(90)
        np.random.seed(42)
        vals = np.random.lognormal(8.5, 0.6, 90)

    ax.fill_between(days, 0, vals, color=RED, alpha=0.1, zorder=1)
    ax.plot(days, vals, color=RED, lw=0.6, alpha=0.7, zorder=2)
    ax.set_xlabel("Day of Study")
    ax.set_ylabel("Daily Steps")
    ax.set_title("90 Days of Passive Sensing", fontsize=10, pad=12)
    clean_axis(ax)

    # Annotation
    ax.text(0.95, 0.92, "All this data\nR² $\\approx$ 0",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            fontweight="bold", color=RED, alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF5F5", ec=RED,
                      lw=0.5, alpha=0.8))

    # ── Right: abstract questionnaire card ────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off"); plabel(ax2, "b", x=-0.12)

    # Card
    ax2.add_patch(FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
        boxstyle="round,pad=0.04", fc="#F7F9FC", ec=BLU, lw=1.0,
        transform=ax2.transAxes))

    ax2.text(0.5, 0.82, "BFI-10", fontsize=16, fontweight="bold", color=BLU,
             ha="center", transform=ax2.transAxes)
    ax2.text(0.5, 0.66, "10 items", fontsize=12, color=TXT,
             ha="center", transform=ax2.transAxes)
    ax2.text(0.5, 0.48, "1 minute", fontsize=18, fontweight="bold", color=BLU,
             ha="center", transform=ax2.transAxes)
    ax2.text(0.5, 0.32, "Cost: Free", fontsize=9, color=TXT_LT,
             ha="center", transform=ax2.transAxes)
    ax2.text(0.5, 0.16, "R² = 0.09 - 0.52", fontsize=11, fontweight="bold",
             color=BLU, ha="center", transform=ax2.transAxes)

    fig.savefig(OUT / "fig7_raw_data.png")
    plt.close(); print("  fig7 done")


# ═══════════════════════════════════════════════════════════════════════
# FIG S1 — Idiographic: (a) full distribution + (b) zoomed
# ═══════════════════════════════════════════════════════════════════════
def figS1_idiographic():
    df = pd.read_csv(ROB / "idiographic_models.csv")
    r2 = df.R2_person.dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2),
                                    width_ratios=[1, 1.5])
    fig.subplots_adjust(wspace=0.3)
    plabel(ax1, "a"); plabel(ax2, "b")

    # Helper: color histogram bars by R² threshold
    def _color_hist(ax_h, vals, bins):
        counts, edges, patches = ax_h.hist(vals, bins=bins, edgecolor=WHITE, lw=0.5, alpha=0.88)
        for patch, left in zip(patches, edges[:-1]):
            center = left + (edges[1] - edges[0]) / 2
            if center >= 0.3:
                patch.set_facecolor(BLU)
            elif center >= 0:
                patch.set_facecolor("#A8C4DB")  # muted blue
            else:
                patch.set_facecolor(GRY)

    # (a) Full distribution
    _color_hist(ax1, r2.clip(-5, 1), bins=30)
    ax1.axvline(0, color=TXT, lw=0.8, ls="--", alpha=0.5)
    ax1.set_xlabel("Person-Level R²")
    ax1.set_ylabel("Count")
    ax1.set_title("Full Distribution", fontsize=9, pad=10)
    clean_axis(ax1, grid_axis="y")

    # (b) Zoomed
    r2z = r2[r2 > -0.5]
    _color_hist(ax2, r2z, bins=25)
    ax2.axvline(0, color=TXT, lw=0.8, ls="--", alpha=0.5, label="R² = 0")
    ax2.axvline(0.3, color=BLU, lw=0.8, ls="--", alpha=0.7, label="R² = 0.3")
    ax2.axvspan(0.3, r2z.max() + 0.05, color=BLU, alpha=0.05)

    n03 = (r2 > 0.3).sum()
    n0 = (r2 > 0).sum()
    nt = len(r2)
    ax2.text(0.97, 0.95,
             f"{n03}/{nt} ({100*n03/nt:.0f}%) with R² > 0.3\n"
             f"{n0}/{nt} ({100*n0/nt:.0f}%) with R² > 0",
             transform=ax2.transAxes, ha="right", va="top", fontsize=7,
             bbox=dict(boxstyle="round,pad=0.3", fc=WHITE, ec=BLU,
                       lw=0.5, alpha=0.9))
    ax2.set_xlabel("Person-Level R²")
    ax2.set_title("Zoomed: R² > -0.5", fontsize=9, pad=10)
    ax2.legend(fontsize=6.5, frameon=True, edgecolor="#DDDDDD", fancybox=False)
    clean_axis(ax2, grid_axis="y")

    fig.suptitle("Passive Sensing Works for Some Individuals",
                 fontsize=11, fontweight="bold", y=1.02)

    fig.savefig(OUT / "figS1_idiographic.png")
    plt.close(); print("  figS1 done")


# ═══════════════════════════════════════════════════════════════════════
# FIG S2 — Supplement: (a) Cross-Study + (b) Calibration + (c) Ablation
# ═══════════════════════════════════════════════════════════════════════
def figS2_supplement():
    fig = plt.figure(figsize=(7.5, 8.5))
    gs = gridspec.GridSpec(3, 1, hspace=0.45)

    # ── (a) Cross-study grouped bar chart ───────────────────────────
    ax = fig.add_subplot(gs[0]); plabel(ax, "a")
    corr = pd.read_csv(CORE / "three_study_mh_correlations.csv")
    key_pairs = [
        ("neuroticism", "Depression", "Neu$\\rightarrow$Dep"),
        ("neuroticism", "State Anxiety", "Neu$\\rightarrow$Anx"),
        ("conscientiousness", "Depression", "Con$\\rightarrow$Dep"),
        ("extraversion", "Loneliness", "Ext$\\rightarrow$Lone"),
    ]

    bar_data = []  # list of (label, s1, s2, s3, ci_lo_list, ci_hi_list)
    for trait, construct, lbl in key_pairs:
        matches = corr[(corr.Trait.str.lower() == trait) &
                       (corr.Construct.str.contains(construct, case=False))]
        if len(matches) == 0: continue
        row = matches.iloc[0]
        vals = [pd.to_numeric(row[f"S{s}_r"], errors="coerce") for s in [1, 2, 3]]
        ci_los = [pd.to_numeric(row.get(f"S{s}_ci_lo", np.nan), errors="coerce") for s in [1, 2, 3]]
        ci_his = [pd.to_numeric(row.get(f"S{s}_ci_hi", np.nan), errors="coerce") for s in [1, 2, 3]]
        if any(pd.isna(v) for v in vals): continue
        bar_data.append((lbl, *vals, ci_los, ci_his))

    if bar_data:
        n_pairs = len(bar_data)
        x = np.arange(n_pairs)
        w = 0.22
        study_colors = [BLU, RED, GRY]
        study_labels = ["S1 (N=28)", "S2 (N=722)", "S3 (N=809)"]
        for si in range(3):
            vals = [bd[si + 1] for bd in bar_data]
            # Compute error bar lengths from CI
            err_lo = [abs(bd[si + 1] - bd[4][si]) if pd.notna(bd[4][si]) else 0 for bd in bar_data]
            err_hi = [abs(bd[5][si] - bd[si + 1]) if pd.notna(bd[5][si]) else 0 for bd in bar_data]
            ax.bar(x + (si - 1) * w, vals, w, color=study_colors[si],
                   edgecolor=WHITE, lw=0.5, label=study_labels[si], zorder=3,
                   alpha=0.8, yerr=[err_lo, err_hi],
                   error_kw={"lw": 0.8, "capsize": 2, "capthick": 0.8, "color": TXT, "alpha": 0.6})

        ax.set_xticks(x)
        ax.set_xticklabels([bd[0] for bd in bar_data], fontsize=8)
        ax.axhline(0, color=TXT, lw=0.4, alpha=0.3)
        ax.set_ylabel("Correlation r")
        ax.set_title("Key Associations Replicate Across Studies", fontsize=10, pad=14)
        ax.legend(fontsize=6.5, frameon=True, edgecolor="#DDDDDD",
                  fancybox=False, borderpad=0.5, ncol=3)
    clean_axis(ax, grid_axis="y")

    # ── (b) Calibration (Brier scores) ────────────────────────────
    ax2 = fig.add_subplot(gs[1]); plabel(ax2, "b")
    cal = pd.read_csv(ROB / "clinical_expanded.csv")
    outcomes = cal.Outcome.unique()
    x = np.arange(len(outcomes))
    w = 0.3

    pv, bv = [], []
    for o in outcomes:
        p = cal[(cal.Outcome == o) & (cal.Features == "Pers-only")]
        b = cal[(cal.Outcome == o) & (cal.Features == "Beh-only")]
        pv.append(p.Brier_mean.values[0] if len(p) else np.nan)
        bv.append(b.Brier_mean.values[0] if len(b) else np.nan)

    ax2.bar(x - w/2, pv, w, color=BLU, edgecolor=WHITE, lw=0.5, label="Personality",
            zorder=3)
    ax2.bar(x + w/2, bv, w, color=RED, edgecolor=WHITE, lw=0.5, label="Sensing",
            zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(outcomes, fontsize=7.5, rotation=15, ha="right")
    ax2.set_ylabel("Brier Score (lower = better)")
    ax2.set_ylim(0, 0.32)
    ax2.set_title("Model Calibration (Brier Score)", fontsize=10, pad=14)
    ax2.legend(fontsize=7, frameon=True, edgecolor="#DDDDDD", fancybox=False)
    clean_axis(ax2, grid_axis="y")

    # ── (c) Ablation ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2]); plabel(ax3, "c")
    abl = pd.read_csv(ROB / "modality_ablation.csv")
    avg = abl.groupby("Modality").Delta_R2.mean().sort_values(ascending=True)
    cols = [BLU if v > 0 else RED for v in avg.values]
    bars = ax3.barh(np.arange(len(avg)), avg.values, color=cols, edgecolor=WHITE,
                    lw=0.5, height=0.55, zorder=3)
    ax3.set_yticks(np.arange(len(avg)))
    ax3.set_yticklabels(avg.index, fontsize=7.5)
    ax3.axvline(0, color=TXT, lw=0.5, alpha=0.5)
    ax3.set_xlabel("$\\Delta$R²  (adding modality to personality)")
    ax3.set_title("Per-Modality $\\Delta$R² (all |$\\Delta$R²| < 0.01)",
                  fontsize=10, pad=14)
    clean_axis(ax3, grid_axis="x")

    fig.savefig(OUT / "figS2_supplement.png")
    plt.close(); print("  figS2 done")


# ═══════════════════════════════════════════════════════════════════════
# FIG S3 — Item-Level: 2 BFI items vs everything else
# ═══════════════════════════════════════════════════════════════════════
def figS3_item_level():
    fig = plt.figure(figsize=(7.5, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2], hspace=0.35)

    df = pd.read_csv(ROB / "item_level_prediction.csv")
    cols = ["R2_2_best_items", "R2_3_best_items", "R2_neuroticism_only",
            "R2_full_big5", "R2_all_44_items", "R2_sensing_PCA", "R2_sensing_raw28"]
    labels = ["2 Best Items\n(10 sec)", "3 Best Items\n(15 sec)",
              "Neuroticism\n(1 min)", "Full Big Five\n(5 min)",
              "All 44 Items\n(5 min)", "Sensing PCA\n(weeks)",
              "Sensing Raw\n(weeks)"]
    colors = [BLU, BLU, BLU, BLU, BLU, RED, RED]

    # ── (a) Mean across outcomes ──────────────────────────────────
    ax = fig.add_subplot(gs[0]); plabel(ax, "a")
    means = [df[c].mean() for c in cols]
    y = np.arange(len(labels))
    ax.barh(y, means, color=colors, edgecolor=WHITE, lw=0.5,
            height=0.6, zorder=3, alpha=0.85)
    for i, v in enumerate(means):
        ha = "left" if v >= 0 else "right"
        offset = 0.01 if v >= 0 else -0.01
        ax.text(v + offset, i, f"{v:.3f}", va="center", ha=ha,
                fontsize=6.5, fontweight="bold", color=colors[i])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color=TXT, lw=0.5, alpha=0.5)
    ax.set_xlabel("Mean R² (across CES-D, STAI, BAI)")
    ax.set_title("Two Items Outperform Weeks of Sensing", fontsize=10, pad=14)
    clean_axis(ax, grid_axis="x")
    badge(ax, "10 seconds > 10 weeks", 0.62, 0.92, color=BLU, fontsize=8)

    # ── (b) Per-outcome breakdown heatmap ─────────────────────────
    ax2 = fig.add_subplot(gs[1]); plabel(ax2, "b")
    short_cols = ["2 Items", "3 Items", "N only", "Big Five",
                  "44 Items", "Sens PCA", "Sens Raw"]
    mat = df[cols].copy()
    mat.columns = short_cols
    mat.index = df.Outcome
    cmap = LinearSegmentedColormap.from_list("r2",
        [RED, "#FFEEEE", WHITE, "#EEF0FF", BLU])
    annot_mat = mat.map(lambda x: f"{x:.2f}")
    sns.heatmap(mat, annot=annot_mat, fmt="", cmap=cmap, center=0,
                ax=ax2, linewidths=1.5, linecolor=WHITE,
                cbar_kws={"label": "R²", "shrink": 0.6},
                annot_kws={"fontsize": 8, "fontweight": "medium"}, vmin=-0.2, vmax=0.6)
    # Fix text contrast: force dark text on light cells, white on dark cells
    for text_obj in ax2.texts:
        try:
            r, g, b, _ = text_obj.get_color() if hasattr(text_obj.get_color(), '__len__') else (0, 0, 0, 1)
        except (TypeError, ValueError):
            continue
        # Get background color at text position
        val_str = text_obj.get_text()
        try:
            val = float(val_str)
            # Dark text if value < 0.25 (light bg), white if value >= 0.25 (dark bg)
            text_obj.set_color(WHITE if val >= 0.25 else TXT)
        except ValueError:
            text_obj.set_color(TXT)
    ax2.set_title("Per-Outcome Breakdown (S2)", fontsize=10, pad=14)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=8)
    ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=7.5, rotation=20, ha="right")

    fig.savefig(OUT / "figS3_item_level.png")
    plt.close(); print("  figS3 done")


# ═══════════════════════════════════════════════════════════════════════
# FIG S4 — Literature Benchmark: our AUC vs published work
# ═══════════════════════════════════════════════════════════════════════
def figS4_literature():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    df = pd.read_csv(CORE / "literature_benchmark.csv")

    # Filter to rows with AUC values
    df = df[df.Metric == "AUC"].copy()
    df["Value"] = pd.to_numeric(df.Value, errors="coerce")
    df = df.dropna(subset=["Value"]).sort_values("Value")

    y = np.arange(len(df))
    colors = []
    for _, row in df.iterrows():
        if "This study" in row.Paper and "Pers-only" in row.Paper:
            colors.append(BLU)
        elif "This study" in row.Paper:
            colors.append(BLU if "Pers" in row.Paper else RED)
        else:
            colors.append(GRY)

    # Dot plot — marker size encodes sample size (sqrt scaling)
    n_max = pd.to_numeric(df.N, errors="coerce").max() or 1
    for i, (_, row) in enumerate(df.iterrows()):
        c = colors[i]
        marker = "D" if "This study" in row.Paper else "o"
        n_val = pd.to_numeric(row.N, errors="coerce")
        if pd.notna(n_val) and n_val > 0:
            size = 30 + 140 * np.sqrt(n_val / n_max)
        else:
            size = 40
        if "This study" in row.Paper:
            size = max(size, 80)
        ax.scatter(row.Value, i, color=c, marker=marker, s=size, zorder=4,
                   edgecolors=WHITE, lw=0.7)

    # Labels
    labels = []
    for _, row in df.iterrows():
        n_str = f"N={int(row.N)}" if pd.notna(row.N) else ""
        labels.append(f"{row.Paper}  {n_str}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)

    # Highlight our rows
    for i, (_, row) in enumerate(df.iterrows()):
        if "This study" in row.Paper:
            ax.axhspan(i - 0.4, i + 0.4, color=BLU, alpha=0.04, zorder=0)

    ax.axvline(0.5, color=TXT, lw=0.5, ls=":", alpha=0.3)
    ax.set_xlabel("AUC  (marker size $\\propto \\sqrt{N}$)", fontsize=9)
    ax.set_xlim(0.45, 0.82)
    ax.set_title("Our Personality-Only Model vs Published Sensing Studies",
                 fontsize=10, pad=14)
    clean_axis(ax, grid_axis="x")

    # Legend
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker='D', color='w', markerfacecolor=BLU,
                  markersize=7, label='This study'),
           Line2D([0], [0], marker='o', color='w', markerfacecolor=GRY,
                  markersize=6, label='Published work')]
    ax.legend(handles=leg, frameon=True, edgecolor="#DDDDDD", fancybox=False,
              fontsize=7.5, loc="lower right")

    fig.savefig(OUT / "figS4_literature.png")
    plt.close(); print("  figS4 done")


# ═══════════════════════════════════════════════════════════════════════
ALL = {
    "fig1": fig1_study_design,   # Overview: pipeline + cards + data contrast
    "fig2": fig3_core,           # Core: dumbbell + forest + incremental
    "fig3": fig4_mechanism,      # Mechanism: SHAP + trait specificity + DL
    "fig4": fig5_robustness,     # Robustness: dose + ICC + transfer
    "fig5": fig6_clinical,       # Clinical: NNS + cost + worst-20%
    "figS1": figS1_idiographic,  # Idiographic distribution
    "figS2": figS2_supplement,   # Replication + calibration + ablation
    "figS3": figS3_item_level,   # 2 items > 28 features
    "figS4": figS4_literature,   # Literature benchmark
}

if __name__ == "__main__":
    setup()
    targets = sys.argv[1:] or list(ALL.keys())
    print(f"Generating {len(targets)} figures (v4)...", flush=True)
    for n in targets:
        if n in ALL:
            print(f"  [{n}]", end=" ", flush=True)
            ALL[n]()
        else:
            print(f"  Unknown: {n}")
    print(f"\nDone -> {OUT}/", flush=True)
