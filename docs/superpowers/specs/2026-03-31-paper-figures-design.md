# Design: Paper Figure & Table System

**Date**: 2026-03-31
**Status**: Approved
**Target**: JMIR Mental Health (primary), JMIR mHealth and uHealth (backup)

## Goal

Design a complete figure/table system where: (1) looking at figures alone tells the full story, (2) figures + tables together form a complete slide deck, (3) follows JMIR formatting (no hard limits, large tables to Multimedia Appendices).

## Figure System: 10 Figures (8 Main + 2 Supplement)

### Narrative Arc

The 10 figures follow a story: What did we study? → What did we find? → Why? → Can we fix sensing? → So what clinically? → What's sensing good for?

### Main Text Figures (8)

#### Fig 1: Study Design Overview (NEW)
- **Story**: "Here's what we compared across 3 universities"
- **Type**: Schematic/infographic (not a data plot)
- **Content**: Three columns (S1/S2/S3), each showing:
  - University, year, N
  - Personality measure (BFI-44 or BFI-10) with icon (clipboard, ~seconds)
  - Sensing modalities with icons (phone, watch, GPS) with duration (~weeks)
  - Arrow down to outcomes (depression, anxiety, stress, GPA)
- **Key visual**: Left side = "5-minute questionnaire", Right side = "weeks of continuous sensing"
- **Data source**: Manual/hardcoded from study metadata
- **Implementation**: matplotlib with custom layout, or tikz in LaTeX
- **Color scheme**: Blue for personality, Red/grey for sensing (consistent throughout all figures)

#### Fig 2: Grand Synthesis — Personality vs Sensing R² (REDESIGN)
- **Story**: "Personality wins 14/15 comparisons"
- **Current**: Horizontal bar chart, OK but dense
- **Redesign goals**:
  - Cleaner layout: grouped by study (S1, S2, S3 sections)
  - Personality bars in blue, sensing in red, with 95% CI whiskers
  - Vertical dashed line at R²=0
  - Bold annotation: "14/15 (93%)" personality wins
  - Larger font, publication-ready
- **Data source**: `results/core/grand_synthesis.csv`
- **Size**: Full width, ~6 inches tall

#### Fig 3: Meta-Analysis Forest Plot (KEEP, minor polish)
- **Story**: "Neuroticism is the universal predictor (r=.44–.63)"
- **Current**: Good, keep as-is with minor font/spacing improvements
- **Data source**: `results/core/meta_analysis.csv`

#### Fig 4: SHAP Feature Importance Heatmap (NEW — main text)
- **Story**: "Neuroticism ranks #1 across all studies and models"
- **Type**: Heatmap with studies as columns, traits as rows
- **Content**:
  - Rows: Big Five traits (E, A, C, N, O)
  - Columns: Study × Outcome (S2-CES-D, S2-STAI, S3-BDI-II, S3-STAI, S3-PSS, etc.)
  - Cell color: Mean |SHAP| value (darker = more important)
  - Annotation: Rank number in each cell (1=most important)
  - Neuroticism row should be visually dominant (darkest)
- **Data source**: `results/core/three_study_shap_mh.csv` or `results/by_study/s*/tables/shap_personality_*.csv`

#### Fig 5: Deep Learning Cannot Rescue Sensing (REDESIGN from Analysis 42)
- **Story**: "Even CNN, MOMENT, and GradientBoosting fail — it's the signal, not the model"
- **Current**: Two-panel grouped bar chart, OK but needs polish
- **Redesign goals**:
  - Single panel with all 8 outcomes on x-axis (S3 first, then S2)
  - 5 model types as grouped bars with distinct colors
  - Personality bars clearly above zero line, all sensing bars below
  - Annotation: "Foundation model R² = -1.0 to -1.7" callout
  - Cleaner legend, larger fonts
- **Data source**: `results/robustness/deep_learning_comparison.csv`

#### Fig 6: Dose-Response — More Data Doesn't Help (NEW — main text)
- **Story**: "7 days of sensing ≈ 92 days — the problem isn't data quantity"
- **Type**: Line plot
- **Content**:
  - x-axis: Days of sensing data (7, 14, 30, 60, 92)
  - y-axis: R² (cross-validated)
  - Lines: one per outcome (BDI-II, STAI, PSS, CESD, UCLA)
  - Horizontal reference band: personality R² range (shaded blue)
  - All sensing lines should be flat near zero, well below personality band
- **Data source**: `results/robustness/dose_response.csv`

#### Fig 7: Stable but Irrelevant — Sensing Temporal Reliability (REDESIGN from Analysis 44)
- **Story**: "Sensing features are highly reliable (ICC up to 0.98) — the problem is construct relevance, not measurement quality"
- **Current**: Line plot with CI bands, good but needs polish
- **Redesign goals**:
  - Cleaner lines, fewer overlapping CI bands
  - Personality reference band (BFI-44 r=0.85, BFI-10 r=0.75) as shaded horizontal region
  - Key message annotation: "Sensing ICC ≥ Personality test-retest"
  - Modality labels on right side of lines (not legend)
- **Data source**: `results/robustness/temporal_reliability.csv`

#### Fig 8: Clinical Practical Significance — NNS (REDESIGN from Analysis 43)
- **Story**: "Per 100 people screened, personality catches more true cases"
- **Current**: Two-panel (TP per 100 + NNS bars)
- **Redesign goals**:
  - Focus on Panel A (TP per 100) as the main message — more intuitive than NNS
  - Pictograph/icon array style if feasible (100 person icons, colored by TP/FP/FN/TN)
  - If pictograph too complex, clean grouped bar chart with annotations: "+2.4 more cases per 100"
  - Three colors: Pers-only (blue), Pers+Beh (purple), Beh-only (grey)
- **Data source**: `results/robustness/nns_comparison.csv`

### Supplement Figures (2)

#### Fig S1: Cost-Effectiveness (existing, polish)
- **Story**: "10 seconds and $0 vs weeks and $100+"
- **Data source**: `results/core/cost_effectiveness.csv`
- **Note**: Important for framing but less "novel" — supplement is fine

#### Fig S2: Idiographic Value — Person-Specific R² Distribution (NEW)
- **Story**: "17% of individuals show R²>0.3 — sensing works for some people"
- **Type**: Histogram or raincloud plot of per-person R² values
- **Content**:
  - Distribution of person-specific R² from idiographic models
  - Vertical line at R²=0 and R²=0.3
  - Annotation: "17% above 0.3", "83% at or below 0"
- **Data source**: `results/robustness/idiographic_models.csv`

## Table System: 5 Main Tables + Multimedia Appendices

### Main Tables

| Table | Content | Source | Status |
|-------|---------|--------|--------|
| 1 | Three-study overview (N, measures, duration) | Manual | Keep as-is |
| 2 | Grand synthesis R² with 95% CI | `grand_synthesis.csv` | Keep as-is |
| 3 | Meta-analysis: Neuroticism correlations | `meta_analysis.csv` | Keep as-is |
| 4 | Clinical classification AUC | `clinical_classification.csv` | Keep as-is |
| 5 | Deep learning model comparison | `deep_learning_comparison.csv` | NEW |

### Multimedia Appendices (JMIR format)

- **Appendix 1**: Full 44 robustness analyses summary table
- **Appendix 2**: Incremental validity (moved from main)
- **Appendix 3**: SHAP vs OLS agreement (moved from main)
- **Appendix 4**: All study-specific detailed tables
- **Appendix 5**: STROBE checklist

## Visual Design Conventions

All figures must follow these conventions for consistency:

- **Colors**: Blue (#2980b9) = personality, Red (#e74c3c) = sensing/behavior, Grey (#95a5a6) = combined/neutral, Purple (#8e44ad) = deep learning/MOMENT, Orange (#f39c12) = GradientBoosting
- **Font**: Arial/Helvetica, 10pt minimum for axis labels, 12pt for titles
- **Size**: Full width = 7.5 inches (JMIR column width), half width = 3.5 inches
- **DPI**: 300 for submission, 600 for final
- **Format**: PNG for submission, PDF/EPS for final
- **Annotations**: Key findings annotated directly on figure (not just caption)
- **R² presentation**: Always show 0 reference line for R²; negative R² is meaningful

## Implementation

All figures generated by a single script: `scripts/05_paper_materials/paper_figures.py`

This script:
1. Loads all required CSVs
2. Generates all 10 figures with consistent styling
3. Saves to `paper/figures/fig1_study_design.png` through `paper/figures/figS2_idiographic.png`
4. Applies shared style settings (font, colors, sizes)

Table 5 (new) generated alongside figures or as a separate LaTeX table in `paper/tables.tex`.

## What Gets Created vs What Gets Modified

**New files**:
- `scripts/05_paper_materials/paper_figures.py` — master figure generation script
- `paper/figures/fig1_study_design.png` through `paper/figures/figS2_idiographic.png` (10 PNGs)

**Modified files**:
- `paper/figures.tex` — update figure references and captions
- `paper/tables.tex` — add Table 5 (deep learning comparison)
