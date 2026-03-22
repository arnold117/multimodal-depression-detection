# Paper Pipeline: Analysis Fixes + Manuscript Writing

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix analysis gaps (CI, FDR) and write the full manuscript with two framing versions (old: personality-predicts-MH; new: sensing-vs-questionnaires), ready for submission to CHB or JMIR Mental Health.

**Architecture:** Three phases — (A) fix analysis outputs so Results numbers are final, (B) write shared manuscript sections (Method, Results, references.bib), (C) write two versions of framing-dependent sections (Intro, Discussion, Abstract). Both framings share ~70% of the manuscript.

**Tech Stack:** Python 3.12 (scikit-learn, statsmodels, scipy), LaTeX (natbib/apalike), BibTeX

---

## Phase A: Analysis Fixes

### Task 1: Fix requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add missing dependencies and pin Python version**

Add `optuna` and `joblib` (used in 5 and 2 scripts respectively). Add Python version comment. Pin to `>=` minimum versions that match current environment:

```
# Python >= 3.10 required

# Core
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
pyarrow>=6.0.0

# ML & Statistics
scikit-learn>=1.0.0
statsmodels>=0.13.0
pingouin>=0.5.0
factor-analyzer>=0.4.0
scikit-posthocs>=0.7.0
semopy>=2.3.0
optuna>=3.0.0
joblib>=1.1.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
```

- [ ] **Step 2: Verify all imports are covered**

Run:
```bash
grep -rh "^import \|^from " scripts/*.py src/**/*.py | sort -u
```
Cross-check against requirements.txt. Standard library imports (pathlib, sys, warnings, etc.) don't need listing.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "fix: add missing optuna and joblib to requirements.txt"
```

---

### Task 2: Add 95% CI to grand_synthesis.csv

The core comparison table (16 rows: personality R² vs sensing R² across all outcomes) currently has only point estimates. `supplementary_core.py` already has `kfold_r2()` returning CI — but `supplementary_phase16g.py` uses `quick_cv_r2()` which doesn't.

**Files:**
- Modify: `scripts/supplementary_phase16g.py` — update `quick_cv_r2()` to return CI; update grand synthesis loop
- Output: `results/comparison/supplementary/grand_synthesis.csv` — add CI columns

- [ ] **Step 1: Modify `quick_cv_r2()` to return fold-level R² values**

In `scripts/supplementary_phase16g.py` (line 50), change the `quick_cv_r2()` function. The current function returns `(mean_r2, N)` as a tuple. Change to return a dict with CI. **Critical:** preserve the existing small-N guard (`ns = min(n_splits, len(y_c) // 3)`) which prevents crashes on S1 (N=28):

```python
def quick_cv_r2(X, y, n_splits=5, n_repeats=5, alpha=1.0):
    """Quick Ridge CV, returns dict with R2_mean, R2_ci_lo, R2_ci_hi, N."""
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X_c, y_c = X[mask], y[mask]
    if len(y_c) < 20:
        return {"R2_mean": np.nan, "R2_ci_lo": np.nan, "R2_ci_hi": np.nan, "N": len(y_c)}
    ns = min(n_splits, len(y_c) // 3)  # PRESERVE: small-N guard for S1
    if ns < 2:
        return {"R2_mean": np.nan, "R2_ci_lo": np.nan, "R2_ci_hi": np.nan, "N": len(y_c)}
    cv = RepeatedKFold(n_splits=ns, n_repeats=n_repeats, random_state=RS)
    r2s = []
    for tr, te in cv.split(X_c):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_c[tr])
        Xte = sc.transform(X_c[te])
        m = Ridge(alpha=alpha)
        m.fit(Xtr, y_c[tr])
        r2s.append(r2_score(y_c[te], m.predict(Xte)))
    r2s = np.array(r2s)
    return {
        "R2_mean": float(np.mean(r2s)),
        "R2_ci_lo": float(np.percentile(r2s, 2.5)),
        "R2_ci_hi": float(np.percentile(r2s, 97.5)),
        "N": len(y_c),
    }
```

- [ ] **Step 2: Update ALL 16 call sites of `quick_cv_r2()`**

There are **16 call sites** in the file using tuple unpacking. All must be updated. The call sites are at lines: 184, 185, 186, 193, 258, 259, 260, 379, 380, 381, 384, 479, 480, 493, 494, 506, 507.

Current patterns and their replacements:

```python
# Pattern 1: r2, n = quick_cv_r2(...)
# Replace with:
_res = quick_cv_r2(...)
r2, n = _res["R2_mean"], _res["N"]

# Pattern 2: r2, _ = quick_cv_r2(...)
# Replace with:
_res = quick_cv_r2(...)
r2 = _res["R2_mean"]

# Pattern 3 (grand synthesis section, ~line 479+): r2_p, n = ... and r2_b, _ = ...
# Replace with:
_res_p = quick_cv_r2(...)
r2_p, n = _res_p["R2_mean"], _res_p["N"]
_res_b = quick_cv_r2(...)
r2_b = _res_b["R2_mean"]
```

Use descriptive temp names (`_res_pers`, `_res_sens`, `_res_both`) in the grand synthesis section to preserve CI for output.

- [ ] **Step 3: Update grand synthesis loop to include CI columns**

In the grand synthesis section (~line 463–556), update the dict appended to `all_comparisons` to include CI:

```python
all_comparisons.append({
    "Study": study_label,
    "Outcome": outcome_label,
    "Domain": domain,
    "N": res_pers["N"],
    "R2_personality": res_pers["R2_mean"],
    "R2_pers_ci_lo": res_pers["R2_ci_lo"],
    "R2_pers_ci_hi": res_pers["R2_ci_hi"],
    "R2_sensing": res_sens["R2_mean"],
    "R2_sens_ci_lo": res_sens["R2_ci_lo"],
    "R2_sens_ci_hi": res_sens["R2_ci_hi"],
    "Pers_wins": res_pers["R2_mean"] > res_sens["R2_mean"],
    "Delta": res_pers["R2_mean"] - res_sens["R2_mean"],
})
```

- [ ] **Step 4: Run the grand synthesis section and verify output**

Run:
```bash
/Users/arnold/miniforge3/envs/qbio/bin/python scripts/supplementary_phase16g.py
```
Check the output CSV has the new columns:
```bash
head -3 results/comparison/supplementary/grand_synthesis.csv
```
Expected new columns: `R2_pers_ci_lo`, `R2_pers_ci_hi`, `R2_sens_ci_lo`, `R2_sens_ci_hi`

- [ ] **Step 5: Commit**

```bash
git add scripts/supplementary_phase16g.py results/comparison/supplementary/grand_synthesis.csv
git commit -m "feat: add 95% CI to grand synthesis comparison table"
```

---

### Task 3: Add FDR correction to Phase 16 supplementary analyses

The 41 supplementary analyses report various "trace signals" (8 out of 41) without family-wise correction. Need to collect all p-values and apply BH-FDR.

**Files:**
- Create: `scripts/supplementary_fdr_correction.py` — standalone script that reads existing CSVs and applies FDR
- Output: `results/comparison/supplementary/phase16_fdr_summary.csv`

- [ ] **Step 1: Create FDR correction script**

This script reads the **4 supplementary CSVs that contain p-values** (verified by column inspection) and applies BH-FDR across the family of tests. Most Phase 16 CSVs (37/41) report only R² without formal p-values — for those, R² ≤ 0 is definitively null and no p-value correction is needed.

**CSVs with p-values (exact column names):**

| CSV | P-value column(s) | Rows |
|-----|-------------------|------|
| `delong_tests.csv` | `p_value` | 5 AUC comparisons |
| `full_demographics.csv` | `p_personality`, `p_behavior` | per-outcome demographic controls |
| `missing_as_signal.csv` | `p_completeness` | completeness-outcome correlation |
| `sensing_reliability.csv` | `Split_half_p` | per-feature split-half reliability |

Note: `incremental_validity.csv` (in `results/comparison/`, not `supplementary/`) already has FDR correction applied (`p_fdr`, `sig_fdr` columns) from Phase 15. Do not re-correct.

```python
#!/usr/bin/env python3
"""Apply BH-FDR correction across all Phase 16 supplementary analyses with p-values."""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.stats.multitest import multipletests

PROJECT = Path(__file__).parent.parent
SUP = PROJECT / "results" / "comparison" / "supplementary"
OUT = SUP / "phase16_fdr_summary.csv"

results = []

# 1. DeLong tests (5 AUC comparisons: Pers-only vs Pers+Beh)
df = pd.read_csv(SUP / "delong_tests.csv")
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
```

- [ ] **Step 2: Run the FDR script and examine results**

```bash
/Users/arnold/miniforge3/envs/qbio/bin/python scripts/supplementary_fdr_correction.py
cat results/comparison/supplementary/phase16_fdr_summary.csv
```

Key question: Do any of the 8 "trace signals" survive FDR? If not, the paper narrative simplifies to "sensing adds nothing, period."

- [ ] **Step 3: Commit**

```bash
git add scripts/supplementary_fdr_correction.py results/comparison/supplementary/phase16_fdr_summary.csv
git commit -m "feat: add BH-FDR correction across Phase 16 supplementary analyses"
```

---

### Task 4: Add CI to clinical classification metrics

Currently `clinical_classification.csv` has CI for AUC but not for Sensitivity, Specificity, PPV, NPV.

**Files:**
- Modify: `scripts/clinical_utility.py` — store fold-level metrics, compute percentile CI
- Output: `results/comparison/clinical_classification.csv` — add CI columns

- [ ] **Step 1: Modify `run_classification()` to collect fold-level metrics**

In `scripts/clinical_utility.py`, the `run_classification()` function computes Sensitivity, Specificity etc. per fold but only stores the mean. Modify to also store percentile CI:

After the CV loop (line ~103), before the return dict is constructed (line ~118), add CI computation using the **exact variable names** from the code (`sens_list`, `spec_list`, `ppv_list`, `npv_list`, `f1_list` — defined at line 74):

```python
    # Add CI (insert before the return dict at line 118)
    sens_ci_lo = float(np.percentile(sens_list, 2.5))
    sens_ci_hi = float(np.percentile(sens_list, 97.5))
    spec_ci_lo = float(np.percentile(spec_list, 2.5))
    spec_ci_hi = float(np.percentile(spec_list, 97.5))
    ppv_ci_lo = float(np.percentile(ppv_list, 2.5))
    ppv_ci_hi = float(np.percentile(ppv_list, 97.5))
    npv_ci_lo = float(np.percentile(npv_list, 2.5))
    npv_ci_hi = float(np.percentile(npv_list, 97.5))
```

Then add to the return dict (line 118-125):
```python
    return {
        # ... existing keys ...
        "Sens_CI_lo": sens_ci_lo, "Sens_CI_hi": sens_ci_hi,
        "Spec_CI_lo": spec_ci_lo, "Spec_CI_hi": spec_ci_hi,
        "PPV_CI_lo": ppv_ci_lo, "PPV_CI_hi": ppv_ci_hi,
        "NPV_CI_lo": npv_ci_lo, "NPV_CI_hi": npv_ci_hi,
    }
```

- [ ] **Step 2: Run clinical_utility.py and verify output**

```bash
/Users/arnold/miniforge3/envs/qbio/bin/python scripts/clinical_utility.py
head -2 results/comparison/clinical_classification.csv
```
Verify new CI columns appear.

- [ ] **Step 3: Commit**

```bash
git add scripts/clinical_utility.py results/comparison/clinical_classification.csv
git commit -m "feat: add 95% CI for sensitivity, specificity, PPV, NPV"
```

---

## Phase B: Shared Manuscript Content

### Task 5: Build references.bib

**Files:**
- Modify: `paper/references.bib`

- [ ] **Step 1: Populate with all identified references**

The `paper/outline_comparison.md` identifies 30+ essential references. Build a complete .bib file covering:

**Must-cite (from outline):**
- Kotov et al. (2010) Psychological Bulletin — N→MH meta-analysis
- Insel (2018) World Psychiatry — digital phenotyping vision
- Wang et al. (2014) UbiComp — StudentLife dataset
- Torous et al. (2021) World Psychiatry — digital psychiatry review
- Saeb et al. (2015) JMIR — early GPS→PHQ-9
- Stachl et al. (2020) PNAS — smartphone→Big Five
- Chikersal et al. (2021) ACM TOCHI — sensing→depression
- Busshart et al. (2026) JMIR mHealth — scoping review (our gap)
- Muller et al. (2021) Scientific Reports — GPS fails at scale
- Xu et al. (2022) IMWUT — GLOBEM dataset
- Das Swain et al. (2022) CHI — semantic gap theory
- Currey & Torous (2022) BJPsych Open — mindLAMP weak correlations
- Adler et al. (2022) PLOS ONE — cross-study generalization fails

**Methods references:**
- Lundberg & Lee (2017) — SHAP
- Breiman (2001) — Random Forest
- Tibshirani (1996) — Lasso
- Benjamini & Hochberg (1995) — FDR
- DerSimonian & Laird (1986) — random-effects meta-analysis
- Cohen (1988) — effect sizes

**Instruments:**
- John & Srivastava (1999) — BFI-44
- Rammstedt & John (2007) — BFI-10
- Kroenke et al. (2001) — PHQ-9
- Radloff (1977) — CES-D
- Spielberger (1983) — STAI
- Beck et al. (1988) — BAI
- Cohen et al. (1983) — PSS
- Lowe et al. (2010) — PHQ-4

**Additional (personality-MH):**
- Lahey (2009) — neuroticism as transdiagnostic
- Widiger & Trull (2007) — FFM and psychopathology
- Roberts et al. (2007) — conscientiousness across lifespan

**Additional (sensing):**
- Harari et al. (2020) — sensing in social/behavioral science
- Onnela & Rauch (2016) — digital phenotyping concept

Use WebSearch to find exact BibTeX entries from Google Scholar or publisher sites. For each reference: search for the paper title, find the official publisher page (e.g., APA PsycNet, PubMed, ACM DL), and extract the BibTeX. **Do not fabricate entries** — every field (year, journal, volume, pages, DOI) must come from the actual publication. If a BibTeX entry cannot be verified, leave a `% TODO: verify` comment. Total: ~40-50 entries.

- [ ] **Step 2: Verify .bib file compiles**

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Check for undefined citation warnings.

- [ ] **Step 3: Commit**

```bash
git add paper/references.bib
git commit -m "feat: populate references.bib with ~45 entries"
```

---

### Task 6: Write Method section

**Files:**
- Modify: `paper/main.tex` — fill in Method section (~2000-2500 words)

**Source data for writing:**
- `README.md` lines 1-80 — study descriptions, N, instruments
- `project_summary.md` — detailed methodology
- `report/01_study1_studentlife.md` — S1 details
- `report/02_study2_nethealth.md` — S2 details
- `report/03_study3_globem.md` — S3 details
- `results/comparison/supplementary/full_demographics.csv` — demographic data

The Method section is identical for both framings.

- [ ] **Step 1: Write Datasets subsection**

Include Table 1 (three-study overview). For each study:
- University, year, N, sensing period, sensing modalities
- Recruitment, eligibility, attrition
- Demographics (age M±SD, % female, ethnicity if available)
- Ethical approval statement (cite original study IRB)

- [ ] **Step 2: Write Measures subsection**

For each instrument: full name, abbreviation, # items, response scale, reliability (Cronbach's α from our data), clinical cutoff (if applicable), citation.

Organize as:
- Big Five (BFI-44 vs BFI-10, note S3 uses short form)
- Depression (PHQ-9, CES-D, BDI-II)
- Anxiety (STAI, BAI, GAD-7)
- Other (PSS, Loneliness, Flourishing, GPA)
- Behavioral features (per study: modalities → PCA composites)

- [ ] **Step 3: Write Analytic Strategy subsection**

Cover:
- ML pipeline: Ridge/ElasticNet/RF/SVR, 5-fold repeated CV, StandardScaler inside folds
- Permutation tests for significance
- SHAP interpretability approach
- Clinical classification: 10×10 stratified CV, LR+RF, Youden's J
- Incremental validity: nested F-test, BH-FDR
- Meta-analysis: random-effects (DerSimonian-Laird), Fisher z, I², forest plots
- Demographic controls: hierarchical regression

- [ ] **Step 4: Add Data/Code Availability, Ethics, CRediT statements**

Add after Discussion or as required by target journal:
```latex
\section*{Data Availability}
Study 1 (StudentLife) data are available from the StudentLife project
(https://studentlife.cs.dartmouth.edu/). Study 2 (NetHealth) data are
available via https://nethealth.nd.edu/ with registration. Study 3 (GLOBEM)
data are available via PhysioNet with credentialed access and data use
agreement. Analysis code is available at [GitHub URL].

\section*{Ethics Statement}
All three studies received institutional ethical approval from their
respective universities. [Cite specific IRB numbers from original papers.]
This secondary analysis was conducted under [NUS IRB number/exemption].

\section*{Author Contributions (CRediT)}
[Arnold]: Conceptualization, Data curation, Formal analysis, Methodology,
Software, Visualization, Writing -- original draft.
[Author B]: [TBD].
Cyrus S.~H.~Ho: Supervision, Writing -- review \& editing.

\section*{Conflicts of Interest}
The authors declare no competing interests.
```

- [ ] **Step 5: Commit**

```bash
git add paper/main.tex
git commit -m "feat: write Method section + ethics/data availability statements"
```

---

### Task 7: Write Results section

**Files:**
- Modify: `paper/main.tex` — fill in Results section (~2000-2500 words)

**Source data:**
- `results/comparison/supplementary/grand_synthesis.csv` — core comparison (with new CI)
- `results/comparison/meta_analysis.csv` — pooled effects
- `results/comparison/clinical_classification.csv` — AUC, Sens, Spec (with new CI)
- `results/comparison/incremental_validity.csv` — FDR-corrected incremental tests (already exists in `results/comparison/` — check exact path)
- `results/comparison/shap_ranking_comparison.csv` — SHAP vs β
- `results/comparison/mlp_regression.csv` — MLP robustness
- `results/comparison/supplementary/phase16_fdr_summary.csv` — FDR on Phase 16 (new)

The Results section is ~90% identical for both framings. The only difference is narrative emphasis:
- Old framing: leads with personality's success
- New framing: leads with the comparison question

Write the new framing version (as it's the recommended one), with comments marking where old framing would differ.

**Important:** The existing `main.tex` has old-framing section headers (e.g., "Personality Predicts Mental Health Across Three Studies"). When writing the Results in new-framing style, update the subsection titles to match (e.g., "Head-to-Head: Personality Questionnaires vs Passive Sensing"). Also update the `\title{}` to the new framing title from `outline_comparison.md`.

- [ ] **Step 1: Update title and Results subsection headers for new framing**

Change the title (line 18-20) to:
```latex
\title{Can Passive Smartphone Sensing Replace Personality Questionnaires\\
       for Mental Health Prediction? Evidence from Three University\\
       Samples ($N = 1{,}559$)}
```

Update Results subsection headers to match RQ structure.

- [ ] **Step 2: Write 3.1 — Head-to-head comparison (personality vs sensing)**

Present grand_synthesis.csv as Table 2: R² [95% CI] for personality-only vs sensing-only, for all 15 outcomes × 3 studies.
Key stats: personality wins 14/15 (93%), median ΔR² = X.XX.

- [ ] **Step 2: Write 3.2 — Meta-analytic synthesis**

Present meta_analysis.csv as Table 3 + Figure 1 (forest plot).
Key stats: pooled r for N→Depression, N→Anxiety, C→GPA with 95% CI.
Note high I² and discuss briefly.

- [ ] **Step 3: Write 3.3 — Clinical classification**

Present clinical_classification.csv as Table 4.
Key stats: Personality-only AUC [CI] vs Pers+Beh AUC [CI] vs Beh-only AUC [CI].
DeLong test results.

- [ ] **Step 4: Write 3.4 — Incremental validity**

Present incremental_validity.csv as Table 5.
Key finding: 1/8 survives FDR (BDI-II, ΔR²=0.024).
S2 exception: Pers+Beh AUC 0.83-0.86.

- [ ] **Step 5: Write 3.5 — Methodological checks**

SHAP ≈ OLS β (τ=0.94), MLP < Ridge, gender controls.
Phase 16 FDR summary: X/41 survive correction.
Present as brief subsection or supplementary table reference.

- [ ] **Step 6: Commit**

```bash
git add paper/main.tex
git commit -m "feat: write Results section with tables and figure references"
```

---

## Phase C: Framing-Dependent Content

### Task 8: Write Introduction — New Framing ("Can sensing replace questionnaires?")

**Files:**
- Modify: `paper/main.tex` — fill in Introduction (~1500 words)

**Source:** `paper/outline_comparison.md` lines 36-60 (new framing outline)

- [ ] **Step 1: Write 1.1 — The digital phenotyping promise**

Set up the "target": Insel (2018) "smartphones as stethoscope", Torous et al. (2021), Saeb et al. (2015). The implicit assumption: continuous behavioral data > infrequent self-report.

- [ ] **Step 2: Write 1.2 — The personality questionnaire as baseline**

Introduce the "challenger": Kotov et al. (2010), BFI takes 5 minutes, robust N→MH link. Frame questionnaires as "low-tech but effective."

- [ ] **Step 3: Write 1.3 — The missing head-to-head comparison**

Cite Busshart et al. (2026) scoping review. Most sensing studies don't include personality baseline. Most personality studies don't include sensing. We uniquely have both in 3 datasets.

- [ ] **Step 4: Write 1.4 — Present study and RQs**

Three explicit research questions:
- RQ1: Which predicts mental health better — personality questionnaires or passive sensing?
- RQ2: Does sensing add incremental validity beyond personality?
- RQ3: Can complex models (MLP, ensembles) compensate for sensing's limitations?

- [ ] **Step 5: Commit**

```bash
git add paper/main.tex
git commit -m "feat: write Introduction (new framing — sensing vs questionnaires)"
```

---

### Task 9: Write Discussion — New Framing

**Files:**
- Modify: `paper/main.tex` — fill in Discussion (~2000 words)

**Source:** `paper/outline_comparison.md` lines 87-105, `report/07_grand_synthesis.md`

- [ ] **Step 1: Write 4.1 — An asymmetric contest**

5-min questionnaire >> weeks of continuous sensing. Why: personality captures stable traits, sensing captures state noise. Connect to Das Swain's "semantic gap."

- [ ] **Step 2: Write 4.2 — When does sensing help? (nuanced, not absolute)**

S2 Pers+Beh AUC 0.85 shows high-quality wearable+communication data can add marginally. But this requires Fitbit-level compliance — not typical. The 17% idiographic finding is intriguing but not actionable a priori.

- [ ] **Step 3: Write 4.3 — Implications for digital phenotyping**

Argue for "questionnaire baseline" as standard practice in sensing studies. Cite Muller (2021), Xu (2022 GLOBEM), Adler (2022) as converging evidence. Suggest reporting questionnaire-only AUC alongside sensing AUC.

- [ ] **Step 4: Write 4.4 — Simple models suffice**

SHAP = OLS β, MLP < Ridge. Complexity doesn't help when signal isn't there.

- [ ] **Step 5: Write 4.5 — Limitations**

WEIRD college samples, cross-sectional, BFI-10 reliability, secondary data, no clinical diagnoses, S1 small N, no medication control, race/ethnicity not available for all studies.

- [ ] **Step 6: Write Conclusion (1 paragraph)**

- [ ] **Step 7: Commit**

```bash
git add paper/main.tex
git commit -m "feat: write Discussion + Conclusion (new framing)"
```

---

### Task 10: Write Introduction — Old Framing ("Personality predicts MH")

**Files:**
- Create: `paper/main_old_framing.tex` — copy of main.tex with old framing Intro/Discussion

**Prerequisite:** Tasks 8-9 must be complete (main.tex has new-framing Intro/Discussion + shared Method/Results).

- [ ] **Step 1: Copy main.tex as base**

```bash
cp paper/main.tex paper/main_old_framing.tex
```

- [ ] **Step 2: Rewrite Introduction for old framing (~1500 words)**

Follow `outline_comparison.md` lines 1-25 (old framing outline):
- 1.1 Personality → MH literature (Kotov 2010, Lahey 2009)
- 1.2 Passive sensing promise (Wang 2014, Harari 2020)
- 1.3 Gaps: few multi-study replications, clinical utility rarely evaluated
- 1.4 Present study: 3 datasets, unified pipeline, hypotheses

Update title to old framing:
```latex
\title{Personality Traits Predict Mental Health Across Three Universities:\\
       A Multi-Study Machine Learning Investigation ($N = 1{,}559$)}
```

- [ ] **Step 3: Commit**

```bash
git add paper/main_old_framing.tex
git commit -m "feat: write Introduction (old framing — personality predicts MH)"
```

---

### Task 11: Write Discussion — Old Framing + Abstracts

**Files:**
- Modify: `paper/main_old_framing.tex` — old framing Discussion

- [ ] **Step 1: Rewrite Discussion for old framing (~2000 words)**

- 4.1 Neuroticism as transdiagnostic marker (confirmed across 3 universities)
- 4.2 Limited value of passive sensing (acknowledged but not central)
- 4.3 Simple models suffice
- 4.4 Limitations
- 4.5 Implications: personality screening as low-cost triage

- [ ] **Step 2: Write Abstract — New Framing (in main.tex, ~250 words)**

Structure: Background (sensing promise) → Gap (no head-to-head) → Method (3 studies, N=1559) → Results (personality wins 14/15, AUC 0.65-0.80 vs 0.53-0.60, 1/8 FDR-significant incremental) → Conclusion (questionnaires outperform sensing; recommend baseline reporting).

- [ ] **Step 3: Write Abstract — Old Framing (in main_old_framing.tex, ~250 words)**

Structure: Background (personality→MH) → Gap (few multi-study ML replications) → Method → Results (N #1 predictor, meta r=0.44-0.63, AUC 0.75-0.86) → Conclusion (personality robustly predicts MH across settings).

- [ ] **Step 4: Commit**

```bash
git add paper/main.tex paper/main_old_framing.tex
git commit -m "feat: write Discussion (old framing) + both Abstracts"
```

---

## Phase D: Final Polish

### Task 12: Create LaTeX tables from CSV data

**Files:**
- Create: `paper/tables.tex` — all tables as LaTeX, included from main.tex

- [ ] **Step 1: Generate Table 1 (Three-study overview)**

Columns: Study, University, Year, N, Sensing, BFI version, MH instruments, Sensing duration.

- [ ] **Step 2: Generate Table 2 (Grand synthesis: R² comparison with CI)**

From `grand_synthesis.csv`. Format: `R² [CI_lo, CI_hi]` for personality and sensing columns.

- [ ] **Step 3: Generate Table 3 (Meta-analysis)**

From `meta_analysis.csv`. Columns: Trait→Outcome, k, N, pooled r [95% CI], I², p.

- [ ] **Step 4: Generate Table 4 (Clinical classification with CI)**

From `clinical_classification.csv`. Format AUC [CI], Sensitivity [CI], Specificity [CI].

- [ ] **Step 5: Generate Table 5 (Incremental validity)**

From `incremental_validity.csv`. Include p_fdr column.

- [ ] **Step 6: Add `\input{tables}` before `\end{document}` in both main.tex files and commit**

```bash
git add paper/tables.tex paper/main.tex paper/main_old_framing.tex
git commit -m "feat: add formatted LaTeX tables from analysis results"
```

---

### Task 13: Final verification and cleanup

- [ ] **Step 1: Compile both versions**

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex main_old_framing && bibtex main_old_framing && pdflatex main_old_framing && pdflatex main_old_framing
```

- [ ] **Step 2: Check for undefined references, missing citations**

Review the .log files for warnings.

- [ ] **Step 3: Word count check**

```bash
texcount paper/main.tex || echo "texcount not found; install via: brew install texlive"
texcount paper/main_old_framing.tex || true
```
Target: 6,000-8,000 for JMIR, 8,000-12,000 for CHB.

- [ ] **Step 4: Final commit**

```bash
git add paper/
git commit -m "feat: complete manuscript — two framing versions ready for review"
```
