#!/usr/bin/env python3
"""
Phase 12 Step 1: Score NetHealth Survey Instruments and Compute GPA

Scores (Wave 1 baseline):
  - Big Five Inventory (BFI-44) → 5 trait means (same instrument as StudentLife)
  - CES-D → depression total (0-60), uses pre-computed CESDOverall
  - Loneliness (SELSA-S 15 items) → total + 3 subscales
  - Self-Esteem (Rosenberg) → total (pre-computed)
  - STAI Trait Anxiety → total (pre-computed)
  - BAI → anxiety total (pre-computed)

Computes:
  - GPA from course-level registrar grades (first semester + cumulative)

Input:  data/raw/nethealth/BasicSurvey(3-6-20).csv
        data/raw/nethealth/CourseGrades(3-6-20).csv
Output: data/processed/nethealth/scores/survey_scores.parquet
        data/processed/nethealth/scores/gpa.parquet
        data/processed/nethealth/scores/bigfive_items.parquet
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent

# Reuse BFI scoring constants from StudentLife script
sys.path.insert(0, str(project_root / 'scripts'))
from score_surveys import BFI_TRAITS, BFI_REVERSE, cronbach_alpha

NH_DATA_DIR = project_root / 'data' / 'raw' / 'nethealth'
NH_SCORE_DIR = project_root / 'data' / 'processed' / 'nethealth' / 'scores'
NH_SCORE_DIR.mkdir(parents=True, exist_ok=True)

SURVEY_FILE = NH_DATA_DIR / 'BasicSurvey(3-6-20).csv'
GRADES_FILE = NH_DATA_DIR / 'CourseGrades(3-6-20).csv'

# NetHealth BFI uses 5-point text labels (different wording from StudentLife)
NH_BFI_LIKERT = {
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Neither Agree nor Disagree': 3,
    'Agree': 4,
    'Strongly Agree': 5,
}

# Loneliness (SELSA-S 15 items) uses 7-point scale
NH_LONELY_LIKERT = {
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Somewhat Disagree': 3,
    'Neither Agree nor Disagree': 4,
    'Somewhat Agree': 5,
    'Agree': 6,
    'Strongly Agree': 7,
}

# SELSA-S reverse items (positively worded → higher = LESS lonely)
# Items 3, 5, 6, 9, 10 are typically reverse-scored in SELSA-S
SELSA_REVERSE = {3, 5, 6, 9, 10}

# GPA grade point mapping
GRADE_POINTS = {
    'A': 4.0, 'A-': 3.67,
    'B+': 3.33, 'B': 3.0, 'B-': 2.67,
    'C+': 2.33, 'C': 2.0, 'C-': 1.67,
    'D': 1.0, 'F': 0.0,
}
# P (Pass), S (Satisfactory), U (Unsatisfactory), V (?) → exclude from GPA


# ──────────────────────────────────────────────────────────────────────
# BFI-44 (Wave 1)
# ──────────────────────────────────────────────────────────────────────

def score_big_five(survey: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score BFI-44 from Wave 1 data. Uses same BFI_TRAITS/BFI_REVERSE as StudentLife."""
    bfi_cols = [f'BigFive_{i}_1' for i in range(1, 45)]
    df = survey[['egoid'] + bfi_cols].copy()

    # Map text → numeric
    for col in bfi_cols:
        df[col] = df[col].map(NH_BFI_LIKERT)

    # Rename to bfi_1..bfi_44 for consistency
    rename = {f'BigFive_{i}_1': f'bfi_{i}' for i in range(1, 45)}
    df = df.rename(columns=rename)

    # Reverse code (same items as StudentLife — BFI-44 standard)
    for item_num in BFI_REVERSE:
        col = f'bfi_{item_num}'
        df[col] = 6 - df[col]

    # Save item-level
    item_df = df[['egoid'] + [f'bfi_{i}' for i in range(1, 45)]].copy()

    # Compute trait means (allow up to 2 missing per trait)
    scores = {'egoid': df['egoid'].values}
    for trait, items in BFI_TRAITS.items():
        cols = [f'bfi_{i}' for i in items]
        data = df[cols]
        n_missing = data.isnull().sum(axis=1)
        means = data.mean(axis=1)
        means[n_missing > 2] = np.nan
        scores[trait] = means.values

    return pd.DataFrame(scores), item_df


# ──────────────────────────────────────────────────────────────────────
# CES-D (Wave 1) — use pre-computed total
# ──────────────────────────────────────────────────────────────────────

def score_cesd(survey: pd.DataFrame) -> pd.DataFrame:
    """Use pre-computed CESDOverall_1 from NetHealth dataset."""
    df = survey[['egoid', 'CESDOverall_1']].copy()
    df = df.rename(columns={'CESDOverall_1': 'cesd_total'})
    return df


# ──────────────────────────────────────────────────────────────────────
# Loneliness — SELSA-S 15 items + 3 subscale scores (Wave 1)
# ──────────────────────────────────────────────────────────────────────

def score_loneliness(survey: pd.DataFrame) -> pd.DataFrame:
    """Score SELSA-S 15-item loneliness scale from Wave 1."""
    item_cols = [f'lonely_{i}_1' for i in range(1, 16)]
    df = survey[['egoid'] + item_cols + ['selsa_rom_1', 'selsa_fam_1', 'selsa_soc_1']].copy()

    # Map text → numeric (7-point scale)
    for col in item_cols:
        df[col] = df[col].map(NH_LONELY_LIKERT)

    # Reverse code
    for item_num in SELSA_REVERSE:
        col = f'lonely_{item_num}_1'
        if col in df.columns:
            df[col] = 8 - df[col]

    # Total score (sum of 15 items)
    scored_cols = [f'lonely_{i}_1' for i in range(1, 16)]
    n_missing = df[scored_cols].isnull().sum(axis=1)
    df['loneliness_total'] = df[scored_cols].sum(axis=1)
    df.loc[n_missing > 3, 'loneliness_total'] = np.nan

    # Rename subscale scores
    df = df.rename(columns={
        'selsa_rom_1': 'selsa_romantic',
        'selsa_fam_1': 'selsa_family',
        'selsa_soc_1': 'selsa_social',
    })

    return df[['egoid', 'loneliness_total', 'selsa_romantic', 'selsa_family', 'selsa_social']]


# ──────────────────────────────────────────────────────────────────────
# Pre-computed scales (Wave 1)
# ──────────────────────────────────────────────────────────────────────

def load_precomputed(survey: pd.DataFrame) -> pd.DataFrame:
    """Load pre-computed scale totals from Wave 1."""
    cols = {
        'SelfEsteem_1': 'self_esteem_total',
        'STAITraitTotal_1': 'stai_trait_total',
        'BAIsum_1': 'bai_total',
    }
    available = {k: v for k, v in cols.items() if k in survey.columns}
    df = survey[['egoid'] + list(available.keys())].copy()
    df = df.rename(columns=available)
    return df


# ──────────────────────────────────────────────────────────────────────
# GPA (from course-level registrar data)
# ──────────────────────────────────────────────────────────────────────

def compute_gpa(grades_file: Path) -> pd.DataFrame:
    """Compute first-semester and cumulative GPA from course grades."""
    df = pd.read_csv(grades_file)

    # Map letter grades to points, drop non-GPA grades (P, S, U, V)
    df['gpa_points'] = df['FinalGrade'].map(GRADE_POINTS)
    df_graded = df.dropna(subset=['gpa_points']).copy()

    # Identify first semester per student
    first_sem = df_graded.groupby('egoid')['AcademicPeriod'].min().reset_index()
    first_sem.columns = ['egoid', 'first_period']

    # First semester GPA
    df_first = df_graded.merge(first_sem, on='egoid')
    df_first = df_first[df_first['AcademicPeriod'] == df_first['first_period']]
    gpa_first = df_first.groupby('egoid')['gpa_points'].mean().reset_index()
    gpa_first.columns = ['egoid', 'gpa_first_semester']

    # Cumulative GPA (all semesters)
    gpa_cumulative = df_graded.groupby('egoid')['gpa_points'].mean().reset_index()
    gpa_cumulative.columns = ['egoid', 'gpa_overall']

    # Merge
    gpa = gpa_cumulative.merge(gpa_first, on='egoid', how='outer')

    return gpa


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 12 STEP 1: SCORE NETHEALTH SURVEYS")
    print("=" * 60)

    # Load full survey
    print("\n  Loading BasicSurvey...")
    survey = pd.read_csv(SURVEY_FILE)
    print(f"  {len(survey)} participants, {survey.shape[1]} columns")

    # 1. BFI-44
    print("\n[1/5] Big Five (BFI-44, Wave 1)...")
    big5, bfi_items = score_big_five(survey)
    n_bfi = big5.dropna(subset=['conscientiousness']).shape[0]
    print(f"  {n_bfi} participants with complete BFI scores")
    for trait in BFI_TRAITS:
        m = big5[trait].mean()
        s = big5[trait].std()
        print(f"    {trait:20s}  M={m:.2f}  SD={s:.2f}")

    # 2. CES-D
    print("\n[2/5] CES-D Depression (Wave 1, pre-computed)...")
    cesd = score_cesd(survey)
    n_cesd = cesd['cesd_total'].notna().sum()
    print(f"  {n_cesd} participants, M={cesd['cesd_total'].mean():.1f}, SD={cesd['cesd_total'].std():.1f}")

    # 3. Loneliness
    print("\n[3/5] Loneliness (SELSA-S 15 items, Wave 1)...")
    lonely = score_loneliness(survey)
    n_lonely = lonely['loneliness_total'].notna().sum()
    print(f"  {n_lonely} participants, M={lonely['loneliness_total'].mean():.1f}, SD={lonely['loneliness_total'].std():.1f}")

    # 4. Pre-computed scales
    print("\n[4/5] Pre-computed scales (Wave 1)...")
    precomp = load_precomputed(survey)
    for col in precomp.columns:
        if col == 'egoid':
            continue
        n = precomp[col].notna().sum()
        print(f"  {col:25s}  N={n}, M={precomp[col].mean():.1f}, SD={precomp[col].std():.1f}")

    # 5. GPA
    print("\n[5/5] GPA (from registrar grades)...")
    gpa = compute_gpa(GRADES_FILE)
    n_gpa = len(gpa)
    print(f"  {n_gpa} students with grades")
    print(f"  Cumulative GPA:     M={gpa['gpa_overall'].mean():.2f}, SD={gpa['gpa_overall'].std():.2f}, range=[{gpa['gpa_overall'].min():.2f}, {gpa['gpa_overall'].max():.2f}]")
    print(f"  First-semester GPA: M={gpa['gpa_first_semester'].mean():.2f}, SD={gpa['gpa_first_semester'].std():.2f}")

    # Merge all survey scores
    scores = big5
    for df in [cesd, lonely, precomp]:
        scores = scores.merge(df, on='egoid', how='outer')

    # Save
    scores.to_parquet(NH_SCORE_DIR / 'survey_scores.parquet', index=False)
    gpa.to_parquet(NH_SCORE_DIR / 'gpa.parquet', index=False)
    bfi_items.to_parquet(NH_SCORE_DIR / 'bigfive_items.parquet', index=False)

    print(f"\n  Saved: data/processed/nethealth/scores/survey_scores.parquet ({len(scores)} rows)")
    print(f"  Saved: data/processed/nethealth/scores/gpa.parquet ({n_gpa} rows)")
    print(f"  Saved: data/processed/nethealth/scores/bigfive_items.parquet")

    # BFI Reliability
    print("\n" + "─" * 60)
    print("BIG FIVE RELIABILITY (Cronbach's α)")
    print("─" * 60)
    for trait, items in BFI_TRAITS.items():
        cols = [f'bfi_{i}' for i in items]
        alpha = cronbach_alpha(bfi_items[cols])
        print(f"  {trait:20s}  α = {alpha:.3f}")

    # Sample overlap
    print("\n" + "─" * 60)
    print("SAMPLE OVERLAP")
    print("─" * 60)
    bfi_ids = set(big5.dropna(subset=['conscientiousness'])['egoid'])
    gpa_ids = set(gpa['egoid'])
    cesd_ids = set(cesd.dropna(subset=['cesd_total'])['egoid'])
    print(f"  BFI complete:        {len(bfi_ids)}")
    print(f"  GPA available:       {len(gpa_ids)}")
    print(f"  CES-D available:     {len(cesd_ids)}")
    print(f"  BFI ∩ GPA:           {len(bfi_ids & gpa_ids)}")
    print(f"  BFI ∩ CES-D:         {len(bfi_ids & cesd_ids)}")
    print(f"  BFI ∩ GPA ∩ CES-D:   {len(bfi_ids & gpa_ids & cesd_ids)}")

    print("\n" + "=" * 60)
    print("PHASE 12 STEP 1 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
