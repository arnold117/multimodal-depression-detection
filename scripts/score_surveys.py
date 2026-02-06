#!/usr/bin/env python3
"""
Step 2: Score Survey Instruments and Load Academic Data

Scores:
  - Big Five Inventory (BFI-44) → 5 trait means
  - PHQ-9 → depression total (0-27)
  - PSS-10 → perceived stress total (0-40)
  - UCLA Loneliness Scale → loneliness total (20-80)
  - Flourishing Scale → well-being total (8-56)
  - PANAS → positive/negative affect means

Loads:
  - GPA data (overall, 13-course, CS65)

Input:  data/raw/dataset/survey/, data/raw/dataset/education/
Output: data/processed/scores/survey_scores.parquet
        data/processed/scores/gpa.parquet
        data/processed/scores/bigfive_items.parquet (item-level for reliability)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent

SURVEY_DIR = project_root / 'data' / 'raw' / 'dataset' / 'survey'
EDUCATION_DIR = project_root / 'data' / 'raw' / 'dataset' / 'education'
SCORE_DIR = project_root / 'data' / 'processed' / 'scores'

SCORE_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# BFI-44
# ──────────────────────────────────────────────────────────────────────

LIKERT_MAP = {
    'Disagree Strongly': 1, 'Disagree strongly': 1,
    'Disagree a little': 2,
    'Neither agree nor disagree': 3,
    'Agree a little': 4,
    'Agree strongly': 5, 'Agree Strongly': 5,
}

BFI_TRAITS = {
    'extraversion':       [1, 6, 11, 16, 21, 26, 31, 36],
    'agreeableness':      [2, 7, 12, 17, 22, 27, 32, 37, 42],
    'conscientiousness':  [3, 8, 13, 18, 23, 28, 33, 38, 43],
    'neuroticism':        [4, 9, 14, 19, 24, 29, 34, 39],
    'openness':           [5, 10, 15, 20, 25, 30, 35, 40, 41, 44],
}
BFI_REVERSE = {2, 6, 8, 9, 12, 18, 21, 23, 24, 27, 31, 35, 37, 41, 43}


def score_big_five() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score BFI-44. Returns (trait_scores, item_level_data)."""
    df = pd.read_csv(SURVEY_DIR / 'BigFive.csv')
    df = df[df['type'] == 'pre'].copy()
    item_cols = df.columns[2:]
    assert len(item_cols) == 44

    rename = {old: f'bfi_{i+1}' for i, old in enumerate(item_cols)}
    df = df.rename(columns=rename)

    for col in rename.values():
        df[col] = df[col].map(LIKERT_MAP)

    for item_num in BFI_REVERSE:
        col = f'bfi_{item_num}'
        if col in df.columns:
            df[col] = 6 - df[col]

    # Save item-level (for reliability checks)
    item_df = df[['uid'] + [f'bfi_{i}' for i in range(1, 45)]].copy()

    # Compute trait means (allow up to 2 missing per trait)
    scores = {'uid': df['uid'].values}
    for trait, items in BFI_TRAITS.items():
        cols = [f'bfi_{i}' for i in items]
        data = df[cols]
        n_missing = data.isnull().sum(axis=1)
        means = data.mean(axis=1)
        means[n_missing > 2] = np.nan
        scores[trait] = means.values

    return pd.DataFrame(scores), item_df


# ──────────────────────────────────────────────────────────────────────
# PHQ-9
# ──────────────────────────────────────────────────────────────────────

PHQ9_MAP = {
    'Not at all': 0, 'Several days': 1,
    'More than half the days': 2, 'Nearly every day': 3,
}


def score_phq9() -> pd.DataFrame:
    df = pd.read_csv(SURVEY_DIR / 'PHQ-9.csv')
    df = df[df['type'] == 'pre'].copy()
    item_cols = df.columns[2:11]
    for col in item_cols:
        df[col] = df[col].map(PHQ9_MAP)
    df['phq9_total'] = df[item_cols].sum(axis=1)
    return df[['uid', 'phq9_total']]


# ──────────────────────────────────────────────────────────────────────
# PSS-10
# ──────────────────────────────────────────────────────────────────────

PSS_MAP = {
    'Never': 0, 'Almost never': 1,
    'Sometime': 2, 'Sometimes': 2,
    'Fairly often': 3, 'Very often': 4,
}
PSS_REVERSE = {4, 5, 7, 8}


def score_pss() -> pd.DataFrame:
    df = pd.read_csv(SURVEY_DIR / 'PerceivedStressScale.csv')
    df = df[df['type'] == 'pre'].copy()
    item_cols = df.columns[2:12]
    for i, col in enumerate(item_cols):
        df[col] = df[col].map(PSS_MAP)
        if (i + 1) in PSS_REVERSE:
            df[col] = 4 - df[col]
    df['pss_total'] = df[item_cols].sum(axis=1)
    return df[['uid', 'pss_total']]


# ──────────────────────────────────────────────────────────────────────
# UCLA Loneliness
# ──────────────────────────────────────────────────────────────────────

LONELY_MAP = {'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4}
LONELY_REVERSE = {1, 5, 6, 9, 10, 15, 16, 19, 20}


def score_loneliness() -> pd.DataFrame:
    df = pd.read_csv(SURVEY_DIR / 'LonelinessScale.csv')
    df = df[df['type'] == 'pre'].copy()
    item_cols = df.columns[2:22]
    for i, col in enumerate(item_cols):
        df[col] = df[col].map(LONELY_MAP)
        if (i + 1) in LONELY_REVERSE:
            df[col] = 5 - df[col]
    df['loneliness_total'] = df[item_cols].sum(axis=1)
    return df[['uid', 'loneliness_total']]


# ──────────────────────────────────────────────────────────────────────
# Flourishing Scale
# ──────────────────────────────────────────────────────────────────────

def score_flourishing() -> pd.DataFrame:
    df = pd.read_csv(SURVEY_DIR / 'FlourishingScale.csv')
    df = df[df['type'] == 'pre'].copy()
    item_cols = df.columns[2:10]
    for col in item_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['flourishing_total'] = df[item_cols].sum(axis=1)
    return df[['uid', 'flourishing_total']]


# ──────────────────────────────────────────────────────────────────────
# PANAS
# ──────────────────────────────────────────────────────────────────────

def score_panas() -> pd.DataFrame:
    df = pd.read_csv(SURVEY_DIR / 'panas.csv')
    df = df[df['type'] == 'pre'].copy()

    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    pa_items = ['Interested', 'Strong', 'Enthusiastic', 'Proud',
                'Alert', 'Inspired', 'Determined ', 'Attentive', 'Active ']
    na_items = ['Distressed', 'Upset', 'Guilty', 'Scared',
                'Hostile ', 'Irritable', 'Nervous', 'Jittery', 'Afraid ']

    pa_cols = [c for c in df.columns if c.strip() in [p.strip() for p in pa_items]]
    na_cols = [c for c in df.columns if c.strip() in [n.strip() for n in na_items]]

    df['panas_positive'] = df[pa_cols].mean(axis=1)
    df['panas_negative'] = df[na_cols].mean(axis=1)
    return df[['uid', 'panas_positive', 'panas_negative']]


# ──────────────────────────────────────────────────────────────────────
# GPA
# ──────────────────────────────────────────────────────────────────────

def load_gpa() -> pd.DataFrame:
    df = pd.read_csv(EDUCATION_DIR / 'grades.csv')
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        'gpa all': 'gpa_overall',
        'gpa 13s': 'gpa_13s',
        'cs 65': 'cs65_grade',
    })
    return df[['uid', 'gpa_overall', 'gpa_13s', 'cs65_grade']]


# ──────────────────────────────────────────────────────────────────────
# Reliability
# ──────────────────────────────────────────────────────────────────────

def cronbach_alpha(items: pd.DataFrame) -> float:
    items = items.dropna()
    k = items.shape[1]
    if k < 2 or len(items) < 3:
        return np.nan
    item_vars = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 2: SCORE SURVEYS AND LOAD ACADEMIC DATA")
    print("=" * 60)

    # Score all instruments
    print("\n[1/7] Big Five (BFI-44)...")
    big5, bfi_items = score_big_five()
    n = big5.dropna().shape[0]
    print(f"  {n} participants scored")

    print("\n[2/7] PHQ-9 (Depression)...")
    phq9 = score_phq9()
    print(f"  {len(phq9)} participants, range: {phq9['phq9_total'].min()}-{phq9['phq9_total'].max()}")

    print("\n[3/7] PSS-10 (Stress)...")
    pss = score_pss()
    print(f"  {len(pss)} participants, range: {pss['pss_total'].min()}-{pss['pss_total'].max()}")

    print("\n[4/7] UCLA Loneliness...")
    lonely = score_loneliness()
    print(f"  {len(lonely)} participants, range: {lonely['loneliness_total'].min()}-{lonely['loneliness_total'].max()}")

    print("\n[5/7] Flourishing Scale...")
    flourish = score_flourishing()
    print(f"  {len(flourish)} participants, range: {flourish['flourishing_total'].min():.0f}-{flourish['flourishing_total'].max():.0f}")

    print("\n[6/7] PANAS (Affect)...")
    panas = score_panas()
    print(f"  {len(panas)} participants, PA: {panas['panas_positive'].mean():.2f}, NA: {panas['panas_negative'].mean():.2f}")

    print("\n[7/7] GPA...")
    gpa = load_gpa()
    print(f"  {len(gpa)} participants, GPA range: {gpa['gpa_overall'].min():.2f}-{gpa['gpa_overall'].max():.2f}")

    # Merge all survey scores
    survey = big5
    for df in [phq9, pss, lonely, flourish, panas]:
        survey = survey.merge(df, on='uid', how='outer')

    # Save
    survey.to_parquet(SCORE_DIR / 'survey_scores.parquet', index=False)
    gpa.to_parquet(SCORE_DIR / 'gpa.parquet', index=False)
    bfi_items.to_parquet(SCORE_DIR / 'bigfive_items.parquet', index=False)

    print(f"\n  Saved: data/processed/scores/survey_scores.parquet ({len(survey)} participants)")
    print(f"  Saved: data/processed/scores/gpa.parquet ({len(gpa)} participants)")
    print(f"  Saved: data/processed/scores/bigfive_items.parquet (item-level)")

    # Reliability report
    print("\n" + "─" * 60)
    print("BIG FIVE RELIABILITY (Cronbach's α)")
    print("─" * 60)
    for trait, items in BFI_TRAITS.items():
        cols = [f'bfi_{i}' for i in items]
        alpha = cronbach_alpha(bfi_items[cols])
        print(f"  {trait:20s}  α = {alpha:.3f}")

    print("\n" + "=" * 60)
    print("STEP 2 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
