#!/usr/bin/env python3
"""
Phase 13 Step 1: Score GLOBEM Survey Instruments

Reads 4 annual cohorts (INS-W_1 through INS-W_4) and pools them.

Personality (baseline PRE only):
  - BFI-10 → 5 trait scores (already pre-computed, range 2-10)

Mental health outcomes:
  - BDI-II → depression (from dep_endterm.csv, fallback post.csv)
  - STAI State → anxiety (from pre.csv; note: STAIS_PRE in W1, STAI_PRE in W2+)
  - PSS-10 → perceived stress (from pre.csv)
  - CESD-10 → depressive symptoms (from pre.csv, W2+ only; CESD-9 as fallback)
  - UCLA-10 → loneliness (from pre.csv)

Input:  data/raw/globem/INS-W_{1..4}/SurveyData/{pre,post,dep_endterm}.csv
Output: data/processed/globem/scores/survey_scores.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent

GB_DATA_DIR = project_root / 'data' / 'raw' / 'globem'
GB_SCORE_DIR = project_root / 'data' / 'processed' / 'globem' / 'scores'
GB_SCORE_DIR.mkdir(parents=True, exist_ok=True)

COHORTS = ['INS-W_1', 'INS-W_2', 'INS-W_3', 'INS-W_4']


def load_cohort_surveys(cohort: str) -> dict[str, pd.DataFrame]:
    """Load pre, post, and dep_endterm CSVs for one cohort."""
    base = GB_DATA_DIR / cohort / 'SurveyData'
    result = {}
    for name in ['pre', 'post', 'dep_endterm']:
        path = base / f'{name}.csv'
        if path.exists():
            df = pd.read_csv(path)
            # Drop unnamed index column if present
            if df.columns[0] == '' or df.columns[0].startswith('Unnamed'):
                df = df.drop(columns=[df.columns[0]])
            result[name] = df
    return result


def score_personality(pre: pd.DataFrame) -> pd.DataFrame:
    """Extract BFI-10 trait scores (already pre-computed in dataset).

    BFI-10 has 2 items per dimension → scores range from 2 to 10.
    We normalize to 1-5 scale (per-item mean) for comparability with BFI-44.
    """
    trait_map = {
        'extraversion': 'BFI10_extroversion_PRE',
        'agreeableness': 'BFI10_agreeableness_PRE',
        'conscientiousness': 'BFI10_conscientiousness_PRE',
        'neuroticism': 'BFI10_neuroticism_PRE',
        'openness': 'BFI10_openness_PRE',
    }

    rows = []
    for _, r in pre.iterrows():
        pid = r['pid']
        traits = {}
        n_missing = 0
        for trait_name, col_name in trait_map.items():
            val = r.get(col_name, np.nan)
            if pd.isna(val):
                n_missing += 1
                traits[trait_name] = np.nan
            else:
                # Normalize sum (2-10) to per-item mean (1-5)
                traits[trait_name] = val / 2.0
        if n_missing < 5:  # At least 1 trait available
            rows.append({'pid': pid, **traits})

    return pd.DataFrame(rows)


def score_depression(surveys: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Extract BDI-II depression score.

    Priority: dep_endterm.csv > post.csv (BDI2_POST) > pre.csv (BDI2_PRE).
    """
    pid_scores = {}

    # 1. dep_endterm (highest priority — end-of-term measure)
    if 'dep_endterm' in surveys:
        for _, r in surveys['dep_endterm'].iterrows():
            if pd.notna(r.get('BDI2')):
                pid_scores[r['pid']] = r['BDI2']

    # 2. post.csv fallback
    if 'post' in surveys:
        for _, r in surveys['post'].iterrows():
            if r['pid'] not in pid_scores and pd.notna(r.get('BDI2_POST')):
                pid_scores[r['pid']] = r['BDI2_POST']

    # 3. pre.csv fallback (W2+ only has BDI2_PRE)
    if 'pre' in surveys:
        for _, r in surveys['pre'].iterrows():
            if r['pid'] not in pid_scores and pd.notna(r.get('BDI2_PRE')):
                pid_scores[r['pid']] = r['BDI2_PRE']

    return pd.DataFrame([
        {'pid': pid, 'bdi2_total': score}
        for pid, score in pid_scores.items()
    ])


def score_mental_health(pre: pd.DataFrame) -> pd.DataFrame:
    """Extract STAI, PSS, CESD, UCLA from pre.csv (baseline).

    Handles naming inconsistency: STAIS_PRE (W1) vs STAI_PRE (W2+).
    """
    rows = []
    for _, r in pre.iterrows():
        pid = r['pid']
        row = {'pid': pid}

        # STAI State — handle naming inconsistency
        stai = r.get('STAIS_PRE', np.nan)
        if pd.isna(stai):
            stai = r.get('STAI_PRE', np.nan)
        row['stai_state'] = stai if pd.notna(stai) else np.nan

        # PSS-10
        row['pss_10'] = r.get('PSS_10items_PRE', np.nan)

        # CESD — prefer 10-item, fallback to 9-item
        cesd = r.get('CESD_10items_PRE', np.nan)
        if pd.isna(cesd):
            cesd = r.get('CESD_9items_PRE', np.nan)
        row['cesd_total'] = cesd if pd.notna(cesd) else np.nan

        # UCLA Loneliness
        row['ucla_loneliness'] = r.get('UCLA_10items_PRE', np.nan)

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Phase 13 Step 1: GLOBEM Survey Scoring")
    print("=" * 60)

    all_personality = []
    all_depression = []
    all_mental_health = []

    for cohort in COHORTS:
        print(f"\n--- {cohort} ---")
        surveys = load_cohort_surveys(cohort)

        if 'pre' not in surveys:
            print(f"  WARNING: no pre.csv found, skipping")
            continue

        pre = surveys['pre']
        cohort_num = COHORTS.index(cohort) + 1
        print(f"  Pre-survey N = {len(pre)}")

        # Personality (BFI-10)
        pers = score_personality(pre)
        pers['cohort'] = cohort_num
        all_personality.append(pers)
        print(f"  BFI-10 scored: N = {len(pers)}")

        # Depression (BDI-II)
        dep = score_depression(surveys)
        dep['cohort'] = cohort_num
        all_depression.append(dep)
        print(f"  BDI-II scored: N = {len(dep)}")

        # Other mental health (STAI, PSS, CESD, UCLA)
        mh = score_mental_health(pre)
        mh['cohort'] = cohort_num
        all_mental_health.append(mh)
        print(f"  Mental health scored: N = {len(mh)}")

    # Pool all cohorts
    personality = pd.concat(all_personality, ignore_index=True)
    depression = pd.concat(all_depression, ignore_index=True)
    mental_health = pd.concat(all_mental_health, ignore_index=True)

    # Merge into single dataset
    merged = personality.merge(depression[['pid', 'bdi2_total']], on='pid', how='outer')
    merged = merged.merge(mental_health[['pid', 'stai_state', 'pss_10', 'cesd_total', 'ucla_loneliness']],
                          on='pid', how='outer')

    # Fill cohort from any source
    if 'cohort_x' in merged.columns:
        merged['cohort'] = merged['cohort_x'].fillna(merged.get('cohort_y', np.nan))
        merged = merged.drop(columns=[c for c in merged.columns if c.startswith('cohort_')])

    print(f"\n{'=' * 60}")
    print(f"POOLED RESULTS")
    print(f"{'=' * 60}")
    print(f"Total participants: {len(merged)}")

    # Descriptive stats
    traits = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
    outcomes = ['bdi2_total', 'stai_state', 'pss_10', 'cesd_total', 'ucla_loneliness']

    print(f"\nPersonality (BFI-10, 1-5 scale):")
    for t in traits:
        n = merged[t].notna().sum()
        if n > 0:
            print(f"  {t:20s}: N={n:4d}, M={merged[t].mean():.2f}, SD={merged[t].std():.2f}")

    print(f"\nMental Health Outcomes:")
    for o in outcomes:
        n = merged[o].notna().sum()
        if n > 0:
            print(f"  {o:20s}: N={n:4d}, M={merged[o].mean():.2f}, SD={merged[o].std():.2f}")

    # Per-cohort counts
    print(f"\nPer-cohort personality N:")
    for c in range(1, 5):
        n = merged[merged['cohort'] == c][traits[0]].notna().sum()
        print(f"  Cohort {c}: N={n}")

    # Save
    merged.to_parquet(GB_SCORE_DIR / 'survey_scores.parquet', index=False)
    print(f"\nSaved: {GB_SCORE_DIR / 'survey_scores.parquet'}")
    print(f"Columns: {list(merged.columns)}")


if __name__ == '__main__':
    main()
