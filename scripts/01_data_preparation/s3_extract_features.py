#!/usr/bin/env python3
"""
Phase 13 Step 2: Extract GLOBEM Behavioral Features

All feature files are RAPIDS pre-aggregated (one row per participant-day,
14-day rolling windows). We select key interpretable features per modality
and aggregate to person-level (median across days).

Main analysis: ~18 selected features from individual modality files
Sensitivity: rapids.csv → person-level median → PCA

Input:  data/raw/globem/INS-W_{1..4}/FeatureData/*.csv
Output: data/processed/globem/features/combined_features.parquet
        data/processed/globem/features/rapids_features.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent

GB_DATA_DIR = project_root / 'data' / 'raw' / 'globem'
GB_FEAT_DIR = project_root / 'data' / 'processed' / 'globem' / 'features'
GB_FEAT_DIR.mkdir(parents=True, exist_ok=True)

COHORTS = ['INS-W_1', 'INS-W_2', 'INS-W_3', 'INS-W_4']
MIN_DAYS = 14  # Minimum days of data required

# ── Key features to extract (column substrings → short names) ──
# Each tuple: (file, column_substring, short_name)
# Using column substrings to match across the verbose RAPIDS names

STEPS_FEATURES = [
    ('avgsumsteps:14dhist', 'steps_avg'),
    ('stdsumsteps:14dhist', 'steps_std'),
    ('countepisodeactivebout:14dhist', 'active_bout_count'),
    ('sumdurationsedentarybout:14dhist', 'sedentary_duration'),
]

SLEEP_FEATURES = [
    ('avgdurationasleepmain:14dhist', 'sleep_duration_avg'),
    ('avgefficiencymain:14dhist', 'sleep_efficiency'),
    ('avgdurationtofallasleepmain:14dhist', 'sleep_onset_duration'),
    ('countepisodemain:14dhist', 'sleep_episode_count'),
]

CALL_FEATURES = [
    ('incoming_count:14dhist', 'call_incoming_count'),
    ('outgoing_count:14dhist', 'call_outgoing_count'),
    ('incoming_distinctcontacts:14dhist', 'call_incoming_contacts'),
    ('outgoing_distinctcontacts:14dhist', 'call_outgoing_contacts'),
]

SCREEN_FEATURES = [
    # Use only the base unlock features (not location-mapped ones)
    ('rapids_countepisodeunlock:14dhist', 'screen_unlock_count'),
    ('rapids_sumdurationunlock:14dhist', 'screen_duration_total'),
    ('rapids_avgdurationunlock:14dhist', 'screen_duration_avg'),
]

LOCATION_FEATURES = [
    ('barnett_hometime:14dhist', 'loc_hometime'),
    ('barnett_rog:14dhist', 'loc_radius_gyration'),
    ('barnett_siglocsvisited:14dhist', 'loc_unique_locations'),
    ('barnett_siglocentropy:14dhist', 'loc_entropy'),
]


def find_column(df_columns: list[str], substring: str) -> str | None:
    """Find the first column matching the substring."""
    # For screen features, we need exact matching to avoid location-mapped variants
    matches = [c for c in df_columns if substring in c]
    if not matches:
        return None
    # Prefer shortest match (most specific)
    return min(matches, key=len)


def extract_modality_features(
    modality_file: str,
    feature_specs: list[tuple[str, str]],
    cohort_dirs: list[Path],
) -> pd.DataFrame:
    """Extract and aggregate features from one modality across all cohorts.

    For each participant, compute median across all their person-days.
    """
    all_dfs = []
    for cohort_dir in cohort_dirs:
        path = cohort_dir / 'FeatureData' / modality_file
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        df = pd.read_csv(path)
        # Drop unnamed index
        if df.columns[0] == '' or df.columns[0].startswith('Unnamed'):
            df = df.drop(columns=[df.columns[0]])
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  {modality_file}: {len(combined)} person-days, {combined['pid'].nunique()} participants")

    # Map columns to short names
    col_map = {}
    for substring, short_name in feature_specs:
        col = find_column(list(combined.columns), substring)
        if col:
            col_map[col] = short_name
        else:
            print(f"    WARNING: no column matching '{substring}'")

    if not col_map:
        return pd.DataFrame()

    # Select and rename
    subset = combined[['pid'] + list(col_map.keys())].copy()
    subset = subset.rename(columns=col_map)

    # Count days per participant (for minimum threshold)
    day_counts = subset.groupby('pid').size().reset_index(name='n_days')

    # Aggregate to person-level: median across days
    person_level = subset.groupby('pid')[list(col_map.values())].median().reset_index()
    person_level = person_level.merge(day_counts, on='pid')

    # Filter by minimum days
    before = len(person_level)
    person_level = person_level[person_level['n_days'] >= MIN_DAYS].drop(columns=['n_days'])
    after = len(person_level)
    if before > after:
        print(f"    Filtered {before - after} participants with <{MIN_DAYS} days")

    return person_level


def extract_rapids_sensitivity(cohort_dirs: list[Path]) -> pd.DataFrame:
    """Extract person-level features from rapids.csv for sensitivity analysis.

    Strategy: aggregate to person-level median, drop columns with >50% missing,
    then save for PCA in merge step.
    """
    print("\n--- RAPIDS Sensitivity Features ---")
    all_dfs = []
    for cohort_dir in cohort_dirs:
        path = cohort_dir / 'FeatureData' / 'rapids.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.columns[0] == '' or df.columns[0].startswith('Unnamed'):
            df = df.drop(columns=[df.columns[0]])
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  rapids.csv: {len(combined)} person-days, {combined['pid'].nunique()} participants, {len(combined.columns)} columns")

    # Drop date, keep only numeric columns
    combined = combined.drop(columns=['date'], errors='ignore')
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in numeric_cols if c != 'pid']
    print(f"  Numeric feature columns: {len(feat_cols)} (dropped {len(combined.columns) - len(feat_cols) - 1} non-numeric)")

    # Person-level median
    person_level = combined.groupby('pid')[feat_cols].median().reset_index()

    # Drop columns with >50% missing
    n = len(person_level)
    missing_frac = person_level[feat_cols].isna().sum() / n
    keep_cols = missing_frac[missing_frac < 0.5].index.tolist()
    dropped = len(feat_cols) - len(keep_cols)
    print(f"  Dropped {dropped} columns with >50% missing, keeping {len(keep_cols)}")

    # Drop zero-variance columns
    variances = person_level[keep_cols].var()
    nonzero_cols = variances[variances > 0].index.tolist()
    dropped_zv = len(keep_cols) - len(nonzero_cols)
    if dropped_zv > 0:
        print(f"  Dropped {dropped_zv} zero-variance columns, keeping {len(nonzero_cols)}")

    result = person_level[['pid'] + nonzero_cols]
    print(f"  Final: {len(result)} participants × {len(nonzero_cols)} features")
    return result


def main():
    print("=" * 60)
    print("Phase 13 Step 2: GLOBEM Feature Extraction")
    print("=" * 60)

    cohort_dirs = [GB_DATA_DIR / c for c in COHORTS]

    # ── Main analysis: individual modality features ──
    print("\n--- Main Analysis Features ---")

    modalities = [
        ('steps.csv', STEPS_FEATURES),
        ('sleep.csv', SLEEP_FEATURES),
        ('call.csv', CALL_FEATURES),
        ('screen.csv', SCREEN_FEATURES),
        ('location.csv', LOCATION_FEATURES),
    ]

    all_features = []
    for filename, specs in modalities:
        feat = extract_modality_features(filename, specs, cohort_dirs)
        if not feat.empty:
            all_features.append(feat)

    # Merge all modalities on pid (outer join to keep all participants)
    combined = all_features[0]
    for feat in all_features[1:]:
        combined = combined.merge(feat, on='pid', how='outer')

    feat_cols = [c for c in combined.columns if c != 'pid']
    print(f"\n  Combined: {len(combined)} participants × {len(feat_cols)} features")

    # Report feature availability
    print(f"\n  Feature availability:")
    for c in feat_cols:
        n = combined[c].notna().sum()
        print(f"    {c:30s}: N={n:4d} ({n/len(combined)*100:.0f}%)")

    combined.to_parquet(GB_FEAT_DIR / 'combined_features.parquet', index=False)
    print(f"\n  Saved: {GB_FEAT_DIR / 'combined_features.parquet'}")

    # ── Sensitivity: RAPIDS features ──
    rapids = extract_rapids_sensitivity(cohort_dirs)
    rapids.to_parquet(GB_FEAT_DIR / 'rapids_features.parquet', index=False)
    print(f"  Saved: {GB_FEAT_DIR / 'rapids_features.parquet'}")


if __name__ == '__main__':
    main()
