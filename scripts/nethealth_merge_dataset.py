#!/usr/bin/env python3
"""
Phase 12 Step 3: Merge NetHealth Features, Surveys, and GPA into Analysis Dataset

Combines:
  - Behavioral features (from nethealth_extract_features.py)
  - Survey scores (from nethealth_score_surveys.py)
  - GPA data (from nethealth_score_surveys.py)
  - PCA behavioral composites (computed here)

Input:  data/processed/nethealth/features/combined_features.parquet
        data/processed/nethealth/scores/survey_scores.parquet
        data/processed/nethealth/scores/gpa.parquet
Output: data/processed/nethealth/nethealth_analysis_dataset.parquet
        results/nethealth/tables/descriptive_stats.csv
        results/nethealth/figures/correlation_heatmap.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

project_root = Path(__file__).parent.parent

NH_FEATURE_DIR = project_root / 'data' / 'processed' / 'nethealth' / 'features'
NH_SCORE_DIR = project_root / 'data' / 'processed' / 'nethealth' / 'scores'
NH_OUTPUT_DIR = project_root / 'data' / 'processed' / 'nethealth'
NH_TABLE_DIR = project_root / 'results' / 'nethealth' / 'tables'
NH_FIGURE_DIR = project_root / 'results' / 'nethealth' / 'figures'

for d in [NH_TABLE_DIR, NH_FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

# NetHealth behavioral feature groups for PCA composites
NH_ACTIVITY_FEATURES = [
    'steps_mean', 'steps_std', 'light_active_min_mean',
    'fairly_active_min_mean', 'very_active_min_mean',
    'total_active_min_mean', 'active_ratio_mean',
]
NH_SLEEP_FEATURES = [
    'sleep_duration_mean', 'sleep_duration_std',
    'sleep_interruptions_mean', 'sleep_efficiency_mean',
    'sleep_regularity',
]
NH_COMMUNICATION_FEATURES = [
    'call_count_per_day', 'call_duration_per_day',
    'sms_count_per_day', 'total_unique_contacts',
    'total_comm_per_day',
]

PERSONALITY = ['extraversion', 'agreeableness', 'conscientiousness',
               'neuroticism', 'openness']
OUTCOMES = ['gpa_overall', 'cesd_total', 'loneliness_total',
            'self_esteem_total', 'stai_trait_total', 'bai_total']


def make_pca_composite(df: pd.DataFrame, feature_cols: list, name: str) -> pd.Series:
    """Compute first principal component from feature group."""
    available = [c for c in feature_cols if c in df.columns]
    if len(available) < 2:
        return pd.Series(np.nan, index=df.index, name=name)

    data = df[available].copy()
    valid_mask = data.notna().all(axis=1)

    if valid_mask.sum() < 10:
        return pd.Series(np.nan, index=df.index, name=name)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[valid_mask])
    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(scaled).ravel()

    result = pd.Series(np.nan, index=df.index, name=name)
    result[valid_mask] = pc1

    var_explained = pca.explained_variance_ratio_[0]
    print(f"    {name}: {len(available)} features, {valid_mask.sum()} users, "
          f"variance explained = {var_explained:.1%}")
    return result


def main():
    print("=" * 60)
    print("PHASE 12 STEP 3: MERGE NETHEALTH DATASET")
    print("=" * 60)

    # Load data
    print("\n  Loading features...")
    features = pd.read_parquet(NH_FEATURE_DIR / 'combined_features.parquet')
    print(f"    Features: {features.shape[0]} users × {features.shape[1]} columns")

    print("  Loading survey scores...")
    scores = pd.read_parquet(NH_SCORE_DIR / 'survey_scores.parquet')
    print(f"    Surveys: {scores.shape[0]} users × {scores.shape[1]} columns")

    print("  Loading GPA...")
    gpa = pd.read_parquet(NH_SCORE_DIR / 'gpa.parquet')
    print(f"    GPA: {gpa.shape[0]} users")

    # Merge all
    print("\n  Merging...")
    df = scores.merge(gpa, on='egoid', how='outer')
    df = df.merge(features, on='egoid', how='outer')
    print(f"    Merged: {df.shape[0]} users × {df.shape[1]} columns")

    # PCA composites
    print("\n  Computing PCA composites...")
    df['nh_activity_pc1'] = make_pca_composite(df, NH_ACTIVITY_FEATURES, 'nh_activity_pc1')
    df['nh_sleep_pc1'] = make_pca_composite(df, NH_SLEEP_FEATURES, 'nh_sleep_pc1')
    df['nh_communication_pc1'] = make_pca_composite(df, NH_COMMUNICATION_FEATURES, 'nh_communication_pc1')

    NH_BEHAVIOR_PC = ['nh_activity_pc1', 'nh_sleep_pc1', 'nh_communication_pc1']

    # Save
    df.to_parquet(NH_OUTPUT_DIR / 'nethealth_analysis_dataset.parquet', index=False)
    print(f"\n  Saved: nethealth_analysis_dataset.parquet ({df.shape[0]} × {df.shape[1]})")

    # Descriptive statistics
    print("\n  Computing descriptive statistics...")
    desc_cols = PERSONALITY + OUTCOMES + NH_BEHAVIOR_PC
    desc_cols = [c for c in desc_cols if c in df.columns]
    desc = df[desc_cols].describe().T
    desc['n_valid'] = df[desc_cols].notna().sum()
    desc.to_csv(NH_TABLE_DIR / 'descriptive_stats.csv')

    # Sample availability report
    print(f"\n{'─' * 60}")
    print("SAMPLE AVAILABILITY")
    print(f"{'─' * 60}")
    has_bfi = df['conscientiousness'].notna()
    has_gpa = df['gpa_overall'].notna()
    has_cesd = df['cesd_total'].notna()
    has_activity = df['nh_activity_pc1'].notna()
    has_sleep = df['nh_sleep_pc1'].notna()
    has_comm = df['nh_communication_pc1'].notna()

    print(f"  Total participants:           {len(df)}")
    print(f"  BFI-44 complete:              {has_bfi.sum()}")
    print(f"  GPA available:                {has_gpa.sum()}")
    print(f"  CES-D available:              {has_cesd.sum()}")
    print(f"  Fitbit Activity:              {has_activity.sum()}")
    print(f"  Fitbit Sleep:                 {has_sleep.sum()}")
    print(f"  Communication:                {has_comm.sum()}")
    print(f"  ──────────────────────────────────────")
    print(f"  BFI + GPA (core validation):  {(has_bfi & has_gpa).sum()}")
    print(f"  BFI + CES-D:                  {(has_bfi & has_cesd).sum()}")
    print(f"  BFI + GPA + any behavior:     {(has_bfi & has_gpa & (has_activity | has_sleep | has_comm)).sum()}")
    print(f"  All data:                     {(has_bfi & has_gpa & has_cesd & has_activity & has_sleep & has_comm).sum()}")

    # Correlation heatmap
    print("\n  Generating correlation heatmap...")
    corr_cols = PERSONALITY + ['gpa_overall', 'cesd_total', 'loneliness_total'] + NH_BEHAVIOR_PC
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr_data = df[corr_cols].dropna()

    if len(corr_data) >= 20:
        labels = {
            'extraversion': 'E', 'agreeableness': 'A', 'conscientiousness': 'C',
            'neuroticism': 'N', 'openness': 'O', 'gpa_overall': 'GPA',
            'cesd_total': 'CES-D', 'loneliness_total': 'Lonely',
            'nh_activity_pc1': 'Activity', 'nh_sleep_pc1': 'Sleep',
            'nh_communication_pc1': 'Comm',
        }
        corr_matrix = corr_data.rename(columns=labels).corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.5, ax=ax)
        ax.set_title(f'NetHealth Correlation Matrix (N={len(corr_data)})', fontsize=14)
        plt.tight_layout()
        fig.savefig(NH_FIGURE_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: results/nethealth/figures/correlation_heatmap.png")

    # Key correlations
    print(f"\n{'─' * 60}")
    print("KEY CORRELATIONS (Personality → GPA)")
    print(f"{'─' * 60}")
    bfi_gpa = df[has_bfi & has_gpa]
    from scipy import stats as sp_stats
    for trait in PERSONALITY:
        r, p = sp_stats.pearsonr(bfi_gpa[trait].values, bfi_gpa['gpa_overall'].values)
        sig = '*' if p < 0.05 else ''
        print(f"  {trait:20s}  r = {r:+.3f}  p = {p:.4f} {sig}")

    print(f"\n{'=' * 60}")
    print("PHASE 12 STEP 3 COMPLETE")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
