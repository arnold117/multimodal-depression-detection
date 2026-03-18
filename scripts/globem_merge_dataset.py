#!/usr/bin/env python3
"""
Phase 13 Step 3: Merge GLOBEM Features and Surveys into Analysis Dataset

Combines:
  - Behavioral features (from globem_extract_features.py)
  - Survey scores (from globem_score_surveys.py)
  - PCA behavioral composites (computed here)

Input:  data/processed/globem/features/combined_features.parquet
        data/processed/globem/scores/survey_scores.parquet
Output: data/processed/globem/globem_analysis_dataset.parquet
        results/globem/tables/descriptive_stats.csv
        results/globem/figures/correlation_heatmap.png
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

GB_FEATURE_DIR = project_root / 'data' / 'processed' / 'globem' / 'features'
GB_SCORE_DIR = project_root / 'data' / 'processed' / 'globem' / 'scores'
GB_OUTPUT_DIR = project_root / 'data' / 'processed' / 'globem'
GB_TABLE_DIR = project_root / 'results' / 'globem' / 'tables'
GB_FIGURE_DIR = project_root / 'results' / 'globem' / 'figures'

for d in [GB_TABLE_DIR, GB_FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

# GLOBEM behavioral feature groups for PCA composites
GB_ACTIVITY_FEATURES = [
    'steps_avg', 'steps_std', 'active_bout_count', 'sedentary_duration',
]
GB_SLEEP_FEATURES = [
    'sleep_duration_avg', 'sleep_efficiency', 'sleep_onset_duration',
    'sleep_episode_count',
]
GB_COMMUNICATION_FEATURES = [
    'call_incoming_count', 'call_outgoing_count',
    'call_incoming_contacts', 'call_outgoing_contacts',
]
GB_DIGITAL_FEATURES = [
    'screen_unlock_count', 'screen_duration_total', 'screen_duration_avg',
]
GB_MOBILITY_FEATURES = [
    'loc_hometime', 'loc_radius_gyration', 'loc_unique_locations', 'loc_entropy',
]

PERSONALITY = ['extraversion', 'agreeableness', 'conscientiousness',
               'neuroticism', 'openness']
OUTCOMES = ['bdi2_total', 'stai_state', 'pss_10', 'cesd_total', 'ucla_loneliness']


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
    scores = pca.fit_transform(scaled)

    result = pd.Series(np.nan, index=df.index, name=name)
    result[valid_mask] = scores.ravel()

    var_explained = pca.explained_variance_ratio_[0]
    print(f"  PCA {name}: {len(available)} features, "
          f"N={valid_mask.sum()}, variance explained={var_explained:.1%}")
    return result


def main():
    print("=" * 60)
    print("Phase 13 Step 3: Merge GLOBEM Dataset")
    print("=" * 60)

    # Load data
    scores = pd.read_parquet(GB_SCORE_DIR / 'survey_scores.parquet')
    features = pd.read_parquet(GB_FEATURE_DIR / 'combined_features.parquet')
    print(f"Surveys: {len(scores)} participants")
    print(f"Features: {len(features)} participants")

    # Merge on pid (outer to keep all)
    df = scores.merge(features, on='pid', how='outer')
    print(f"Merged: {len(df)} participants")

    # PCA composites
    print("\nPCA Composites:")
    df['activity_pc1'] = make_pca_composite(df, GB_ACTIVITY_FEATURES, 'activity_pc1')
    df['sleep_pc1'] = make_pca_composite(df, GB_SLEEP_FEATURES, 'sleep_pc1')
    df['communication_pc1'] = make_pca_composite(df, GB_COMMUNICATION_FEATURES, 'communication_pc1')
    df['digital_pc1'] = make_pca_composite(df, GB_DIGITAL_FEATURES, 'digital_pc1')
    df['mobility_pc1'] = make_pca_composite(df, GB_MOBILITY_FEATURES, 'mobility_pc1')

    BEHAVIOR_PC = ['activity_pc1', 'sleep_pc1', 'communication_pc1',
                   'digital_pc1', 'mobility_pc1']

    # Save dataset
    df.to_parquet(GB_OUTPUT_DIR / 'globem_analysis_dataset.parquet', index=False)
    print(f"\nSaved: {GB_OUTPUT_DIR / 'globem_analysis_dataset.parquet'}")
    print(f"Shape: {df.shape}")

    # Descriptive statistics
    stat_cols = PERSONALITY + OUTCOMES + BEHAVIOR_PC
    stats_rows = []
    for col in stat_cols:
        if col in df.columns:
            s = df[col].dropna()
            stats_rows.append({
                'Variable': col,
                'N': len(s),
                'Mean': s.mean(),
                'SD': s.std(),
                'Min': s.min(),
                'Max': s.max(),
            })
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(GB_TABLE_DIR / 'descriptive_stats.csv', index=False)
    print(f"\nDescriptive Statistics:")
    print(stats_df.to_string(index=False))

    # Correlation heatmap: personality + outcomes
    corr_cols = PERSONALITY + OUTCOMES
    available = [c for c in corr_cols if c in df.columns]
    corr_data = df[available].dropna()
    print(f"\nCorrelation heatmap N = {len(corr_data)}")

    if len(corr_data) > 10:
        corr = corr_data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, ax=ax, square=True)
        ax.set_title('GLOBEM: Personality × Mental Health Correlations')
        plt.tight_layout()
        fig.savefig(GB_FIGURE_DIR / 'correlation_heatmap.png', dpi=300,
                    bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {GB_FIGURE_DIR / 'correlation_heatmap.png'}")

    # Key correlations for quick check
    print(f"\nKey Personality-Outcome Correlations (pairwise):")
    for outcome in OUTCOMES:
        if outcome not in df.columns:
            continue
        for trait in PERSONALITY:
            subset = df[[trait, outcome]].dropna()
            if len(subset) > 10:
                r = subset[trait].corr(subset[outcome])
                print(f"  {trait:20s} × {outcome:20s}: r = {r:+.3f} (N={len(subset)})")


if __name__ == '__main__':
    main()
