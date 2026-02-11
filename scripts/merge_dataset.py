#!/usr/bin/env python3
"""
Step 3: Merge Features, Survey Scores, and GPA into Analysis Dataset

Combines:
  - Behavioral features (from extract_features.py)
  - Survey scores (from score_surveys.py)
  - GPA data (from score_surveys.py)
  - PCA behavioral composites (computed here)

Input:  data/processed/features/combined_features.parquet
        data/processed/scores/survey_scores.parquet
        data/processed/scores/gpa.parquet
Output: data/processed/analysis_dataset.parquet
        results/tables/descriptive_stats.csv
        results/figures/correlation_heatmap.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats as sp_stats

project_root = Path(__file__).parent.parent

FEATURE_DIR = project_root / 'data' / 'processed' / 'features'
SCORE_DIR = project_root / 'data' / 'processed' / 'scores'
OUTPUT_DIR = project_root / 'data' / 'processed'
TABLE_DIR = project_root / 'results' / 'tables'
FIGURE_DIR = project_root / 'results' / 'figures'

for d in [TABLE_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

# Behavioral feature groups for PCA composites
GPS_FEATURES = [
    'location_variance_mean', 'distance_traveled_mean',
    'radius_of_gyration_mean', 'max_distance_from_home',
    'home_stay_ratio', 'n_significant_locations', 'movement_entropy',
]
APP_FEATURES = [
    'app_switches_mean', 'unique_apps_mean',
    'night_usage_ratio', 'weekend_weekday_ratio',
]
COMM_FEATURES = [
    'call_count_mean', 'call_duration_mean',
    'sms_count_mean', 'total_unique_contacts',
]
ACTIVITY_FEATURES = [
    'moving_ratio_mean', 'still_ratio_mean',
    'activity_transitions_mean',
]
PHONELOCK_FEATURES = [
    'unlock_count_mean', 'session_duration_mean',
    'night_unlock_ratio', 'screen_time_hours_mean',
]
BLUETOOTH_FEATURES = [
    'bt_unique_devices_mean', 'bt_scan_count_mean', 'bt_device_entropy',
]
CONVERSATION_FEATURES = [
    'convo_count_mean', 'convo_duration_mean', 'convo_night_ratio',
]
AUDIO_FEATURES = [
    'audio_silence_ratio', 'audio_voice_ratio', 'audio_noise_ratio',
]


def make_behavioral_composites(df: pd.DataFrame) -> pd.DataFrame:
    """PCA-based composite scores for each behavioral modality."""
    composites = df[['uid']].copy()
    scaler = StandardScaler()

    modalities = {
        'mobility_pc1': GPS_FEATURES,
        'digital_pc1': APP_FEATURES,
        'social_pc1': COMM_FEATURES,
        'activity_pc1': ACTIVITY_FEATURES,
        'screen_pc1': PHONELOCK_FEATURES,
        'proximity_pc1': BLUETOOTH_FEATURES,
        'face2face_pc1': CONVERSATION_FEATURES,
        'audio_pc1': AUDIO_FEATURES,
    }

    for name, features in modalities.items():
        available = [f for f in features if f in df.columns]
        data = df[available].fillna(df[available].median())
        scaled = scaler.fit_transform(data)
        pca = PCA(n_components=1, random_state=42)
        composites[name] = pca.fit_transform(scaled).flatten()
        print(f"    {name}: {pca.explained_variance_ratio_[0]:.1%} variance "
              f"({len(available)} features)")

    return composites


def make_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct interaction and composite features from PCA components."""
    scaler = StandardScaler()

    # Interaction terms
    df['digital_x_mobility'] = df['digital_pc1'] * df['mobility_pc1']
    df['screen_x_activity'] = df['screen_pc1'] * df['activity_pc1']

    # Social isolation index: high = more socially engaged, less isolated
    iso_cols = ['digital_pc1', 'mobility_pc1', 'proximity_pc1', 'face2face_pc1']
    available = [c for c in iso_cols if c in df.columns and df[c].notna().sum() > 0]
    if len(available) >= 3:
        z = pd.DataFrame(scaler.fit_transform(df[available].fillna(0)),
                         columns=available, index=df.index)
        sign = {c: -1 if c == 'digital_pc1' else 1 for c in available}
        df['social_isolation_index'] = sum(sign[c] * z[c] for c in available) / len(available)

    # Night behavior index
    if 'night_unlock_ratio' in df.columns and 'audio_silence_ratio' in df.columns:
        df['night_behavior_index'] = df['night_unlock_ratio'] * df['audio_silence_ratio']

    new_feats = [c for c in ['digital_x_mobility', 'screen_x_activity',
                              'social_isolation_index', 'night_behavior_index']
                 if c in df.columns]
    print(f"  Interaction features added: {new_feats}")
    return df


def descriptive_stats(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    rows = []
    for col in cols:
        d = df[col].dropna()
        rows.append({
            'Variable': col, 'N': len(d),
            'Mean': d.mean(), 'SD': d.std(),
            'Min': d.min(), 'Max': d.max(),
            'Skew': d.skew(), 'Kurtosis': d.kurtosis(),
        })
    return pd.DataFrame(rows)


def correlation_heatmap(df: pd.DataFrame, cols: list, output_path: Path):
    short = {
        'extraversion': 'E', 'agreeableness': 'A', 'conscientiousness': 'C',
        'neuroticism': 'N', 'openness': 'O',
        'gpa_overall': 'GPA', 'gpa_13s': 'GPA13', 'cs65_grade': 'CS65',
        'phq9_total': 'PHQ9', 'pss_total': 'PSS',
        'loneliness_total': 'Lonely', 'flourishing_total': 'Flourish',
        'panas_positive': 'PA', 'panas_negative': 'NA',
        'mobility_pc1': 'Mobility', 'digital_pc1': 'Digital',
        'social_pc1': 'Social', 'activity_pc1': 'Activity',
        'screen_pc1': 'Screen', 'proximity_pc1': 'Proximity',
        'face2face_pc1': 'Face2Face', 'audio_pc1': 'Audio',
    }

    corr = df[cols].corr()
    labels = [short.get(c, c[:10]) for c in cols]

    # Significance annotation
    annot = corr.round(2).astype(str)
    n = len(df)
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i != j:
                mask = df[[c1, c2]].dropna().index
                if len(mask) >= 3:
                    _, p = sp_stats.pearsonr(df.loc[mask, c1], df.loc[mask, c2])
                    if p < 0.001:
                        annot.iloc[i, j] += '***'
                    elif p < 0.01:
                        annot.iloc[i, j] += '**'
                    elif p < 0.05:
                        annot.iloc[i, j] += '*'

    fig, ax = plt.subplots(figsize=(14, 12))
    mask_tri = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask_tri, annot=annot, fmt='',
                xticklabels=labels, yticklabels=labels,
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax, annot_kws={'size': 7})
    ax.set_title('Correlation Matrix (*p<.05, **p<.01, ***p<.001)', fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def main():
    print("=" * 60)
    print("STEP 3: MERGE INTO ANALYSIS DATASET")
    print("=" * 60)

    # Load
    print("\n[1/4] Loading data sources...")
    features = pd.read_parquet(FEATURE_DIR / 'combined_features.parquet')
    surveys = pd.read_parquet(SCORE_DIR / 'survey_scores.parquet')
    gpa = pd.read_parquet(SCORE_DIR / 'gpa.parquet')
    print(f"  Behavioral features: {len(features)} users × {len(features.columns)-1} features")
    print(f"  Survey scores:       {len(surveys)} users × {len(surveys.columns)-1} measures")
    print(f"  GPA data:            {len(gpa)} users")

    # Behavioral composites
    print("\n[2/4] Computing behavioral PCA composites...")
    composites = make_behavioral_composites(features)

    # Merge (inner join on GPA — our study requires academic data)
    print("\n[3/4] Merging datasets...")
    merged = gpa.merge(surveys, on='uid', how='inner')
    merged = merged.merge(composites, on='uid', how='left')
    merged = merged.merge(features, on='uid', how='left')

    # Interaction features
    print("\n  Computing interaction features...")
    merged = make_interaction_features(merged)
    print(f"  Final: {len(merged)} participants × {len(merged.columns)} variables")
    print(f"  Participants: {sorted(merged['uid'].tolist())}")

    # Check missing
    key_cols = ['extraversion', 'agreeableness', 'conscientiousness',
                'neuroticism', 'openness', 'gpa_overall',
                'phq9_total', 'pss_total', 'loneliness_total',
                'flourishing_total', 'panas_positive', 'panas_negative',
                'mobility_pc1', 'digital_pc1', 'social_pc1', 'activity_pc1',
                'screen_pc1', 'proximity_pc1', 'face2face_pc1', 'audio_pc1']
    missing = {c: merged[c].isnull().sum() for c in key_cols if merged[c].isnull().any()}
    if missing:
        print(f"  Missing: {missing}")

    # Save
    merged.to_parquet(OUTPUT_DIR / 'analysis_dataset.parquet', index=False)
    print(f"\n  Output: data/processed/analysis_dataset.parquet")

    # Descriptive stats & correlation
    print("\n[4/4] Descriptive statistics & correlation heatmap...")
    available = [c for c in key_cols if c in merged.columns]
    desc = descriptive_stats(merged, available)
    desc.to_csv(TABLE_DIR / 'descriptive_stats.csv', index=False)
    print(f"  Saved: descriptive_stats.csv")
    print(desc.to_string(index=False))

    correlation_heatmap(merged, available, FIGURE_DIR / 'correlation_heatmap.png')

    print("\n" + "=" * 60)
    print("STEP 3 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
