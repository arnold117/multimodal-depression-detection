"""
Script to merge all feature modalities into combined feature matrix.

Outputs:
- data/processed/features/combined_features.parquet
- data/processed/features/feature_summary.csv
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np


def main():
    feature_dir = project_root / 'data' / 'processed' / 'features'

    print("Loading feature files...")
    print("=" * 60)

    # Load all feature files
    gps_features = pd.read_parquet(feature_dir / 'gps_mobility_features.parquet')
    app_features = pd.read_parquet(feature_dir / 'app_usage_features.parquet')
    comm_features = pd.read_parquet(feature_dir / 'communication_features.parquet')
    activity_features = pd.read_parquet(feature_dir / 'activity_features.parquet')

    print(f"GPS features: {len(gps_features)} users, {len(gps_features.columns)-1} features")
    print(f"App features: {len(app_features)} users, {len(app_features.columns)-1} features")
    print(f"Communication features: {len(comm_features)} users, {len(comm_features.columns)-1} features")
    print(f"Activity features: {len(activity_features)} users, {len(activity_features.columns)-1} features")

    # Merge all modalities
    print("\nMerging features...")
    combined = gps_features.merge(app_features, on='uid', how='outer')
    combined = combined.merge(comm_features, on='uid', how='outer')
    combined = combined.merge(activity_features, on='uid', how='outer')

    # Load labels
    labels = pd.read_csv(project_root / 'data' / 'processed' / 'labels' / 'item9_labels_pre.csv')
    combined = combined.merge(labels, on='uid', how='left')

    print(f"\nCombined: {len(combined)} users, {len(combined.columns)-1} features+labels")

    # Identify feature columns (exclude uid, labels, metadata)
    exclude_cols = ['uid', 'item9_score', 'item9_binary',
                   'gps_valid_days', 'gps_total_points', 'gps_points_per_day',
                   'app_valid_days', 'app_total_switches',
                   'comm_valid_days',
                   'activity_valid_days', 'activity_total_samples']

    feature_cols = [c for c in combined.columns if c not in exclude_cols]

    print(f"\nFeature columns (excluding metadata): {len(feature_cols)}")

    # Check missing data
    print("\n" + "=" * 60)
    print("MISSING DATA ANALYSIS")
    print("=" * 60)

    missing_summary = []
    for col in feature_cols:
        n_missing = combined[col].isnull().sum()
        pct_missing = (n_missing / len(combined)) * 100
        if n_missing > 0:
            missing_summary.append({
                'feature': col,
                'n_missing': n_missing,
                'pct_missing': pct_missing
            })

    if missing_summary:
        missing_df = pd.DataFrame(missing_summary).sort_values('pct_missing', ascending=False)
        print(f"\nFeatures with missing data ({len(missing_df)} features):")
        print(missing_df.to_string(index=False))
    else:
        print("\nNo missing data!")

    # Handle missing data: fill with 0 for communication features (users without comm data)
    print("\n" + "=" * 60)
    print("HANDLING MISSING DATA")
    print("=" * 60)

    comm_cols = [c for c in feature_cols if c.startswith(('call_', 'sms_', 'total_', 'communication_'))]

    print(f"\nFilling {len(comm_cols)} communication features with 0 for users without data...")
    for col in comm_cols:
        combined[col].fillna(0, inplace=True)

    # Check if any missing data remains
    remaining_missing = combined[feature_cols].isnull().sum().sum()
    print(f"Remaining missing values: {remaining_missing}")

    # Save combined features
    output_file = feature_dir / 'combined_features.parquet'
    combined.to_parquet(output_file, index=False)
    print(f"\nSaved combined features to: {output_file}")

    # Create feature summary
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)

    feature_summary = []
    for col in feature_cols:
        feature_summary.append({
            'feature': col,
            'mean': combined[col].mean(),
            'std': combined[col].std(),
            'min': combined[col].min(),
            'max': combined[col].max(),
            'n_missing': combined[col].isnull().sum()
        })

    summary_df = pd.DataFrame(feature_summary)
    summary_file = feature_dir / 'feature_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved feature summary to: {summary_file}")

    # Print final stats
    print("\n" + "=" * 60)
    print("FINAL FEATURE MATRIX")
    print("=" * 60)
    print(f"Users: {len(combined)}")
    print(f"Total features: {len(feature_cols)}")
    print(f"  GPS: {sum(1 for c in feature_cols if c.startswith(('location_', 'distance_', 'radius_', 'max_distance', 'home_', 'n_significant', 'movement_')))}")
    print(f"  App usage: {sum(1 for c in feature_cols if c.startswith(('app_', 'unique_apps', 'night_', 'weekend_')))}")
    print(f"  Communication: {sum(1 for c in feature_cols if c.startswith(('call_', 'sms_', 'total_', 'communication_')))}")
    print(f"  Activity: {sum(1 for c in feature_cols if c.startswith(('still_', 'moving_', 'sedentary_', 'activity_')))}")
    print(f"\nOutcome distribution:")
    print(f"  No ideation: {(combined['item9_binary'] == 0).sum()}")
    print(f"  With ideation: {(combined['item9_binary'] == 1).sum()}")


if __name__ == '__main__':
    main()
