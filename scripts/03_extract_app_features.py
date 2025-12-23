"""
Script to extract app usage features for all users.

Outputs:
- data/processed/features/app_usage_features.parquet
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
from features.app_usage import AppUsageExtractor


def main():
    # Paths
    data_dir = project_root / 'data' / 'raw' / 'dataset' / 'app_usage'
    timelines_file = project_root / 'data' / 'interim' / 'user_timelines' / 'user_timelines.json'
    output_dir = project_root / 'data' / 'processed' / 'features'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load valid users
    valid_users_file = project_root / 'data' / 'interim' / 'user_timelines' / 'valid_users.csv'
    valid_users = pd.read_csv(valid_users_file)
    user_ids = valid_users['user_id'].tolist()

    print(f"Extracting app usage features for {len(user_ids)} users...")
    print("=" * 60)

    # Initialize extractor
    extractor = AppUsageExtractor(
        data_dir=data_dir,
        timelines_file=timelines_file
    )

    # Extract features
    features_df = extractor.extract_all_users(user_ids)

    print("=" * 60)
    print(f"\nFeature extraction complete:")
    print(f"  Users with features: {len(features_df)}/{len(user_ids)}")
    print(f"  Features per user: {len(features_df.columns) - 1}")

    # Save to parquet
    output_file = output_dir / 'app_usage_features.parquet'
    features_df.to_parquet(output_file, index=False)
    print(f"\nSaved to: {output_file}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY STATISTICS")
    print("=" * 60)

    feature_cols = [c for c in features_df.columns if c not in ['uid', 'app_valid_days', 'app_total_switches']]

    for col in feature_cols:
        print(f"\n{col}:")
        print(f"  Mean: {features_df[col].mean():.4f}")
        print(f"  Std:  {features_df[col].std():.4f}")
        print(f"  Min:  {features_df[col].min():.4f}")
        print(f"  Max:  {features_df[col].max():.4f}")


if __name__ == '__main__':
    main()
