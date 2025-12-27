#!/usr/bin/env python3
"""
Script 01: Preprocess PHQ-9 Data

This script performs Phase 1 preprocessing:
1. Load and encode PHQ-9 survey responses
2. Create outcome labels (Item #9 and total score)
3. Extract temporal alignment for behavioral data

Usage:
    python scripts/01_preprocess_phq9.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import PHQ9Processor, TemporalAligner


def main():
    """
    Main preprocessing pipeline.
    """
    print("="*80)
    print("PHASE 1: PHQ-9 DATA PREPROCESSING")
    print("="*80)

    # Configuration
    data_root = "data/raw/dataset"
    phq9_path = f"{data_root}/survey/PHQ-9.csv"
    labels_output = "data/processed/labels"
    timeline_output = "data/interim/user_timelines"

    # Step 1: Process PHQ-9 labels
    print("\n[Step 1/2] Processing PHQ-9 Labels...")
    print("-" * 80)

    processor = PHQ9Processor(phq9_path)
    processor.load_data()

    # Save pre-assessment labels (primary)
    print("\nPRE-ASSESSMENT (Primary Outcome):")
    processor.save_labels(labels_output, assessment_type='pre')

    # Save post-assessment labels (secondary)
    print("\n" + "="*80)
    print("POST-ASSESSMENT (Secondary/Longitudinal Analysis):")
    processor.save_labels(labels_output, assessment_type='post')

    # Step 2: Extract temporal alignment
    print("\n" + "="*80)
    print("[Step 2/2] Extracting Temporal Alignment...")
    print("-" * 80)

    # Get user IDs from PHQ-9
    import pandas as pd
    phq9_df = pd.read_csv(phq9_path)
    user_ids = phq9_df[phq9_df['type'] == 'pre']['uid'].unique().tolist()
    print(f"Found {len(user_ids)} users with pre-assessment")

    # Extract timelines
    aligner = TemporalAligner(data_root)
    timelines = aligner.get_all_user_timelines(user_ids)

    # Save timelines
    aligner.save_timelines(timelines, f"{timeline_output}/user_timelines.json")

    # Generate quality report
    print("\n" + "="*80)
    print("DATA QUALITY REPORT")
    print("="*80)
    quality_report = aligner.get_data_quality_report(timelines)
    print("\nPer-user coverage statistics:")
    print(quality_report[['user_id', 'total_days', 'n_modalities', 'start_date', 'end_date']].to_string(index=False))

    # Save quality report
    quality_report.to_csv(f"{timeline_output}/data_quality_report.csv", index=False)
    print(f"\n✓ Quality report saved to: {timeline_output}/data_quality_report.csv")

    # Filter users by coverage
    print("\n" + "="*80)
    valid_users = aligner.filter_users_by_coverage(
        timelines,
        min_days=14,
        required_modalities=['app_usage', 'call_log', 'sms', 'sensing/gps']
    )

    # Save valid user list
    valid_users_df = pd.DataFrame({'user_id': valid_users})
    valid_users_df.to_csv(f"{timeline_output}/valid_users.csv", index=False)
    print(f"✓ Valid user list saved to: {timeline_output}/valid_users.csv")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"  Total users with PHQ-9: {len(user_ids)}")
    print(f"  Users with sufficient data: {len(valid_users)}")
    print(f"  Excluded users: {len(user_ids) - len(valid_users)}")
    print(f"\n  Data collection period:")
    print(f"    Mean: {quality_report['total_days'].mean():.1f} ± {quality_report['total_days'].std():.1f} days")
    print(f"    Median: {quality_report['total_days'].median():.1f} days")
    print(f"    Range: {quality_report['total_days'].min():.1f} - {quality_report['total_days'].max():.1f} days")
    print(f"\n  Average modalities per user: {quality_report['n_modalities'].mean():.1f}")

    print("\n" + "="*80)
    print("✓ PHASE 1 COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - {labels_output}/item9_labels_pre.csv")
    print(f"  - {labels_output}/phq9_labels_pre.csv")
    print(f"  - {labels_output}/item9_labels_post.csv")
    print(f"  - {labels_output}/phq9_labels_post.csv")
    print(f"  - {timeline_output}/user_timelines.json")
    print(f"  - {timeline_output}/data_quality_report.csv")
    print(f"  - {timeline_output}/valid_users.csv")
    print("\nNext step: Feature extraction (Phase 2)")
    print("  Run: jupyter notebook notebooks/02_feature_engineering.ipynb")


if __name__ == "__main__":
    main()
