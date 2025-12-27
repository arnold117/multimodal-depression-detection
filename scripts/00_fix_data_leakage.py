#!/usr/bin/env python3
"""
Fix Data Leakage - Remove Label Columns from Features

This script removes item9_score and item9_binary from the feature matrix.
These are the depression labels themselves and should not be used as features.

The data leakage was discovered in Phase 7 SHAP analysis, where these
columns were the top predictive features, explaining XGBoost's perfect
100% accuracy.

Usage:
    python scripts/00_fix_data_leakage.py
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def main():
    print("=" * 80)
    print("FIX DATA LEAKAGE - REMOVE LABEL COLUMNS")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Paths
    features_path = Path('data/processed/features/combined_features.parquet')
    backup_path = Path('data/processed/features/combined_features_with_leakage.parquet')

    # Load features
    print(f"Loading features from: {features_path}")
    df = pd.read_parquet(features_path)

    print(f"Original shape: {df.shape}")
    print(f"Original columns: {len(df.columns)}")
    print(f"\nColumns with 'item9': {[c for c in df.columns if 'item9' in c]}")

    # Create backup
    print(f"\nCreating backup at: {backup_path}")
    df.to_parquet(backup_path, index=False)
    print(f"✓ Backup created")

    # Remove label columns
    label_columns = ['item9_score', 'item9_binary']

    print(f"\nRemoving label columns: {label_columns}")
    df_clean = df.drop(columns=label_columns, errors='ignore')

    print(f"\nCleaned shape: {df_clean.shape}")
    print(f"Cleaned columns: {len(df_clean.columns)}")
    print(f"Columns removed: {len(df.columns) - len(df_clean.columns)}")

    # Verify no item9 columns remain
    item9_cols = [c for c in df_clean.columns if 'item9' in c]
    if item9_cols:
        print(f"\n⚠️  Warning: Found remaining item9 columns: {item9_cols}")
    else:
        print(f"\n✓ Verified: No item9 columns in cleaned data")

    # Save cleaned version
    print(f"\nSaving cleaned features to: {features_path}")
    df_clean.to_parquet(features_path, index=False)
    print(f"✓ Saved cleaned features")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Original features: {len(df.columns)}")
    print(f"Cleaned features: {len(df_clean.columns)}")
    print(f"Removed columns: {label_columns}")
    print(f"\nBackup saved at: {backup_path}")
    print(f"Clean data saved at: {features_path}")

    print("\n" + "=" * 80)
    print("✓ DATA LEAKAGE FIXED")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print("\nNext steps:")
    print("1. Re-run baseline training: python scripts/07_train_baseline.py")
    print("2. Re-run model comparison: python scripts/12_evaluate_all_models.py")
    print("3. Re-run biomarker report: python scripts/13_generate_biomarker_report.py")
    print("\nExpected changes:")
    print("- XGBoost performance will drop from 100% (no longer has labels as features)")
    print("- SHAP analysis will show genuine behavioral biomarkers")
    print("- Top features will be app usage, GPS, communication patterns")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
