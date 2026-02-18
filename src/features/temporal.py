"""
Temporal Alignment Module

This module handles temporal alignment of behavioral data with PHQ-9 assessments.
Since PHQ-9 data lacks timestamps, we infer assessment dates from behavioral data boundaries.

Author: Digital Phenotyping Research
Date: December 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import json


class TemporalAligner:
    """
    Aligns behavioral data timestamps with PHQ-9 assessment periods.

    Strategy:
    - Pre-assessment: Assumed at data collection START (earliest timestamp)
    - Post-assessment: Assumed at data collection END (latest timestamp)
    - Extract features from ALL data before pre-assessment
    """

    def __init__(self, data_root: str):
        """
        Initialize TemporalAligner.

        Args:
            data_root: Root directory of raw behavioral data
        """
        self.data_root = Path(data_root)
        self.user_timelines = {}

    def extract_timestamps_from_file(self, file_path: Path,
                                     timestamp_col: str = 'timestamp') -> Tuple[float, float]:
        """
        Extract min and max timestamps from a single data file.

        Args:
            file_path: Path to data file
            timestamp_col: Name of timestamp column

        Returns:
            Tuple of (min_timestamp, max_timestamp) in Unix format
        """
        try:
            # First, try to identify the timestamp column
            # Handle different timestamp column names
            timestamp_columns = [timestamp_col, 'time', 'resp_time', ' time']

            # Read first line to detect columns
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                columns = first_line.split(',')

            found_col = None
            for col in timestamp_columns:
                if col in columns:
                    found_col = col
                    break

            if found_col is None:
                return None, None

            # Read only the timestamp column to avoid parsing issues with malformed CSVs
            try:
                # Try normal reading first
                df = pd.read_csv(file_path, usecols=[found_col])
                timestamps = pd.to_numeric(df[found_col], errors='coerce')

                # If all values are NaN, the CSV might be malformed - try reading first column directly
                if timestamps.isna().all():
                    col_index = columns.index(found_col)
                    df = pd.read_csv(file_path, header=None, usecols=[col_index], names=[found_col], skiprows=1)
                    timestamps = pd.to_numeric(df[found_col], errors='coerce')
            except:
                # If usecols fails, try reading first column directly (for malformed CSVs)
                try:
                    df = pd.read_csv(file_path, header=None, usecols=[0], names=[found_col], skiprows=1)
                    timestamps = pd.to_numeric(df[found_col], errors='coerce')
                except:
                    return None, None

            # Filter out invalid timestamps (NaN and non-positive values)
            timestamps = timestamps.dropna()
            timestamps = timestamps[timestamps > 0]

            if len(timestamps) == 0:
                return None, None

            return float(timestamps.min()), float(timestamps.max())

        except Exception as e:
            print(f"  Warning: Could not read {file_path.name}: {e}")
            return None, None

    def get_user_timeline(self, user_id: str, modalities: list = None) -> Dict:
        """
        Extract complete timeline for a user across all modalities.

        Args:
            user_id: User identifier (e.g., 'u00')
            modalities: List of modality directories to check
                       Default: ['app_usage', 'call_log', 'sms', 'sensing/gps',
                                'sensing/activity', 'sensing/phonelock']

        Returns:
            Dictionary with timeline information:
            {
                'user_id': str,
                'data_start': timestamp (earliest across all modalities),
                'data_end': timestamp (latest across all modalities),
                'modality_coverage': dict of modality: (start, end, days)
            }
        """
        if modalities is None:
            modalities = [
                'app_usage',
                'call_log',
                'sms',
                'sensing/gps',
                'sensing/activity',
                'sensing/phonelock',
                'sensing/bluetooth',
                'sensing/wifi'
            ]

        timeline = {
            'user_id': user_id,
            'data_start': None,
            'data_end': None,
            'modality_coverage': {}
        }

        all_starts = []
        all_ends = []

        for modality in modalities:
            modality_path = self.data_root / modality

            if not modality_path.exists():
                continue

            # Find user file (different naming conventions)
            user_files = list(modality_path.glob(f"*{user_id}*.csv"))

            if len(user_files) == 0:
                continue

            user_file = user_files[0]

            # Extract timestamps
            min_ts, max_ts = self.extract_timestamps_from_file(user_file)

            if min_ts is not None and max_ts is not None:
                all_starts.append(min_ts)
                all_ends.append(max_ts)

                # Calculate days of coverage
                days = (max_ts - min_ts) / 86400  # seconds to days

                timeline['modality_coverage'][modality] = {
                    'start': min_ts,
                    'end': max_ts,
                    'days': round(days, 2),
                    'start_date': datetime.fromtimestamp(min_ts).strftime('%Y-%m-%d'),
                    'end_date': datetime.fromtimestamp(max_ts).strftime('%Y-%m-%d')
                }

        # Aggregate across modalities
        if len(all_starts) > 0:
            timeline['data_start'] = min(all_starts)
            timeline['data_end'] = max(all_ends)
            timeline['total_days'] = round((timeline['data_end'] - timeline['data_start']) / 86400, 2)
            timeline['start_date'] = datetime.fromtimestamp(timeline['data_start']).strftime('%Y-%m-%d')
            timeline['end_date'] = datetime.fromtimestamp(timeline['data_end']).strftime('%Y-%m-%d')

        return timeline

    def get_all_user_timelines(self, user_ids: list) -> Dict[str, Dict]:
        """
        Extract timelines for all users.

        Args:
            user_ids: List of user identifiers

        Returns:
            Dictionary mapping user_id to timeline dict
        """
        print(f"Extracting timelines for {len(user_ids)} users...")

        timelines = {}
        for user_id in user_ids:
            print(f"  Processing {user_id}...", end=' ')
            timeline = self.get_user_timeline(user_id)

            if timeline['data_start'] is not None:
                timelines[user_id] = timeline
                print(f"✓ ({timeline['total_days']:.1f} days)")
            else:
                print("✗ (no data)")

        return timelines

    def infer_assessment_dates(self, timeline: Dict) -> Dict:
        """
        Infer PHQ-9 assessment dates based on data collection period.

        Strategy:
        - Pre-assessment: At data collection start
        - Post-assessment: At data collection end

        Args:
            timeline: User timeline dict from get_user_timeline()

        Returns:
            Dict with inferred assessment dates:
            {
                'pre_assessment_date': timestamp,
                'post_assessment_date': timestamp,
                'feature_window_start': timestamp (for pre-assessment features),
                'feature_window_end': timestamp (pre-assessment date)
            }
        """
        if timeline['data_start'] is None:
            return None

        assessment_dates = {
            'user_id': timeline['user_id'],
            'pre_assessment_date': timeline['data_start'],
            'post_assessment_date': timeline['data_end'],
            'feature_window_start': timeline['data_start'],
            'feature_window_end': timeline['data_start'],  # Use all data before pre-assessment
            'collection_days': timeline['total_days']
        }

        return assessment_dates

    def save_timelines(self, timelines: Dict[str, Dict], output_path: str):
        """
        Save user timelines to JSON file.

        Args:
            timelines: Dictionary of user timelines
            output_path: Path to save JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(timelines, f, indent=2)

        print(f"\n✓ Timelines saved to: {output_file}")

    def load_timelines(self, input_path: str) -> Dict[str, Dict]:
        """
        Load user timelines from JSON file.

        Args:
            input_path: Path to JSON file

        Returns:
            Dictionary of user timelines
        """
        with open(input_path, 'r') as f:
            timelines = json.load(f)

        print(f"✓ Loaded timelines for {len(timelines)} users")
        return timelines

    def get_data_quality_report(self, timelines: Dict[str, Dict]) -> pd.DataFrame:
        """
        Generate data quality report across users.

        Args:
            timelines: Dictionary of user timelines

        Returns:
            DataFrame with quality metrics per user
        """
        report_data = []

        for user_id, timeline in timelines.items():
            row = {
                'user_id': user_id,
                'total_days': timeline['total_days'],
                'start_date': timeline['start_date'],
                'end_date': timeline['end_date'],
                'n_modalities': len(timeline['modality_coverage'])
            }

            # Add per-modality coverage
            for modality, info in timeline['modality_coverage'].items():
                modality_short = modality.split('/')[-1]  # Remove 'sensing/' prefix
                row[f'{modality_short}_days'] = info['days']

            report_data.append(row)

        report_df = pd.DataFrame(report_data)
        return report_df

    def filter_users_by_coverage(self, timelines: Dict[str, Dict],
                                 min_days: int = 14,
                                 required_modalities: list = None) -> list:
        """
        Filter users based on data coverage requirements.

        Args:
            timelines: Dictionary of user timelines
            min_days: Minimum days of data required
            required_modalities: List of modalities that must be present

        Returns:
            List of user IDs meeting criteria
        """
        if required_modalities is None:
            required_modalities = ['app_usage', 'call_log', 'sms', 'sensing/gps']

        valid_users = []

        for user_id, timeline in timelines.items():
            # Check total days
            if timeline['total_days'] < min_days:
                continue

            # Check required modalities
            has_all_modalities = True
            for modality in required_modalities:
                if modality not in timeline['modality_coverage']:
                    has_all_modalities = False
                    break

                # Check modality has sufficient data
                if timeline['modality_coverage'][modality]['days'] < min_days:
                    has_all_modalities = False
                    break

            if has_all_modalities:
                valid_users.append(user_id)

        print(f"\nData Quality Filter:")
        print(f"  Total users: {len(timelines)}")
        print(f"  Valid users (≥{min_days} days + required modalities): {len(valid_users)}")
        print(f"  Excluded users: {len(timelines) - len(valid_users)}")

        return valid_users


def main():
    """
    Main function for standalone execution.
    """
    # Configuration
    data_root = "data/raw/dataset"
    output_dir = "data/interim/user_timelines"

    # Load user IDs from PHQ-9
    phq9_df = pd.read_csv("data/raw/dataset/survey/PHQ-9.csv")
    user_ids = phq9_df[phq9_df['type'] == 'pre']['uid'].unique().tolist()

    print(f"Found {len(user_ids)} users with PHQ-9 pre-assessment")

    # Extract timelines
    aligner = TemporalAligner(data_root)
    timelines = aligner.get_all_user_timelines(user_ids)

    # Save timelines
    aligner.save_timelines(timelines, f"{output_dir}/user_timelines.json")

    # Generate quality report
    print("\n" + "="*80)
    print("DATA QUALITY REPORT")
    print("="*80)
    quality_report = aligner.get_data_quality_report(timelines)
    print(quality_report.to_string(index=False))

    # Save quality report
    quality_report.to_csv(f"{output_dir}/data_quality_report.csv", index=False)
    print(f"\n✓ Quality report saved to: {output_dir}/data_quality_report.csv")

    # Filter users by coverage
    valid_users = aligner.filter_users_by_coverage(
        timelines,
        min_days=14,
        required_modalities=['app_usage', 'call_log', 'sms', 'sensing/gps']
    )

    print(f"\nValid users for modeling: {valid_users}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"  Mean data collection days: {quality_report['total_days'].mean():.1f} ± {quality_report['total_days'].std():.1f}")
    print(f"  Median: {quality_report['total_days'].median():.1f}")
    print(f"  Range: {quality_report['total_days'].min():.1f} - {quality_report['total_days'].max():.1f}")
    print(f"  Mean modalities per user: {quality_report['n_modalities'].mean():.1f}")


if __name__ == "__main__":
    main()
