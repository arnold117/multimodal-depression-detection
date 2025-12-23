"""
Activity Feature Extractor

Extracts digital biomarkers from physical activity sensor data:
- Activity levels (still/moving ratios)
- Sedentary behavior patterns
- Activity variability
- Activity fragmentation (state transitions)
- Temporal trends

Activity codes:
- 0: vehicle
- 1: bicycle
- 2: foot/walking
- 3: still/stationary

Reference:
Canzian, L., & Musolesi, M. (2015). Trajectories of depression: unobtrusive
monitoring of depressive states by means of smartphone mobility traces analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime


class ActivityExtractor:
    """Extract physical activity features from activity sensor data."""

    def __init__(self,
                 data_dir: Path,
                 timelines_file: Path,
                 chunksize: int = 100000):
        """
        Initialize activity extractor.

        Args:
            data_dir: Directory containing activity files (activity_u*.csv)
            timelines_file: Path to user_timelines.json
            chunksize: Number of rows to read per chunk (for memory efficiency)
        """
        self.data_dir = Path(data_dir)
        self.chunksize = chunksize

        # Load user timelines
        with open(timelines_file, 'r') as f:
            self.timelines = json.load(f)

    def extract_features(self, user_id: str) -> Optional[Dict[str, float]]:
        """
        Extract all activity features for a user.

        Args:
            user_id: User identifier (e.g., 'u00')

        Returns:
            Dictionary of features or None if insufficient data
        """
        # Load activity data (chunk-based for large files)
        activity_data = self._load_activity_data(user_id)

        if activity_data is None or len(activity_data) == 0:
            return None

        # Filter by timeline
        timeline = self.timelines.get(user_id)
        if timeline is None:
            return None

        data_end = timeline['data_end']
        activity_data = activity_data[activity_data['timestamp'] <= data_end]

        if len(activity_data) < 100:  # Minimum activity samples
            return None

        # Compute daily features
        daily_features = self._compute_daily_features(activity_data)

        if daily_features is None or len(daily_features) < 7:
            return None

        # Compute user-level statistics
        features = {}

        # 1. Activity levels
        features['still_ratio_mean'] = daily_features['still_ratio'].mean()
        features['still_ratio_std'] = daily_features['still_ratio'].std()
        features['moving_ratio_mean'] = daily_features['moving_ratio'].mean()
        features['moving_ratio_std'] = daily_features['moving_ratio'].std()

        # 2. Sedentary behavior (high still ratio indicates sedentary)
        sedentary_threshold = 0.7  # >70% still time
        sedentary_days = (daily_features['still_ratio'] > sedentary_threshold).sum()
        features['sedentary_days_ratio'] = sedentary_days / len(daily_features)

        # 3. Activity variability (day-to-day consistency)
        features['moving_ratio_cv'] = features['moving_ratio_std'] / (features['moving_ratio_mean'] + 1e-10)

        # 4. Activity fragmentation (state transitions)
        features['activity_transitions_mean'] = daily_features['transitions'].mean()
        features['activity_transitions_std'] = daily_features['transitions'].std()

        # 5. Temporal trend (activity change over time)
        features['activity_trend_slope'] = self._compute_temporal_trend(daily_features)

        # 6. Data coverage
        features['activity_valid_days'] = len(daily_features)
        features['activity_total_samples'] = len(activity_data)

        return features

    def _load_activity_data(self, user_id: str) -> Optional[pd.DataFrame]:
        """Load and clean activity data for a user using chunk processing."""
        activity_file = self.data_dir / f'activity_{user_id}.csv'

        if not activity_file.exists():
            return None

        try:
            # Read in chunks for memory efficiency
            chunks = []
            for chunk in pd.read_csv(activity_file, chunksize=self.chunksize):
                # Clean column names (remove spaces)
                chunk.columns = chunk.columns.str.strip()

                # Handle different column name variations
                if 'activity inference' in chunk.columns:
                    chunk.rename(columns={'activity inference': 'activity'}, inplace=True)
                elif ' activity inference' in chunk.columns:
                    chunk.rename(columns={' activity inference': 'activity'}, inplace=True)

                # Keep only necessary columns
                chunk = chunk[['timestamp', 'activity']].copy()

                # Remove empty rows and convert activity to numeric
                chunk = chunk.dropna(subset=['timestamp', 'activity'])
                chunk['timestamp'] = pd.to_numeric(chunk['timestamp'], errors='coerce')
                chunk['activity'] = pd.to_numeric(chunk['activity'], errors='coerce')
                chunk = chunk.dropna()

                chunks.append(chunk)

            if len(chunks) == 0:
                return None

            # Concatenate all chunks
            df = pd.concat(chunks, ignore_index=True)

            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df['date'] = df['datetime'].dt.date

            # Activity codes: 0=vehicle, 1=bicycle, 2=foot, 3=still
            df['is_still'] = (df['activity'] == 3).astype(int)
            df['is_moving'] = (df['activity'].isin([1, 2])).astype(int)  # bicycle or foot

            return df

        except Exception as e:
            print(f"  Warning: Could not process {user_id}: {e}")
            return None

    def _compute_daily_features(self, activity_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute daily-level activity features."""
        daily_list = []

        for date, day_data in activity_data.groupby('date'):
            if len(day_data) < 10:  # Minimum samples per day
                continue

            total_samples = len(day_data)

            # Still ratio (proportion of time stationary)
            still_ratio = day_data['is_still'].sum() / total_samples

            # Moving ratio (proportion of time moving)
            moving_ratio = day_data['is_moving'].sum() / total_samples

            # Activity transitions (changes in activity state)
            transitions = (day_data['activity'].diff() != 0).sum()

            daily_list.append({
                'date': date,
                'still_ratio': still_ratio,
                'moving_ratio': moving_ratio,
                'transitions': transitions
            })

        if len(daily_list) == 0:
            return None

        return pd.DataFrame(daily_list)

    def _compute_temporal_trend(self, daily_features: pd.DataFrame) -> float:
        """Compute linear trend in moving ratio over time (decreasing = more sedentary)."""
        x = np.arange(len(daily_features))
        y = daily_features['moving_ratio'].values

        if len(x) < 2:
            return 0.0

        # Linear regression slope
        slope = np.polyfit(x, y, 1)[0]

        return slope

    def extract_all_users(self, user_ids: list) -> pd.DataFrame:
        """
        Extract activity features for all users.

        Args:
            user_ids: List of user IDs to process

        Returns:
            DataFrame with features (rows=users, columns=features)
        """
        results = []

        for user_id in user_ids:
            print(f"Processing {user_id}...", end=' ')
            features = self.extract_features(user_id)

            if features is not None:
                features['uid'] = user_id
                results.append(features)
                print(f"✓ ({features['activity_valid_days']:.0f} days)")
            else:
                print("✗ (insufficient data)")

        if len(results) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Reorder columns (uid first)
        cols = ['uid'] + [c for c in df.columns if c != 'uid']
        df = df[cols]

        return df
