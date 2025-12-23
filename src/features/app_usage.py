"""
App Usage Feature Extractor

Extracts digital biomarkers from app usage patterns:
- Usage volume (app switches, unique apps)
- Circadian disruption (night usage, regularity)
- Variability (day-to-day stability)
- Weekend patterns

Reference:
Wang, R., et al. (2014). StudentLife: assessing mental health, academic
performance and behavioral trends of college students using smartphones.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime


class AppUsageExtractor:
    """Extract app usage features from running app logs."""

    def __init__(self,
                 data_dir: Path,
                 timelines_file: Path):
        """
        Initialize app usage extractor.

        Args:
            data_dir: Directory containing app usage files (running_app_u*.csv)
            timelines_file: Path to user_timelines.json
        """
        self.data_dir = Path(data_dir)

        # Load user timelines
        with open(timelines_file, 'r') as f:
            self.timelines = json.load(f)

    def extract_features(self, user_id: str) -> Optional[Dict[str, float]]:
        """
        Extract all app usage features for a user.

        Args:
            user_id: User identifier (e.g., 'u00')

        Returns:
            Dictionary of features or None if insufficient data
        """
        # Load app usage data
        app_data = self._load_app_data(user_id)

        if app_data is None or len(app_data) == 0:
            return None

        # Filter by timeline (use data before pre-assessment)
        timeline = self.timelines.get(user_id)
        if timeline is None:
            return None

        data_end = timeline['data_end']
        app_data = app_data[app_data['timestamp'] <= data_end]

        if len(app_data) < 50:  # Minimum app switches
            return None

        # Aggregate to daily level
        daily_features = self._compute_daily_features(app_data)

        if daily_features is None or len(daily_features) < 7:
            return None

        # Compute user-level statistics
        features = {}

        # 1. Usage volume
        features['app_switches_mean'] = daily_features['app_switches'].mean()
        features['app_switches_std'] = daily_features['app_switches'].std()
        features['unique_apps_mean'] = daily_features['unique_apps'].mean()
        features['unique_apps_std'] = daily_features['unique_apps'].std()

        # 2. Circadian disruption (night usage: 0-5am)
        features['night_usage_ratio'] = daily_features['night_ratio'].mean()
        features['night_usage_std'] = daily_features['night_ratio'].std()

        # 3. Variability (day-to-day instability marker)
        features['app_switches_cv'] = features['app_switches_std'] / (features['app_switches_mean'] + 1e-10)
        features['unique_apps_cv'] = features['unique_apps_std'] / (features['unique_apps_mean'] + 1e-10)

        # 4. Weekend vs weekday patterns
        weekday_switches = daily_features[daily_features['is_weekend'] == 0]['app_switches'].mean()
        weekend_switches = daily_features[daily_features['is_weekend'] == 1]['app_switches'].mean()
        features['weekend_weekday_ratio'] = weekend_switches / (weekday_switches + 1e-10)

        # 5. Data coverage metrics
        features['app_valid_days'] = len(daily_features)
        features['app_total_switches'] = len(app_data)

        return features

    def _load_app_data(self, user_id: str) -> Optional[pd.DataFrame]:
        """Load and clean app usage data for a user."""
        app_file = self.data_dir / f'running_app_{user_id}.csv'

        if not app_file.exists():
            return None

        try:
            # Read app usage file
            df = pd.read_csv(app_file, usecols=['timestamp', 'RUNNING_TASKS_topActivity_mPackage'],
                           encoding='utf-8-sig')  # Handle BOM

            # Rename for convenience
            df.rename(columns={'RUNNING_TASKS_topActivity_mPackage': 'package'}, inplace=True)

            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df['date'] = df['datetime'].dt.date
            df['hour'] = df['datetime'].dt.hour
            df['dayofweek'] = df['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday

            # Remove system apps (com.android.systemui, launcher, etc.)
            system_packages = [
                'com.android.systemui',
                'com.android.launcher',
                'com.android.launcher2',
                'com.sec.android.app.launcher',
                'com.google.android.apps.paco'  # Study app
            ]
            df = df[~df['package'].isin(system_packages)]

            # Remove rows with missing package names
            df = df.dropna(subset=['package'])

            return df

        except Exception as e:
            print(f"  Warning: Could not process {user_id}: {e}")
            return None

    def _compute_daily_features(self, app_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute daily-level app usage features."""
        daily_list = []

        for date, day_data in app_data.groupby('date'):
            if len(day_data) < 2:  # Need at least 2 switches
                continue

            # App switches (number of app change events)
            app_switches = len(day_data)

            # Unique apps per day
            unique_apps = day_data['package'].nunique()

            # Night usage ratio (0-5am)
            night_switches = len(day_data[day_data['hour'] < 5])
            night_ratio = night_switches / app_switches

            # Weekend indicator
            is_weekend = 1 if day_data['dayofweek'].iloc[0] >= 5 else 0

            daily_list.append({
                'date': date,
                'app_switches': app_switches,
                'unique_apps': unique_apps,
                'night_ratio': night_ratio,
                'is_weekend': is_weekend
            })

        if len(daily_list) == 0:
            return None

        return pd.DataFrame(daily_list)

    def extract_all_users(self, user_ids: list) -> pd.DataFrame:
        """
        Extract app usage features for all users.

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
                print(f"✓ ({features['app_valid_days']:.0f} days)")
            else:
                print("✗ (insufficient data)")

        if len(results) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Reorder columns (uid first)
        cols = ['uid'] + [c for c in df.columns if c != 'uid']
        df = df[cols]

        return df
