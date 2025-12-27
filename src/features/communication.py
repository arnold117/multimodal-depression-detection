"""
Communication Feature Extractor

Extracts digital biomarkers from call and SMS patterns:
- Call patterns (frequency, duration, diversity)
- SMS patterns (sent/received, unique contacts)
- Social diversity (combined unique contacts)
- Communication variability (day-to-day stability)
- Temporal trend (increasing isolation?)

Reference:
Farhan, A. A., et al. (2016). Behavior vs. introspection: refining prediction
of clinical depression via smartphone sensing data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
from datetime import datetime


class CommunicationExtractor:
    """Extract communication features from call and SMS logs."""

    def __init__(self,
                 call_dir: Path,
                 sms_dir: Path,
                 timelines_file: Path):
        """
        Initialize communication extractor.

        Args:
            call_dir: Directory containing call log files (call_log_u*.csv)
            sms_dir: Directory containing SMS files (sms_u*.csv)
            timelines_file: Path to user_timelines.json
        """
        self.call_dir = Path(call_dir)
        self.sms_dir = Path(sms_dir)

        # Load user timelines
        with open(timelines_file, 'r') as f:
            self.timelines = json.load(f)

    def extract_features(self, user_id: str) -> Optional[Dict[str, float]]:
        """
        Extract all communication features for a user.

        Args:
            user_id: User identifier (e.g., 'u00')

        Returns:
            Dictionary of features or None if insufficient data
        """
        # Load call and SMS data
        call_data = self._load_call_data(user_id)
        sms_data = self._load_sms_data(user_id)

        # Filter by timeline
        timeline = self.timelines.get(user_id)
        if timeline is None:
            return None

        data_end = timeline['data_end']

        if call_data is not None:
            call_data = call_data[call_data['timestamp'] <= data_end]
        if sms_data is not None:
            sms_data = sms_data[sms_data['timestamp'] <= data_end]

        # Need at least some communication data
        has_calls = call_data is not None and len(call_data) >= 10
        has_sms = sms_data is not None and len(sms_data) >= 10

        if not (has_calls or has_sms):
            return None

        # Compute daily features
        call_daily = self._compute_call_daily(call_data) if has_calls else None
        sms_daily = self._compute_sms_daily(sms_data) if has_sms else None

        # Merge daily features
        if call_daily is not None and sms_daily is not None:
            daily_features = call_daily.merge(sms_daily, on='date', how='outer').fillna(0)
        elif call_daily is not None:
            daily_features = call_daily
        else:
            daily_features = sms_daily

        if daily_features is None or len(daily_features) < 7:
            return None

        features = {}

        # 1. Call features
        if has_calls:
            features['call_count_mean'] = daily_features['call_count'].mean()
            features['call_count_std'] = daily_features['call_count'].std()
            features['call_duration_mean'] = daily_features['call_duration'].mean()
            features['call_duration_std'] = daily_features['call_duration'].std()
            features['call_incoming_ratio'] = daily_features['call_incoming'].sum() / (daily_features['call_count'].sum() + 1e-10)
            features['call_unique_contacts_mean'] = daily_features['call_unique'].mean()
        else:
            features['call_count_mean'] = 0.0
            features['call_count_std'] = 0.0
            features['call_duration_mean'] = 0.0
            features['call_duration_std'] = 0.0
            features['call_incoming_ratio'] = 0.0
            features['call_unique_contacts_mean'] = 0.0

        # 2. SMS features
        if has_sms:
            features['sms_count_mean'] = daily_features['sms_count'].mean()
            features['sms_count_std'] = daily_features['sms_count'].std()
            features['sms_received_ratio'] = daily_features['sms_received'].sum() / (daily_features['sms_count'].sum() + 1e-10)
            features['sms_unique_contacts_mean'] = daily_features['sms_unique'].mean()
        else:
            features['sms_count_mean'] = 0.0
            features['sms_count_std'] = 0.0
            features['sms_received_ratio'] = 0.0
            features['sms_unique_contacts_mean'] = 0.0

        # 3. Combined social diversity
        if has_calls and has_sms:
            total_contacts_call = call_data['contact'].nunique() if call_data is not None else 0
            total_contacts_sms = sms_data['contact'].nunique() if sms_data is not None else 0
            features['total_unique_contacts'] = total_contacts_call + total_contacts_sms
        else:
            features['total_unique_contacts'] = (call_data['contact'].nunique() if has_calls else 0) + \
                                                (sms_data['contact'].nunique() if has_sms else 0)

        # 4. Communication variability (CV)
        total_comm_mean = daily_features['call_count'].mean() + daily_features.get('sms_count', pd.Series([0])).mean()
        total_comm_std = (daily_features['call_count'] + daily_features.get('sms_count', 0)).std()
        features['total_communication_cv'] = total_comm_std / (total_comm_mean + 1e-10)

        # 5. Temporal trend (linear regression slope)
        features['communication_trend_slope'] = self._compute_temporal_trend(daily_features)

        # 6. Data coverage
        features['comm_valid_days'] = len(daily_features)

        return features

    def _load_call_data(self, user_id: str) -> Optional[pd.DataFrame]:
        """Load and clean call log data for a user."""
        call_file = self.call_dir / f'call_log_{user_id}.csv'

        if not call_file.exists():
            return None

        try:
            df = pd.read_csv(call_file, usecols=['timestamp', 'CALLS_date', 'CALLS_duration', 'CALLS_number', 'CALLS_type'],
                           encoding='utf-8-sig')

            # Remove rows with missing data
            df = df.dropna(subset=['CALLS_date', 'CALLS_number'])

            # Rename columns
            df.rename(columns={
                'CALLS_date': 'call_time',
                'CALLS_duration': 'duration',
                'CALLS_number': 'contact',
                'CALLS_type': 'call_type'
            }, inplace=True)

            # Convert timestamp
            df['call_time'] = pd.to_numeric(df['call_time'], errors='coerce') / 1000  # Convert ms to seconds
            df = df.dropna(subset=['call_time'])

            df['datetime'] = pd.to_datetime(df['call_time'], unit='s')
            df['date'] = df['datetime'].dt.date

            # Call type: 1=incoming, 2=outgoing, 3=missed
            df['is_incoming'] = (df['call_type'] == 1).astype(int)

            # Duration in seconds
            df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0)

            return df

        except Exception as e:
            print(f"  Warning: Could not process calls for {user_id}: {e}")
            return None

    def _load_sms_data(self, user_id: str) -> Optional[pd.DataFrame]:
        """Load and clean SMS data for a user."""
        sms_file = self.sms_dir / f'sms_{user_id}.csv'

        if not sms_file.exists():
            return None

        try:
            df = pd.read_csv(sms_file, usecols=['timestamp', 'MESSAGES_date', 'MESSAGES_address', 'MESSAGES_type'],
                           encoding='utf-8-sig')

            # Remove rows with missing data
            df = df.dropna(subset=['MESSAGES_date', 'MESSAGES_address'])

            # Rename columns
            df.rename(columns={
                'MESSAGES_date': 'sms_time',
                'MESSAGES_address': 'contact',
                'MESSAGES_type': 'sms_type'
            }, inplace=True)

            # Convert timestamp
            df['sms_time'] = pd.to_numeric(df['sms_time'], errors='coerce') / 1000  # Convert ms to seconds
            df = df.dropna(subset=['sms_time'])

            df['datetime'] = pd.to_datetime(df['sms_time'], unit='s')
            df['date'] = df['datetime'].dt.date

            # SMS type: 1=received, 2=sent
            df['is_received'] = (df['sms_type'] == 1).astype(int)

            return df

        except Exception as e:
            print(f"  Warning: Could not process SMS for {user_id}: {e}")
            return None

    def _compute_call_daily(self, call_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute daily call features."""
        if call_data is None or len(call_data) == 0:
            return None

        daily_list = []

        for date, day_data in call_data.groupby('date'):
            daily_list.append({
                'date': date,
                'call_count': len(day_data),
                'call_duration': day_data['duration'].sum(),
                'call_incoming': day_data['is_incoming'].sum(),
                'call_unique': day_data['contact'].nunique()
            })

        return pd.DataFrame(daily_list)

    def _compute_sms_daily(self, sms_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute daily SMS features."""
        if sms_data is None or len(sms_data) == 0:
            return None

        daily_list = []

        for date, day_data in sms_data.groupby('date'):
            daily_list.append({
                'date': date,
                'sms_count': len(day_data),
                'sms_received': day_data['is_received'].sum(),
                'sms_unique': day_data['contact'].nunique()
            })

        return pd.DataFrame(daily_list)

    def _compute_temporal_trend(self, daily_features: pd.DataFrame) -> float:
        """Compute linear trend in total communication over time."""
        # Total communication = calls + SMS
        daily_features = daily_features.copy()
        daily_features['total_comm'] = daily_features['call_count'] + daily_features.get('sms_count', 0)

        # Linear regression: y = slope * x + intercept
        x = np.arange(len(daily_features))
        y = daily_features['total_comm'].values

        if len(x) < 2:
            return 0.0

        # Compute slope
        slope = np.polyfit(x, y, 1)[0]

        return slope

    def extract_all_users(self, user_ids: list) -> pd.DataFrame:
        """
        Extract communication features for all users.

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
                print(f"✓ ({features['comm_valid_days']:.0f} days)")
            else:
                print("✗ (insufficient data)")

        if len(results) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Reorder columns (uid first)
        cols = ['uid'] + [c for c in df.columns if c != 'uid']
        df = df[cols]

        return df
