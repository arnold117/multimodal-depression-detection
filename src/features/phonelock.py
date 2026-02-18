"""
Phone Lock Feature Extractor

Extracts screen usage patterns from phone lock/unlock events.
Each row = one locked period (start=lock time, end=unlock time).
An "unlock" is the gap BETWEEN locked periods.

Features:
- unlock_count_mean/std: daily unlock frequency
- session_duration_mean/std: unlocked session duration (seconds)
- night_unlock_ratio: proportion of unlocks between 0-5am
- screen_time_hours_mean: daily total unlocked time (hours)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json


class PhoneLockExtractor:

    def __init__(self, data_dir: str, timelines_file: str):
        self.data_dir = Path(data_dir)
        self.timelines = self._load_timelines(timelines_file)

    def _load_timelines(self, path):
        with open(path) as f:
            return json.load(f)

    def extract_features(self, user_id: str) -> dict | None:
        filepath = self.data_dir / f'phonelock_{user_id}.csv'
        if not filepath.exists():
            return None

        try:
            df = pd.read_csv(filepath)
        except Exception:
            return None

        if len(df) < 10:
            return None

        # Each row is a LOCKED period. Unlocked sessions are gaps between rows.
        df = df.sort_values('start').reset_index(drop=True)

        # Filter by timeline
        if user_id in self.timelines:
            end_ts = self.timelines[user_id].get('data_end', float('inf'))
            df = df[df['start'] <= end_ts]

        if len(df) < 10:
            return None

        # Compute unlocked sessions (gap between end of lock_i and start of lock_{i+1})
        unlock_starts = df['end'].values[:-1]
        unlock_ends = df['start'].values[1:]
        session_durations = unlock_ends - unlock_starts

        # Filter out negative or extremely long sessions (>12h = likely phone off)
        valid = (session_durations > 0) & (session_durations < 43200)
        session_durations = session_durations[valid]
        unlock_starts_valid = unlock_starts[valid]

        if len(session_durations) < 5:
            return None

        # Daily aggregation
        dates = pd.to_datetime(unlock_starts_valid, unit='s').date
        daily = pd.DataFrame({'date': dates, 'duration': session_durations})
        daily_agg = daily.groupby('date').agg(
            unlock_count=('duration', 'count'),
            screen_time=('duration', 'sum'),
        )

        if len(daily_agg) < 7:
            return None

        # Night unlocks (0-5am)
        hours = pd.to_datetime(unlock_starts_valid, unit='s').hour
        night_mask = hours < 5
        night_ratio = night_mask.mean() if len(hours) > 0 else 0

        return {
            'unlock_count_mean': daily_agg['unlock_count'].mean(),
            'unlock_count_std': daily_agg['unlock_count'].std(),
            'session_duration_mean': session_durations.mean(),
            'session_duration_std': session_durations.std(),
            'night_unlock_ratio': float(night_ratio),
            'screen_time_hours_mean': (daily_agg['screen_time'] / 3600).mean(),
        }

    def extract_all_users(self, user_ids: list) -> pd.DataFrame:
        rows = []
        for uid in user_ids:
            print(f'Processing {uid}...', end=' ')
            features = self.extract_features(uid)
            if features:
                features['uid'] = uid
                rows.append(features)
                print(f'✓')
            else:
                print(f'✗')
        return pd.DataFrame(rows)
