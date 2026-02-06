"""
Conversation Feature Extractor

Extracts face-to-face interaction features from microphone-detected conversations.
Each row = one conversation with start/end timestamps.

Features:
- convo_count_mean/std: conversations per day
- convo_duration_mean/std: conversation duration (seconds)
- convo_night_ratio: proportion of conversations between 0-5am
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


class ConversationExtractor:

    def __init__(self, data_dir: str, timelines_file: str):
        self.data_dir = Path(data_dir)
        self.timelines = self._load_timelines(timelines_file)

    def _load_timelines(self, path):
        with open(path) as f:
            return json.load(f)

    def extract_features(self, user_id: str) -> dict | None:
        filepath = self.data_dir / f'conversation_{user_id}.csv'
        if not filepath.exists():
            return None

        try:
            df = pd.read_csv(filepath, skipinitialspace=True)
        except Exception:
            return None

        # Normalize column names
        df.columns = [c.strip() for c in df.columns]
        if 'start_timestamp' not in df.columns or 'end_timestamp' not in df.columns:
            return None

        df = df.dropna(subset=['start_timestamp', 'end_timestamp'])
        df['start_timestamp'] = pd.to_numeric(df['start_timestamp'], errors='coerce')
        df['end_timestamp'] = pd.to_numeric(df['end_timestamp'], errors='coerce')
        df = df.dropna()

        if len(df) < 10:
            return None

        # Filter by timeline
        if user_id in self.timelines:
            end_ts = self.timelines[user_id].get('data_end', float('inf'))
            df = df[df['start_timestamp'] <= end_ts]

        df['duration'] = df['end_timestamp'] - df['start_timestamp']
        df = df[(df['duration'] > 0) & (df['duration'] < 7200)]  # max 2h

        if len(df) < 10:
            return None

        df['date'] = pd.to_datetime(df['start_timestamp'], unit='s').dt.date
        daily = df.groupby('date').agg(
            count=('duration', 'count'),
            total_duration=('duration', 'sum'),
        )

        if len(daily) < 7:
            return None

        hours = pd.to_datetime(df['start_timestamp'], unit='s').dt.hour
        night_ratio = (hours < 5).mean()

        return {
            'convo_count_mean': daily['count'].mean(),
            'convo_count_std': daily['count'].std(),
            'convo_duration_mean': df['duration'].mean(),
            'convo_duration_std': df['duration'].std(),
            'convo_night_ratio': float(night_ratio),
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
