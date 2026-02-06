"""
WiFi Feature Extractor

Extracts location-proxy features from WiFi scan data.
Unique APs (access points) visited indicates location diversity.

Features:
- wifi_unique_aps_mean/std: unique access points per day
- wifi_location_entropy: Shannon entropy of AP frequency distribution
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


class WiFiExtractor:

    def __init__(self, data_dir: str, timelines_file: str):
        self.data_dir = Path(data_dir)
        self.timelines = self._load_timelines(timelines_file)

    def _load_timelines(self, path):
        with open(path) as f:
            return json.load(f)

    def extract_features(self, user_id: str) -> dict | None:
        filepath = self.data_dir / f'wifi_{user_id}.csv'
        if not filepath.exists():
            return None

        try:
            df = pd.read_csv(filepath)
        except Exception:
            return None

        if 'time' not in df.columns or 'BSSID' not in df.columns:
            return None

        df = df.dropna(subset=['time', 'BSSID'])
        if len(df) < 50:
            return None

        # Filter by timeline
        if user_id in self.timelines:
            end_ts = self.timelines[user_id].get('data_end', float('inf'))
            df = df[df['time'] <= end_ts]

        if len(df) < 50:
            return None

        df['date'] = pd.to_datetime(df['time'], unit='s').dt.date

        daily = df.groupby('date')['BSSID'].nunique()
        if len(daily) < 7:
            return None

        # AP entropy
        ap_counts = df['BSSID'].value_counts().values
        probs = ap_counts / ap_counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        return {
            'wifi_unique_aps_mean': daily.mean(),
            'wifi_unique_aps_std': daily.std(),
            'wifi_location_entropy': entropy,
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
