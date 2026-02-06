"""
Bluetooth Feature Extractor

Extracts social proximity features from Bluetooth scan data.
Each scan detects nearby Bluetooth devices (MAC addresses).

Features:
- bt_unique_devices_mean/std: unique devices per day (social density)
- bt_scan_count_mean: scans per day
- bt_device_entropy: diversity of encountered devices (Shannon entropy)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import json


class BluetoothExtractor:

    def __init__(self, data_dir: str, timelines_file: str):
        self.data_dir = Path(data_dir)
        self.timelines = self._load_timelines(timelines_file)

    def _load_timelines(self, path):
        with open(path) as f:
            return json.load(f)

    def extract_features(self, user_id: str) -> dict | None:
        filepath = self.data_dir / f'bt_{user_id}.csv'
        if not filepath.exists():
            return None

        try:
            df = pd.read_csv(filepath)
        except Exception:
            return None

        if 'time' not in df.columns or 'MAC' not in df.columns:
            return None

        df = df.dropna(subset=['time', 'MAC'])
        if len(df) < 50:
            return None

        # Filter by timeline
        if user_id in self.timelines:
            end_ts = self.timelines[user_id].get('data_end', float('inf'))
            df = df[df['time'] <= end_ts]

        if len(df) < 50:
            return None

        df['date'] = pd.to_datetime(df['time'], unit='s').dt.date

        daily = df.groupby('date').agg(
            unique_devices=('MAC', 'nunique'),
            scan_count=('MAC', 'count'),
        )

        if len(daily) < 7:
            return None

        # Device entropy (how evenly distributed encounters are across devices)
        mac_counts = df['MAC'].value_counts().values
        probs = mac_counts / mac_counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        return {
            'bt_unique_devices_mean': daily['unique_devices'].mean(),
            'bt_unique_devices_std': daily['unique_devices'].std(),
            'bt_scan_count_mean': daily['scan_count'].mean(),
            'bt_device_entropy': entropy,
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
