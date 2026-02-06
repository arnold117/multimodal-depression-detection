"""
EMA (Ecological Momentary Assessment) Feature Extractors

Extracts features from real-time mood and stress self-reports.

Mood features:
- ema_happy_mean/std: average and variability of happiness (1-4)
- ema_sad_mean/std: average and variability of sadness (1-4)

Stress features:
- ema_stress_mean/std: average and variability of stress level (1-5)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


class EMAMoodExtractor:

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def extract_features(self, user_id: str) -> dict | None:
        filepath = self.data_dir / f'Mood_{user_id}.json'
        if not filepath.exists():
            return None

        try:
            records = json.load(open(filepath))
        except Exception:
            return None

        if len(records) < 3:
            return None

        happy_vals = []
        sad_vals = []
        for r in records:
            h = r.get('happy')
            s = r.get('sad')
            if h is not None:
                try:
                    happy_vals.append(float(h))
                except (ValueError, TypeError):
                    pass
            if s is not None:
                try:
                    sad_vals.append(float(s))
                except (ValueError, TypeError):
                    pass

        if len(happy_vals) < 3 and len(sad_vals) < 3:
            return None

        result = {}
        if len(happy_vals) >= 3:
            result['ema_happy_mean'] = np.mean(happy_vals)
            result['ema_happy_std'] = np.std(happy_vals)
        if len(sad_vals) >= 3:
            result['ema_sad_mean'] = np.mean(sad_vals)
            result['ema_sad_std'] = np.std(sad_vals)

        return result if result else None

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


class EMAStressExtractor:

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def extract_features(self, user_id: str) -> dict | None:
        filepath = self.data_dir / f'Stress_{user_id}.json'
        if not filepath.exists():
            return None

        try:
            records = json.load(open(filepath))
        except Exception:
            return None

        # Stress records have varying formats:
        # Some have 'level' key, others use 'null' key with numeric value
        levels = []
        for r in records:
            val = r.get('level') or r.get('null')
            if val is not None:
                try:
                    v = float(val)
                    if 1 <= v <= 5:  # valid stress level
                        levels.append(v)
                except (ValueError, TypeError):
                    pass

        if len(levels) < 3:
            return None

        levels = np.array(levels)

        return {
            'ema_stress_mean': levels.mean(),
            'ema_stress_std': levels.std(),
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
