"""
Audio Feature Extractor

Extracts social environment features from microphone audio inference.
Audio inference codes: 0=silence, 1=voice, 2=noise.

Features:
- audio_silence_ratio: proportion of silent time
- audio_voice_ratio: proportion of voice detected (social proxy)
- audio_noise_ratio: proportion of environmental noise
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


class AudioExtractor:

    def __init__(self, data_dir: str, timelines_file: str, chunksize: int = 500000):
        self.data_dir = Path(data_dir)
        self.timelines = self._load_timelines(timelines_file)
        self.chunksize = chunksize

    def _load_timelines(self, path):
        with open(path) as f:
            return json.load(f)

    def extract_features(self, user_id: str) -> dict | None:
        filepath = self.data_dir / f'audio_{user_id}.csv'
        if not filepath.exists():
            return None

        end_ts = float('inf')
        if user_id in self.timelines:
            end_ts = self.timelines[user_id].get('data_end', float('inf'))

        # Chunk-based reading (files can be 5M+ lines)
        counts = {0: 0, 1: 0, 2: 0}
        total = 0

        try:
            for chunk in pd.read_csv(filepath, chunksize=self.chunksize):
                chunk.columns = [c.strip() for c in chunk.columns]
                if 'audio inference' not in chunk.columns:
                    return None

                chunk['timestamp'] = pd.to_numeric(chunk['timestamp'], errors='coerce')
                chunk = chunk.dropna(subset=['timestamp'])
                chunk = chunk[chunk['timestamp'] <= end_ts]

                vc = chunk['audio inference'].value_counts()
                for code in [0, 1, 2]:
                    counts[code] += vc.get(code, 0)
                total += len(chunk)
        except Exception:
            return None

        if total < 100:
            return None

        return {
            'audio_silence_ratio': counts[0] / total,
            'audio_voice_ratio': counts[1] / total,
            'audio_noise_ratio': counts[2] / total,
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
