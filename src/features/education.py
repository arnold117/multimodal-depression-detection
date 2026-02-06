"""
Education Feature Extractors

Extracts academic engagement features from Piazza and deadline data.

Piazza features:
- piazza_days_online, piazza_views, piazza_contributions
- piazza_questions, piazza_answers
- piazza_engagement_ratio (contributions / views)

Deadline features:
- deadline_total: total number of deadlines
- deadline_mean_per_week: average deadlines per week
- deadline_burstiness: std of weekly deadline counts (clustering)
"""

import pandas as pd
import numpy as np
from pathlib import Path


class PiazzaExtractor:

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def extract_all_users(self, user_ids: list = None) -> pd.DataFrame:
        df = pd.read_csv(self.filepath)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={
            'days online': 'piazza_days_online',
            'views': 'piazza_views',
            'contributions': 'piazza_contributions',
            'questions': 'piazza_questions',
            'notes': 'piazza_notes',
            'answers': 'piazza_answers',
        })

        # Engagement ratio (avoid div by zero)
        df['piazza_engagement_ratio'] = np.where(
            df['piazza_views'] > 0,
            df['piazza_contributions'] / df['piazza_views'],
            0.0,
        )

        cols = ['uid', 'piazza_days_online', 'piazza_views',
                'piazza_contributions', 'piazza_questions',
                'piazza_notes', 'piazza_answers', 'piazza_engagement_ratio']

        if user_ids is not None:
            df = df[df['uid'].isin(user_ids)]

        return df[cols]


class DeadlineExtractor:

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def extract_all_users(self, user_ids: list = None) -> pd.DataFrame:
        df = pd.read_csv(self.filepath)
        date_cols = [c for c in df.columns if c != 'uid']

        rows = []
        for _, row in df.iterrows():
            uid = row['uid']
            if user_ids is not None and uid not in user_ids:
                continue

            daily_counts = row[date_cols].values.astype(float)
            total = np.nansum(daily_counts)

            # Weekly aggregation
            n_weeks = max(1, len(daily_counts) // 7)
            weekly = [np.nansum(daily_counts[i*7:(i+1)*7])
                      for i in range(n_weeks)]

            rows.append({
                'uid': uid,
                'deadline_total': total,
                'deadline_mean_per_week': np.mean(weekly),
                'deadline_burstiness': np.std(weekly),
            })

        return pd.DataFrame(rows)
