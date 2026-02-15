#!/usr/bin/env python3
"""
Phase 8: Weekly Temporal Features

Extract behavioral trend features from raw sensing data.
Current pipeline uses full-semester means/SDs, losing trajectory information.
Behavioral deterioration trends may be more predictive than static levels.

Method:
  - Aggregate raw sensor data to daily → weekly level
  - For each behavioral metric's weekly time series, compute:
    (a) Linear regression slope (direction & speed of change)
    (b) Coefficient of variation (day-to-day stability)
    (c) First-half vs second-half difference (delta)

Input:  data/raw/dataset/sensing/  (raw sensor data)
        data/raw/dataset/app_usage/
        data/raw/dataset/call_log/
        data/raw/dataset/sms/
Output: data/processed/features/temporal_features.parquet
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent

SENSING_DIR = project_root / 'data' / 'raw' / 'dataset' / 'sensing'
APP_DIR = project_root / 'data' / 'raw' / 'dataset' / 'app_usage'
CALL_DIR = project_root / 'data' / 'raw' / 'dataset' / 'call_log'
SMS_DIR = project_root / 'data' / 'raw' / 'dataset' / 'sms'
OUTPUT_DIR = project_root / 'data' / 'processed' / 'features'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target users (those in our analysis dataset with Big Five + GPA)
TARGET_USERS = [
    'u01', 'u02', 'u04', 'u05', 'u07', 'u08', 'u09', 'u10',
    'u12', 'u14', 'u15', 'u16', 'u17', 'u18', 'u19', 'u22',
    'u24', 'u27', 'u30', 'u32', 'u33', 'u43', 'u46', 'u49',
    'u52', 'u54', 'u57', 'u59',
]

MIN_WEEKS = 4  # Minimum weeks needed for trend estimation


# ── helpers ──────────────────────────────────────────────────────

def ts_to_date(ts):
    """Unix timestamp → date."""
    return datetime.utcfromtimestamp(ts).date()


def ts_to_hour(ts):
    """Unix timestamp → hour of day."""
    return datetime.utcfromtimestamp(ts).hour


def compute_trend_features(weekly_series, prefix):
    """
    From a weekly time series, compute trend features.

    Args:
        weekly_series: pd.Series indexed by week number, values = metric
        prefix: feature name prefix

    Returns:
        dict of {feature_name: value}
    """
    vals = weekly_series.dropna()
    if len(vals) < MIN_WEEKS:
        return {}

    x = np.arange(len(vals), dtype=float)
    y = vals.values.astype(float)

    features = {}

    # (a) Linear regression slope (standardized by SD)
    if np.std(y) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        # Standardize: slope per week / SD of y
        features[f'{prefix}_slope'] = slope / np.std(y)
    else:
        features[f'{prefix}_slope'] = 0.0

    # (b) Coefficient of variation (stability)
    mean_val = np.mean(y)
    if abs(mean_val) > 1e-10:
        features[f'{prefix}_cv'] = np.std(y) / abs(mean_val)
    else:
        features[f'{prefix}_cv'] = 0.0

    # (c) First-half vs second-half delta
    mid = len(y) // 2
    first_half = np.mean(y[:mid])
    second_half = np.mean(y[mid:])
    overall_sd = np.std(y)
    if overall_sd > 0:
        features[f'{prefix}_delta'] = (second_half - first_half) / overall_sd
    else:
        features[f'{prefix}_delta'] = 0.0

    return features


# ── modality extractors (daily → weekly → trend) ────────────────

def extract_activity_weekly(uid):
    """Activity: moving_ratio and transitions per week."""
    filepath = SENSING_DIR / 'activity' / f'activity_{uid}.csv'
    if not filepath.exists():
        return {}

    df = pd.read_csv(filepath)
    col = 'timestamp' if 'timestamp' in df.columns else df.columns[0].strip()
    df.rename(columns={col: 'timestamp'}, inplace=True)
    act_col = [c for c in df.columns if 'inference' in c.lower() or 'activity' in c.lower()]
    if not act_col:
        act_col = [df.columns[1].strip()]
    df.rename(columns={act_col[0]: 'activity'}, inplace=True)
    df['activity'] = pd.to_numeric(df['activity'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'activity'])

    if len(df) < 100:
        return {}

    df['date'] = df['timestamp'].apply(ts_to_date)
    df['week'] = df['date'].apply(lambda d: d.isocalendar()[1])

    # Daily features
    daily = df.groupby('date').agg(
        moving_ratio=('activity', lambda x: (x.isin([1, 2])).mean()),
        transitions=('activity', lambda x: (x.diff().abs() > 0).sum()),
        n_samples=('activity', 'count'),
    ).reset_index()
    daily = daily[daily['n_samples'] >= 10]
    daily['week'] = daily['date'].apply(lambda d: d.isocalendar()[1])

    # Weekly aggregation
    weekly = daily.groupby('week').agg(
        moving_ratio=('moving_ratio', 'mean'),
        transitions=('transitions', 'mean'),
        n_days=('date', 'count'),
    )
    weekly = weekly[weekly['n_days'] >= 2]

    features = {}
    features.update(compute_trend_features(weekly['moving_ratio'], 'activity_moving'))
    features.update(compute_trend_features(weekly['transitions'], 'activity_transitions'))
    return features


def extract_gps_weekly(uid):
    """GPS: distance traveled and home-stay per week."""
    filepath = SENSING_DIR / 'gps' / f'gps_{uid}.csv'
    if not filepath.exists():
        return {}

    df = pd.read_csv(filepath, index_col=False)
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    # Filter by accuracy
    if 'accuracy' in df.columns:
        df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
        df = df[df['accuracy'] <= 100]

    df = df.dropna(subset=['time', 'latitude', 'longitude'])
    if len(df) < 50:
        return {}

    df['date'] = df['time'].apply(ts_to_date)
    df['week'] = df['date'].apply(lambda d: d.isocalendar()[1])

    # Home location: centroid of first week
    first_week = df['week'].min()
    home_data = df[df['week'] == first_week]
    home_lat = home_data['latitude'].median()
    home_lon = home_data['longitude'].median()

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Daily features
    daily_rows = []
    for date, day_df in df.groupby('date'):
        if len(day_df) < 5:
            continue
        # Distance traveled
        lats = day_df['latitude'].values
        lons = day_df['longitude'].values
        dists = [haversine_km(lats[i], lons[i], lats[i + 1], lons[i + 1])
                 for i in range(len(lats) - 1)]
        total_dist = sum(dists)

        # Home stay ratio: fraction of points within 0.1 km of home
        home_dists = [haversine_km(lat, lon, home_lat, home_lon)
                      for lat, lon in zip(lats, lons)]
        home_ratio = sum(1 for d in home_dists if d < 0.1) / len(home_dists)

        daily_rows.append({
            'date': date,
            'week': date.isocalendar()[1],
            'distance_km': total_dist,
            'home_ratio': home_ratio,
        })

    if len(daily_rows) < 7:
        return {}

    daily = pd.DataFrame(daily_rows)
    weekly = daily.groupby('week').agg(
        distance_km=('distance_km', 'mean'),
        home_ratio=('home_ratio', 'mean'),
        n_days=('date', 'count'),
    )
    weekly = weekly[weekly['n_days'] >= 2]

    features = {}
    features.update(compute_trend_features(weekly['distance_km'], 'gps_distance'))
    features.update(compute_trend_features(weekly['home_ratio'], 'gps_home'))
    return features


def extract_phonelock_weekly(uid):
    """PhoneLock: unlock count and screen time per week."""
    filepath = SENSING_DIR / 'phonelock' / f'phonelock_{uid}.csv'
    if not filepath.exists():
        return {}

    df = pd.read_csv(filepath)
    if len(df) < 10:
        return {}

    df = df.sort_values('start').reset_index(drop=True)

    # Unlocked sessions = gaps between locked periods
    unlock_starts = df['end'].values[:-1]
    unlock_ends = df['start'].values[1:]
    durations = unlock_ends - unlock_starts

    # Filter: 0 < duration < 12h
    valid = (durations > 0) & (durations < 43200)
    sessions = pd.DataFrame({
        'start': unlock_starts[valid],
        'duration': durations[valid],
    })

    if len(sessions) < 10:
        return {}

    sessions['date'] = sessions['start'].apply(ts_to_date)
    sessions['week'] = sessions['date'].apply(lambda d: d.isocalendar()[1])

    daily = sessions.groupby('date').agg(
        unlock_count=('duration', 'count'),
        screen_time_h=('duration', lambda x: x.sum() / 3600),
    ).reset_index()
    daily['week'] = daily['date'].apply(lambda d: d.isocalendar()[1])

    weekly = daily.groupby('week').agg(
        unlock_count=('unlock_count', 'mean'),
        screen_time_h=('screen_time_h', 'mean'),
        n_days=('date', 'count'),
    )
    weekly = weekly[weekly['n_days'] >= 2]

    features = {}
    features.update(compute_trend_features(weekly['unlock_count'], 'screen_unlock'))
    features.update(compute_trend_features(weekly['screen_time_h'], 'screen_time'))
    return features


def extract_audio_weekly(uid):
    """Audio: silence and voice ratios per week."""
    filepath = SENSING_DIR / 'audio' / f'audio_{uid}.csv'
    if not filepath.exists():
        return {}

    # Read in chunks (audio files are large)
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=500000):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    col = 'timestamp' if 'timestamp' in df.columns else df.columns[0].strip()
    df.rename(columns={col: 'timestamp'}, inplace=True)
    inf_col = [c for c in df.columns if 'inference' in c.lower() or 'audio' in c.lower()]
    if not inf_col:
        inf_col = [df.columns[1].strip()]
    df.rename(columns={inf_col[0]: 'audio'}, inplace=True)
    df['audio'] = pd.to_numeric(df['audio'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'audio'])

    if len(df) < 100:
        return {}

    # Audio codes: 0=silence, 1=voice, 2=noise
    df['date'] = df['timestamp'].apply(ts_to_date)
    df['week'] = df['date'].apply(lambda d: d.isocalendar()[1])

    daily = df.groupby('date').agg(
        silence_ratio=('audio', lambda x: (x == 0).mean()),
        voice_ratio=('audio', lambda x: (x == 1).mean()),
        n_samples=('audio', 'count'),
    ).reset_index()
    daily = daily[daily['n_samples'] >= 50]
    daily['week'] = daily['date'].apply(lambda d: d.isocalendar()[1])

    weekly = daily.groupby('week').agg(
        silence_ratio=('silence_ratio', 'mean'),
        voice_ratio=('voice_ratio', 'mean'),
        n_days=('date', 'count'),
    )
    weekly = weekly[weekly['n_days'] >= 2]

    features = {}
    features.update(compute_trend_features(weekly['silence_ratio'], 'audio_silence'))
    features.update(compute_trend_features(weekly['voice_ratio'], 'audio_voice'))
    return features


def extract_bluetooth_weekly(uid):
    """Bluetooth: unique devices per week (social proximity)."""
    filepath = SENSING_DIR / 'bluetooth' / f'bt_{uid}.csv'
    if not filepath.exists():
        return {}

    df = pd.read_csv(filepath)
    col = 'time' if 'time' in df.columns else df.columns[0].strip()
    df.rename(columns={col: 'time'}, inplace=True)

    if 'MAC' not in df.columns:
        return {}

    df = df.dropna(subset=['time', 'MAC'])
    if len(df) < 50:
        return {}

    df['date'] = df['time'].apply(ts_to_date)
    df['week'] = df['date'].apply(lambda d: d.isocalendar()[1])

    daily = df.groupby('date').agg(
        unique_devices=('MAC', 'nunique'),
    ).reset_index()
    daily['week'] = daily['date'].apply(lambda d: d.isocalendar()[1])

    weekly = daily.groupby('week').agg(
        unique_devices=('unique_devices', 'mean'),
        n_days=('date', 'count'),
    )
    weekly = weekly[weekly['n_days'] >= 2]

    features = {}
    features.update(compute_trend_features(weekly['unique_devices'], 'bt_devices'))
    return features


def extract_conversation_weekly(uid):
    """Conversation: count and duration per week."""
    filepath = SENSING_DIR / 'conversation' / f'conversation_{uid}.csv'
    if not filepath.exists():
        return {}

    df = pd.read_csv(filepath)
    start_col = [c for c in df.columns if 'start' in c.lower()][0]
    end_col = [c for c in df.columns if 'end' in c.lower()][0]
    df.rename(columns={start_col: 'start', end_col: 'end'}, inplace=True)
    df['duration'] = df['end'] - df['start']
    df = df[(df['duration'] > 0) & (df['duration'] < 7200)]  # < 2h

    if len(df) < 5:
        return {}

    df['date'] = df['start'].apply(ts_to_date)
    df['week'] = df['date'].apply(lambda d: d.isocalendar()[1])

    daily = df.groupby('date').agg(
        convo_count=('duration', 'count'),
        convo_duration=('duration', 'mean'),
    ).reset_index()
    daily['week'] = daily['date'].apply(lambda d: d.isocalendar()[1])

    weekly = daily.groupby('week').agg(
        convo_count=('convo_count', 'mean'),
        convo_duration=('convo_duration', 'mean'),
        n_days=('date', 'count'),
    )
    weekly = weekly[weekly['n_days'] >= 2]

    features = {}
    features.update(compute_trend_features(weekly['convo_count'], 'convo_count'))
    features.update(compute_trend_features(weekly['convo_duration'], 'convo_duration'))
    return features


def extract_app_weekly(uid):
    """App usage: switches and unique apps per week."""
    filepath = APP_DIR / f'running_app_{uid}.csv'
    if not filepath.exists():
        return {}

    df = pd.read_csv(filepath)
    ts_col = [c for c in df.columns if 'time' in c.lower()]
    if not ts_col:
        return {}
    df.rename(columns={ts_col[0]: 'timestamp'}, inplace=True)

    # Find package name column (RUNNING_TASKS_topActivity_mPackage)
    pkg_col = [c for c in df.columns if 'mPackage' in c or 'package' in c.lower()]
    if not pkg_col:
        return {}
    df.rename(columns={pkg_col[0]: 'package'}, inplace=True)

    # Remove system apps
    system_apps = ['com.android.systemui', 'com.android.launcher',
                   'edu.dartmouth.cs.paco', 'com.android.settings',
                   'com.google.android.apps.paco']
    df = df[~df['package'].isin(system_apps)]

    if len(df) < 50:
        return {}

    df['date'] = df['timestamp'].apply(ts_to_date)
    df['week'] = df['date'].apply(lambda d: d.isocalendar()[1])

    daily = df.groupby('date').agg(
        app_switches=('package', 'count'),
        unique_apps=('package', 'nunique'),
    ).reset_index()
    daily = daily[daily['app_switches'] >= 2]
    daily['week'] = daily['date'].apply(lambda d: d.isocalendar()[1])

    weekly = daily.groupby('week').agg(
        app_switches=('app_switches', 'mean'),
        unique_apps=('unique_apps', 'mean'),
        n_days=('date', 'count'),
    )
    weekly = weekly[weekly['n_days'] >= 2]

    features = {}
    features.update(compute_trend_features(weekly['app_switches'], 'app_switches'))
    features.update(compute_trend_features(weekly['unique_apps'], 'app_uniqueapps'))
    return features


def extract_communication_weekly(uid):
    """Call + SMS: daily communication volume per week."""
    call_path = CALL_DIR / f'call_log_{uid}.csv'
    sms_path = SMS_DIR / f'sms_{uid}.csv'

    daily_counts = {}

    # Calls
    if call_path.exists():
        try:
            df = pd.read_csv(call_path)
            ts_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
            if ts_col:
                df.rename(columns={ts_col[0]: 'timestamp'}, inplace=True)
                # call_log timestamps may be in milliseconds
                if df['timestamp'].median() > 1e12:
                    df['timestamp'] = df['timestamp'] / 1000
                df = df.dropna(subset=['timestamp'])
                df['date'] = df['timestamp'].apply(ts_to_date)
                for date, group in df.groupby('date'):
                    daily_counts.setdefault(date, 0)
                    daily_counts[date] += len(group)
        except Exception:
            pass

    # SMS
    if sms_path.exists():
        try:
            df = pd.read_csv(sms_path)
            ts_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
            if ts_col:
                df.rename(columns={ts_col[0]: 'timestamp'}, inplace=True)
                if df['timestamp'].median() > 1e12:
                    df['timestamp'] = df['timestamp'] / 1000
                df = df.dropna(subset=['timestamp'])
                df['date'] = df['timestamp'].apply(ts_to_date)
                for date, group in df.groupby('date'):
                    daily_counts.setdefault(date, 0)
                    daily_counts[date] += len(group)
        except Exception:
            pass

    if len(daily_counts) < 7:
        return {}

    daily = pd.DataFrame([
        {'date': d, 'comm_count': c} for d, c in daily_counts.items()
    ])
    daily['week'] = daily['date'].apply(lambda d: d.isocalendar()[1])

    weekly = daily.groupby('week').agg(
        comm_count=('comm_count', 'mean'),
        n_days=('date', 'count'),
    )
    weekly = weekly[weekly['n_days'] >= 2]

    features = {}
    features.update(compute_trend_features(weekly['comm_count'], 'comm_volume'))
    return features


# ── main ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 8: WEEKLY TEMPORAL FEATURES")
    print("=" * 60)

    extractors = [
        ('Activity', extract_activity_weekly),
        ('GPS', extract_gps_weekly),
        ('PhoneLock', extract_phonelock_weekly),
        ('Audio', extract_audio_weekly),
        ('Bluetooth', extract_bluetooth_weekly),
        ('Conversation', extract_conversation_weekly),
        ('App Usage', extract_app_weekly),
        ('Communication', extract_communication_weekly),
    ]

    all_rows = []

    for uid in TARGET_USERS:
        print(f"\n  {uid}:", end='')
        user_features = {'uid': uid}

        for name, extractor in extractors:
            feats = extractor(uid)
            if feats:
                user_features.update(feats)
                print(f" {name}({len(feats)})", end='')
            else:
                print(f" {name}(-)", end='')

        n_temporal = len(user_features) - 1  # exclude uid
        print(f"  → {n_temporal} features")
        all_rows.append(user_features)

    result = pd.DataFrame(all_rows)

    # Report coverage
    print("\n" + "─" * 60)
    print("Feature Coverage:")
    temporal_cols = [c for c in result.columns if c != 'uid']
    for col in sorted(temporal_cols):
        n_valid = result[col].notna().sum()
        print(f"  {col:35s} {n_valid:2d}/{len(result)} users")

    # Summary
    print(f"\n  Total: {len(result)} users × {len(temporal_cols)} temporal features")
    print(f"  Features per user: "
          f"mean={result[temporal_cols].notna().sum(axis=1).mean():.1f}, "
          f"min={result[temporal_cols].notna().sum(axis=1).min()}")

    # Save
    output_path = OUTPUT_DIR / 'temporal_features.parquet'
    result.to_parquet(output_path, index=False)
    print(f"\n  Output: {output_path.relative_to(project_root)}")

    print("\n" + "=" * 60)
    print("PHASE 8 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
