#!/usr/bin/env python3
"""
Phase 12 Step 2: Extract Behavioral Features from NetHealth Fitbit + Communication Data

Extracts per-user features from:
  - Fitbit Activity (daily steps, active minutes, HR zones)
  - Fitbit Sleep (duration, efficiency, regularity)
  - Communication Events (calls, SMS, contacts)

Time window: First semester only (2015-08-16 to 2015-12-18)

Input:  data/raw/nethealth/FitbitActivity(1-30-20).csv
        data/raw/nethealth/FitbitSleep(1-30-20).csv
        data/raw/nethealth/CommEvents(2-28-20).csv
Output: data/processed/nethealth/features/fitbit_activity_features.parquet
        data/processed/nethealth/features/fitbit_sleep_features.parquet
        data/processed/nethealth/features/communication_features.parquet
        data/processed/nethealth/features/combined_features.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent

NH_DATA_DIR = project_root / 'data' / 'raw' / 'nethealth'
NH_FEATURE_DIR = project_root / 'data' / 'processed' / 'nethealth' / 'features'
NH_FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# First semester: orientation (2015-08-16) to end of finals (2015-12-18)
SEM1_START = '2015-08-16'
SEM1_END = '2015-12-18'
MIN_DAYS = 14  # Minimum days of data required


# ──────────────────────────────────────────────────────────────────────
# Fitbit Activity
# ──────────────────────────────────────────────────────────────────────

def extract_fitbit_activity() -> pd.DataFrame:
    """Extract per-user activity features from daily Fitbit data."""
    df = pd.read_csv(NH_DATA_DIR / 'FitbitActivity(1-30-20).csv')
    df['datadate'] = pd.to_datetime(df['datadate'])

    # Filter to first semester
    df = df[(df['datadate'] >= SEM1_START) & (df['datadate'] <= SEM1_END)].copy()

    # Only keep days with reasonable compliance (>50% wear time)
    df = df[df['complypercent'] >= 50].copy()

    features = []
    for egoid, user_df in df.groupby('egoid'):
        if len(user_df) < MIN_DAYS:
            continue

        total_active = (user_df['lightlyactiveminutes'] +
                        user_df['fairlyactiveminutes'] +
                        user_df['veryactiveminutes'])
        total_time = total_active + user_df['sedentaryminutes']
        active_ratio = total_active / total_time.replace(0, np.nan)

        feat = {
            'egoid': egoid,
            'steps_mean': user_df['steps'].mean(),
            'steps_std': user_df['steps'].std(),
            'steps_cv': user_df['steps'].std() / max(user_df['steps'].mean(), 1),
            'sedentary_min_mean': user_df['sedentaryminutes'].mean(),
            'light_active_min_mean': user_df['lightlyactiveminutes'].mean(),
            'fairly_active_min_mean': user_df['fairlyactiveminutes'].mean(),
            'very_active_min_mean': user_df['veryactiveminutes'].mean(),
            'total_active_min_mean': total_active.mean(),
            'active_ratio_mean': active_ratio.mean(),
            'fatburn_min_mean': user_df['fatburnmins'].mean(),
            'cardio_min_mean': user_df['cardiomins'].mean(),
            'activity_n_days': len(user_df),
        }
        features.append(feat)

    result = pd.DataFrame(features)
    print(f"  Fitbit Activity: {len(result)} users with ≥{MIN_DAYS} days")
    return result


# ──────────────────────────────────────────────────────────────────────
# Fitbit Sleep
# ──────────────────────────────────────────────────────────────────────

def extract_fitbit_sleep() -> pd.DataFrame:
    """Extract per-user sleep features from Fitbit sleep data."""
    df = pd.read_csv(NH_DATA_DIR / 'FitbitSleep(1-30-20).csv')
    df['dataDate'] = pd.to_datetime(df['dataDate'])

    # Filter to first semester
    df = df[(df['dataDate'] >= SEM1_START) & (df['dataDate'] <= SEM1_END)].copy()

    # Parse bedtime for circadian analysis
    df['timetobed_parsed'] = pd.to_datetime(df['timetobed'], format='%H:%M:%S', errors='coerce')
    df['bed_hour'] = df['timetobed_parsed'].dt.hour + df['timetobed_parsed'].dt.minute / 60

    features = []
    for egoid, user_df in df.groupby('egoid'):
        if len(user_df) < MIN_DAYS:
            continue

        feat = {
            'egoid': egoid,
            'sleep_duration_mean': user_df['minsasleep'].mean(),
            'sleep_duration_std': user_df['minsasleep'].std(),
            'sleep_interruptions_mean': user_df['minsawake'].mean(),
            'sleep_efficiency_mean': user_df['Efficiency'].mean(),
            'sleep_onset_mean': user_df['bed_hour'].mean(),
            'sleep_regularity': user_df['bed_hour'].std(),  # Lower = more regular
            'time_to_fall_asleep_mean': user_df['minstofallasleep'].mean(),
            'sleep_n_days': len(user_df),
        }
        features.append(feat)

    result = pd.DataFrame(features)
    print(f"  Fitbit Sleep: {len(result)} users with ≥{MIN_DAYS} days")
    return result


# ──────────────────────────────────────────────────────────────────────
# Communication Events
# ──────────────────────────────────────────────────────────────────────

def extract_communication() -> pd.DataFrame:
    """Extract per-user communication features from call/SMS logs.

    Reads the 6GB file in chunks to manage memory.
    """
    chunks = pd.read_csv(
        NH_DATA_DIR / 'CommEvents(2-28-20).csv',
        usecols=['egoid', 'date', 'outgoing', 'eventtype', 'duration', 'alterid'],
        chunksize=5_000_000,
        dtype={'egoid': 'int64', 'alterid': 'float64'},
    )

    # Accumulate per-user daily stats
    user_stats = {}

    for chunk in chunks:
        chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
        chunk = chunk[(chunk['date'] >= SEM1_START) & (chunk['date'] <= SEM1_END)].copy()

        if len(chunk) == 0:
            continue

        for egoid, user_df in chunk.groupby('egoid'):
            if egoid not in user_stats:
                user_stats[egoid] = {
                    'call_count': 0, 'call_outgoing': 0, 'call_duration_total': 0,
                    'sms_count': 0, 'sms_outgoing': 0,
                    'contacts': set(), 'days': set(),
                }

            stats = user_stats[egoid]
            calls = user_df[user_df['eventtype'] == 'Call']
            sms = user_df[user_df['eventtype'].isin(['SMS', 'MMS'])]

            stats['call_count'] += len(calls)
            stats['call_outgoing'] += (calls['outgoing'] == 'Yes').sum()
            stats['call_duration_total'] += calls['duration'].sum()
            stats['sms_count'] += len(sms)
            stats['sms_outgoing'] += (sms['outgoing'] == 'Yes').sum()
            stats['contacts'].update(user_df['alterid'].dropna().unique())
            stats['days'].update(user_df['date'].dt.date.unique())

    # Convert to per-user features
    features = []
    for egoid, stats in user_stats.items():
        n_days = len(stats['days'])
        if n_days < MIN_DAYS:
            continue

        feat = {
            'egoid': egoid,
            'call_count_per_day': stats['call_count'] / n_days,
            'call_duration_per_day': stats['call_duration_total'] / max(n_days, 1),
            'call_outgoing_ratio': stats['call_outgoing'] / max(stats['call_count'], 1),
            'sms_count_per_day': stats['sms_count'] / n_days,
            'sms_outgoing_ratio': stats['sms_outgoing'] / max(stats['sms_count'], 1),
            'total_unique_contacts': len(stats['contacts']),
            'total_comm_per_day': (stats['call_count'] + stats['sms_count']) / n_days,
            'comm_n_days': n_days,
        }
        features.append(feat)

    result = pd.DataFrame(features)
    print(f"  Communication: {len(result)} users with ≥{MIN_DAYS} days")
    return result


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 12 STEP 2: EXTRACT NETHEALTH FEATURES")
    print(f"  Window: {SEM1_START} to {SEM1_END}")
    print("=" * 60)

    # 1. Fitbit Activity
    print("\n[1/3] Fitbit Activity...")
    activity = extract_fitbit_activity()

    # 2. Fitbit Sleep
    print("\n[2/3] Fitbit Sleep...")
    sleep = extract_fitbit_sleep()

    # 3. Communication
    print("\n[3/3] Communication Events (chunked read)...")
    comm = extract_communication()

    # Save individual feature sets
    activity.to_parquet(NH_FEATURE_DIR / 'fitbit_activity_features.parquet', index=False)
    sleep.to_parquet(NH_FEATURE_DIR / 'fitbit_sleep_features.parquet', index=False)
    comm.to_parquet(NH_FEATURE_DIR / 'communication_features.parquet', index=False)

    # Merge into combined features
    combined = activity.merge(sleep, on='egoid', how='outer')
    combined = combined.merge(comm, on='egoid', how='outer')

    combined.to_parquet(NH_FEATURE_DIR / 'combined_features.parquet', index=False)

    # Summary
    print(f"\n{'─' * 60}")
    print(f"FEATURE SUMMARY")
    print(f"{'─' * 60}")
    print(f"  Activity features: {len(activity)} users × {len(activity.columns)-1} features")
    print(f"  Sleep features:    {len(sleep)} users × {len(sleep.columns)-1} features")
    print(f"  Comm features:     {len(comm)} users × {len(comm.columns)-1} features")
    print(f"  Combined:          {len(combined)} users × {len(combined.columns)-1} features")

    # Feature distributions
    print(f"\n  Key feature distributions:")
    for col in ['steps_mean', 'sleep_duration_mean', 'sleep_efficiency_mean',
                'total_comm_per_day', 'total_unique_contacts']:
        if col in combined.columns:
            vals = combined[col].dropna()
            print(f"    {col:30s}  N={len(vals):3d}  M={vals.mean():8.1f}  SD={vals.std():8.1f}")

    print(f"\n  Saved to: data/processed/nethealth/features/")
    print(f"\n{'=' * 60}")
    print("PHASE 12 STEP 2 COMPLETE")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
