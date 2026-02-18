#!/usr/bin/env python3
"""
Step 1: Extract Behavioral Features from Raw Sensor Data

Pipeline:
  1.  Scan raw data to build user timelines (temporal alignment)
  2.  Extract GPS mobility features
  3.  Extract app usage features
  4.  Extract communication features (call + SMS)
  5.  Extract physical activity features
  6.  Extract phone lock/screen features
  7.  Extract Bluetooth social proximity features
  8.  Extract conversation (face-to-face) features
  9.  Extract Piazza academic engagement features
  10. Extract deadline features
  11. Extract EMA mood features
  12. Extract EMA stress features
  13. Extract WiFi location-proxy features
  14. Extract audio environment features
  15. Merge all modalities into combined feature matrix

Input:  data/raw/dataset/
Output: data/processed/features/combined_features.parquet
        data/processed/features/*_features.parquet (per-modality)
        data/interim/user_timelines/
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features import (
    GPSMobilityExtractor, AppUsageExtractor,
    CommunicationExtractor, ActivityExtractor, TemporalAligner,
    PhoneLockExtractor, BluetoothExtractor, ConversationExtractor,
    PiazzaExtractor, DeadlineExtractor,
    EMAMoodExtractor, EMAStressExtractor,
    WiFiExtractor, AudioExtractor,
)

DATA_ROOT = project_root / 'data' / 'raw' / 'dataset'
INTERIM_DIR = project_root / 'data' / 'interim' / 'user_timelines'
FEATURE_DIR = project_root / 'data' / 'processed' / 'features'

for d in [INTERIM_DIR, FEATURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TIMELINES_FILE = str(INTERIM_DIR / 'user_timelines.json')
N_STEPS = 15


def step_header(n, label):
    print(f"\n[Step {n}/{N_STEPS}] {label}")
    print("─" * 60)


def save_and_report(df, name, path):
    df.to_parquet(path, index=False)
    print(f"  → {len(df)} users, {len(df.columns)-1} features → {path.name}")
    return df


# ──────────────────────────────────────────────────────────────────────

def step1_build_timelines() -> list:
    step_header(1, "Building user timelines")
    phq9 = pd.read_csv(DATA_ROOT / 'survey' / 'PHQ-9.csv')
    user_ids = sorted(phq9[phq9['type'] == 'pre']['uid'].unique().tolist())
    print(f"  Users with PHQ-9 pre-assessment: {len(user_ids)}")

    aligner = TemporalAligner(str(DATA_ROOT))
    timelines = aligner.get_all_user_timelines(user_ids)
    aligner.save_timelines(timelines, TIMELINES_FILE)

    quality = aligner.get_data_quality_report(timelines)
    quality.to_csv(INTERIM_DIR / 'data_quality_report.csv', index=False)

    valid_users = sorted(timelines.keys())
    pd.DataFrame({'uid': valid_users}).to_csv(
        INTERIM_DIR / 'valid_users.csv', index=False)
    print(f"  Valid users: {len(valid_users)}")
    return valid_users


def step2_gps(user_ids):
    step_header(2, "GPS mobility features")
    ext = GPSMobilityExtractor(
        data_dir=str(DATA_ROOT / 'sensing' / 'gps'),
        timelines_file=TIMELINES_FILE,
        accuracy_threshold=100.0, min_points_per_day=5)
    return save_and_report(ext.extract_all_users(user_ids),
                           'gps', FEATURE_DIR / 'gps_features.parquet')


def step3_app(user_ids):
    step_header(3, "App usage features")
    ext = AppUsageExtractor(
        data_dir=str(DATA_ROOT / 'app_usage'),
        timelines_file=TIMELINES_FILE)
    return save_and_report(ext.extract_all_users(user_ids),
                           'app', FEATURE_DIR / 'app_features.parquet')


def step4_comm(user_ids):
    step_header(4, "Communication features (call + SMS)")
    ext = CommunicationExtractor(
        call_dir=str(DATA_ROOT / 'call_log'),
        sms_dir=str(DATA_ROOT / 'sms'),
        timelines_file=TIMELINES_FILE)
    return save_and_report(ext.extract_all_users(user_ids),
                           'comm', FEATURE_DIR / 'communication_features.parquet')


def step5_activity(user_ids):
    step_header(5, "Physical activity features")
    ext = ActivityExtractor(
        data_dir=str(DATA_ROOT / 'sensing' / 'activity'),
        timelines_file=TIMELINES_FILE, chunksize=100000)
    return save_and_report(ext.extract_all_users(user_ids),
                           'activity', FEATURE_DIR / 'activity_features.parquet')


def step6_phonelock(user_ids):
    step_header(6, "Phone lock / screen features")
    ext = PhoneLockExtractor(
        data_dir=str(DATA_ROOT / 'sensing' / 'phonelock'),
        timelines_file=TIMELINES_FILE)
    return save_and_report(ext.extract_all_users(user_ids),
                           'phonelock', FEATURE_DIR / 'phonelock_features.parquet')


def step7_bluetooth(user_ids):
    step_header(7, "Bluetooth social proximity features")
    ext = BluetoothExtractor(
        data_dir=str(DATA_ROOT / 'sensing' / 'bluetooth'),
        timelines_file=TIMELINES_FILE)
    return save_and_report(ext.extract_all_users(user_ids),
                           'bluetooth', FEATURE_DIR / 'bluetooth_features.parquet')


def step8_conversation(user_ids):
    step_header(8, "Conversation (face-to-face) features")
    ext = ConversationExtractor(
        data_dir=str(DATA_ROOT / 'sensing' / 'conversation'),
        timelines_file=TIMELINES_FILE)
    return save_and_report(ext.extract_all_users(user_ids),
                           'conversation', FEATURE_DIR / 'conversation_features.parquet')


def step9_piazza(user_ids):
    step_header(9, "Piazza academic engagement features")
    ext = PiazzaExtractor(filepath=str(DATA_ROOT / 'education' / 'piazza.csv'))
    return save_and_report(ext.extract_all_users(user_ids),
                           'piazza', FEATURE_DIR / 'piazza_features.parquet')


def step10_deadlines(user_ids):
    step_header(10, "Deadline features")
    ext = DeadlineExtractor(filepath=str(DATA_ROOT / 'education' / 'deadlines.csv'))
    return save_and_report(ext.extract_all_users(user_ids),
                           'deadline', FEATURE_DIR / 'deadline_features.parquet')


def step11_ema_mood(user_ids):
    step_header(11, "EMA mood features")
    ext = EMAMoodExtractor(data_dir=str(DATA_ROOT / 'EMA' / 'response' / 'Mood'))
    return save_and_report(ext.extract_all_users(user_ids),
                           'ema_mood', FEATURE_DIR / 'ema_mood_features.parquet')


def step12_ema_stress(user_ids):
    step_header(12, "EMA stress features")
    ext = EMAStressExtractor(data_dir=str(DATA_ROOT / 'EMA' / 'response' / 'Stress'))
    return save_and_report(ext.extract_all_users(user_ids),
                           'ema_stress', FEATURE_DIR / 'ema_stress_features.parquet')


def step13_wifi(user_ids):
    step_header(13, "WiFi location-proxy features")
    ext = WiFiExtractor(
        data_dir=str(DATA_ROOT / 'sensing' / 'wifi'),
        timelines_file=TIMELINES_FILE)
    return save_and_report(ext.extract_all_users(user_ids),
                           'wifi', FEATURE_DIR / 'wifi_features.parquet')


def step14_audio(user_ids):
    step_header(14, "Audio environment features")
    ext = AudioExtractor(
        data_dir=str(DATA_ROOT / 'sensing' / 'audio'),
        timelines_file=TIMELINES_FILE, chunksize=500000)
    return save_and_report(ext.extract_all_users(user_ids),
                           'audio', FEATURE_DIR / 'audio_features.parquet')


def step15_merge(dfs: dict) -> pd.DataFrame:
    step_header(15, "Merging all modalities")

    combined = None
    for name, df in dfs.items():
        if combined is None:
            combined = df
        else:
            combined = combined.merge(df, on='uid', how='outer')

    # Fill missing communication features with 0
    comm_cols = [c for c in combined.columns
                 if c.startswith(('call_', 'sms_', 'total_unique', 'total_communication', 'communication_'))]
    for col in comm_cols:
        combined[col] = combined[col].fillna(0)

    combined.to_parquet(FEATURE_DIR / 'combined_features.parquet', index=False)
    n_features = len(combined.columns) - 1
    n_missing = combined.iloc[:, 1:].isnull().sum().sum()
    print(f"  → {len(combined)} users, {n_features} features")
    print(f"  Missing values: {n_missing}")
    return combined


# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 1: EXTRACT ALL FEATURES FROM RAW DATA")
    print("=" * 60)

    user_ids = step1_build_timelines()

    dfs = {}
    dfs['gps'] = step2_gps(user_ids)
    dfs['app'] = step3_app(user_ids)
    dfs['comm'] = step4_comm(user_ids)
    dfs['activity'] = step5_activity(user_ids)
    dfs['phonelock'] = step6_phonelock(user_ids)
    dfs['bluetooth'] = step7_bluetooth(user_ids)
    dfs['conversation'] = step8_conversation(user_ids)
    dfs['piazza'] = step9_piazza(user_ids)
    dfs['deadline'] = step10_deadlines(user_ids)
    dfs['ema_mood'] = step11_ema_mood(user_ids)
    dfs['ema_stress'] = step12_ema_stress(user_ids)
    dfs['wifi'] = step13_wifi(user_ids)
    dfs['audio'] = step14_audio(user_ids)

    combined = step15_merge(dfs)

    # Summary
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Output: data/processed/features/combined_features.parquet")
    print(f"  Users:  {len(combined)}")
    print(f"  Total features: {len(combined.columns) - 1}")

    modality_prefixes = {
        'GPS':          ('location_', 'distance_', 'radius_', 'max_distance', 'home_', 'n_significant', 'movement_', 'gps_'),
        'App':          ('app_', 'unique_apps', 'night_usage', 'weekend_'),
        'Communication':('call_', 'sms_', 'total_unique', 'total_communication', 'communication_', 'comm_'),
        'Activity':     ('still_', 'moving_', 'sedentary_', 'activity_'),
        'PhoneLock':    ('unlock_', 'session_duration', 'screen_time'),
        'Bluetooth':    ('bt_',),
        'Conversation': ('convo_',),
        'Piazza':       ('piazza_',),
        'Deadline':     ('deadline_',),
        'EMA Mood':     ('ema_happy', 'ema_sad'),
        'EMA Stress':   ('ema_stress',),
        'WiFi':         ('wifi_',),
        'Audio':        ('audio_',),
    }
    for mod, prefixes in modality_prefixes.items():
        count = sum(1 for c in combined.columns if c.startswith(prefixes))
        if count > 0:
            print(f"    {mod:15s}: {count} features")


if __name__ == '__main__':
    main()
