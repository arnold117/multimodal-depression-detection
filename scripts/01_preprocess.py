#!/usr/bin/env python3
"""
Step 1: Data Preprocessing Pipeline

Loads raw StudentLife data and extracts features for analysis.

Outputs three types of data:
1. User-level aggregated features (sensor_features.parquet)
2. Daily-level features per modality (daily_features/*.parquet)
3. Sliding window features (temporal_features.parquet)

Usage:
    python scripts/01_preprocess.py --data-dir data/raw/dataset

Outputs:
    - outputs/processed/sensor_features.parquet     (user-level aggregates)
    - outputs/processed/survey_scores.parquet       (PHQ-9 scores)
    - outputs/processed/grades.parquet              (academic grades)
    - outputs/processed/daily_features/
        - gps_daily.parquet                         (daily GPS features)
        - phone_daily.parquet                       (daily phone features)
        - activity_daily.parquet                    (daily activity features)
        - conversation_daily.parquet                (daily conversation features)
    - outputs/processed/temporal_features.parquet   (sliding window features)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from loguru import logger
from omegaconf import OmegaConf
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders.studentlife import StudentLifeLoader


# =============================================================================
# Configuration
# =============================================================================

WINDOW_SIZES = [1, 3, 7]  # Days
AGGREGATIONS = ['mean', 'std', 'min', 'max', 'trend']


def setup_logging(output_dir: Path) -> None:
    """Configure logging."""
    log_file = output_dir / "logs" / f"preprocess_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_file, rotation="10 MB", level="DEBUG")
    logger.info(f"Logging to {log_file}")


# =============================================================================
# Helper Functions
# =============================================================================

def timestamp_to_date(ts: float) -> Optional[datetime]:
    """Convert Unix timestamp to date."""
    try:
        if np.isnan(ts) or ts <= 0:
            return None
        return datetime.fromtimestamp(ts).date()
    except:
        return None


def compute_trend(values: np.ndarray) -> float:
    """Compute linear trend (slope) of values."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    try:
        slope, _, _, _, _ = stats.linregress(x, values)
        return slope if not np.isnan(slope) else 0.0
    except:
        return 0.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two GPS points in km."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# =============================================================================
# Daily Feature Extraction (Per Modality)
# =============================================================================

def extract_gps_daily(gps_df: pl.DataFrame, user_id: str) -> List[Dict]:
    """
    Extract daily GPS features for a user.

    Features per day:
    - location_entropy: diversity of visited locations
    - home_stay_ratio: proportion at most common location
    - n_locations: number of unique locations
    - total_distance: km traveled
    - radius_of_gyration: spread of movement
    """
    user_gps = gps_df.filter(pl.col("user_id") == user_id)

    if len(user_gps) < 10:
        return []

    if "latitude" not in user_gps.columns or "longitude" not in user_gps.columns:
        return []

    if "time" not in user_gps.columns:
        return []

    daily_features = []

    # Get timestamps and coordinates
    times = user_gps["time"].to_numpy()
    lats = user_gps["latitude"].to_numpy()
    lons = user_gps["longitude"].to_numpy()

    # Filter valid data
    valid = (np.abs(lats) <= 90) & (np.abs(lons) <= 180) & ~np.isnan(lats) & ~np.isnan(lons) & ~np.isnan(times)
    times = times[valid]
    lats = lats[valid]
    lons = lons[valid]

    if len(times) < 10:
        return []

    # Group by date
    dates = [timestamp_to_date(t) for t in times]
    unique_dates = sorted(set(d for d in dates if d is not None))

    for date in unique_dates:
        mask = [d == date for d in dates]
        day_lats = lats[mask]
        day_lons = lons[mask]

        if len(day_lats) < 5:
            continue

        try:
            # Location clusters (rounded coordinates)
            lat_bins = np.round(day_lats, 3)
            lon_bins = np.round(day_lons, 3)
            locations = list(zip(lat_bins, lon_bins))
            loc_counts = Counter(locations)
            total = sum(loc_counts.values())

            # Entropy
            probs = np.array(list(loc_counts.values())) / total
            entropy = -np.sum(probs * np.log(probs + 1e-10))

            # Home stay ratio
            most_common_count = loc_counts.most_common(1)[0][1]
            home_ratio = most_common_count / total

            # Number of locations
            n_locations = len(loc_counts)

            # Total distance
            total_dist = 0
            for i in range(1, len(day_lats)):
                total_dist += haversine_distance(
                    day_lats[i-1], day_lons[i-1],
                    day_lats[i], day_lons[i]
                )

            # Radius of gyration
            center_lat = np.mean(day_lats)
            center_lon = np.mean(day_lons)
            distances_from_center = [
                haversine_distance(center_lat, center_lon, lat, lon)
                for lat, lon in zip(day_lats, day_lons)
            ]
            radius_gyration = np.sqrt(np.mean(np.array(distances_from_center)**2))

            daily_features.append({
                "user_id": user_id,
                "date": date,
                "location_entropy": entropy,
                "home_stay_ratio": home_ratio,
                "n_locations": n_locations,
                "total_distance": total_dist,
                "radius_of_gyration": radius_gyration
            })

        except Exception as e:
            logger.debug(f"GPS daily extraction failed for {user_id} on {date}: {e}")
            continue

    return daily_features


def extract_phone_daily(phonelock_df: pl.DataFrame, user_id: str) -> List[Dict]:
    """
    Extract daily phone usage features.

    Features per day:
    - unlock_count: number of phone unlocks
    - screen_time: total hours screen on
    - session_duration_mean: average session length (minutes)
    - session_duration_std: variability in session length
    - night_usage_ratio: proportion of usage during 0-5am
    """
    user_phone = phonelock_df.filter(pl.col("user_id") == user_id)

    if len(user_phone) < 5:
        return []

    if "start" not in user_phone.columns or "end" not in user_phone.columns:
        return []

    daily_features = []

    starts = user_phone["start"].to_numpy()
    ends = user_phone["end"].to_numpy()

    # Filter valid sessions
    valid = (starts > 0) & (ends > starts) & ~np.isnan(starts) & ~np.isnan(ends)
    starts = starts[valid]
    ends = ends[valid]

    if len(starts) == 0:
        return []

    # Group by date
    dates = [timestamp_to_date(s) for s in starts]
    unique_dates = sorted(set(d for d in dates if d is not None))

    for date in unique_dates:
        mask = [d == date for d in dates]
        day_starts = starts[mask]
        day_ends = ends[mask]

        if len(day_starts) == 0:
            continue

        try:
            durations = day_ends - day_starts

            # Unlock count
            unlock_count = len(day_starts)

            # Screen time (hours)
            screen_time = np.sum(durations) / 3600

            # Session duration stats (minutes)
            session_mean = np.mean(durations) / 60
            session_std = np.std(durations) / 60 if len(durations) > 1 else 0

            # Night usage (0-5am)
            night_time = 0
            for s, e in zip(day_starts, day_ends):
                try:
                    start_dt = datetime.fromtimestamp(s)
                    if start_dt.hour < 5:
                        night_time += (e - s)
                except:
                    pass

            night_ratio = night_time / max(1, np.sum(durations))

            daily_features.append({
                "user_id": user_id,
                "date": date,
                "unlock_count": unlock_count,
                "screen_time": screen_time,
                "session_duration_mean": session_mean,
                "session_duration_std": session_std,
                "night_usage_ratio": night_ratio
            })

        except Exception as e:
            logger.debug(f"Phone daily extraction failed for {user_id} on {date}: {e}")
            continue

    return daily_features


def extract_activity_daily(activity_df: pl.DataFrame, user_id: str) -> List[Dict]:
    """
    Extract daily activity features.

    Features per day:
    - still_ratio: proportion of time stationary
    - walking_ratio: proportion of time walking
    - running_ratio: proportion of time running
    - activity_transitions: number of state changes
    """
    user_act = activity_df.filter(pl.col("user_id") == user_id)

    if len(user_act) < 10:
        return []

    # Find activity inference column (handle CSV column name variations)
    activity_col = None
    for col in ["activity_inference", " activity inference", "activity inference"]:
        if col in user_act.columns:
            activity_col = col
            break

    if activity_col is None:
        return []

    # Try to find timestamp column
    time_col = None
    for col in ["timestamp", "time", "start_timestamp"]:
        if col in user_act.columns:
            time_col = col
            break

    if time_col is None:
        return []

    daily_features = []

    times = user_act[time_col].to_numpy()
    activities = user_act[activity_col].to_numpy()

    # Filter valid
    valid = ~np.isnan(times) & ~np.isnan(activities)
    times = times[valid]
    activities = activities[valid]

    if len(times) < 10:
        return []

    # Group by date
    dates = [timestamp_to_date(t) for t in times]
    unique_dates = sorted(set(d for d in dates if d is not None))

    for date in unique_dates:
        mask = [d == date for d in dates]
        day_activities = activities[mask]

        if len(day_activities) < 5:
            continue

        try:
            # Activity ratios (0=still, 1=walking, 2=running, 3=unknown)
            still_ratio = np.mean(day_activities == 0)
            walking_ratio = np.mean(day_activities == 1)
            running_ratio = np.mean(day_activities == 2)

            # Transitions
            transitions = np.sum(np.diff(day_activities) != 0)

            daily_features.append({
                "user_id": user_id,
                "date": date,
                "still_ratio": still_ratio,
                "walking_ratio": walking_ratio,
                "running_ratio": running_ratio,
                "activity_transitions": transitions
            })

        except Exception as e:
            logger.debug(f"Activity daily extraction failed for {user_id} on {date}: {e}")
            continue

    return daily_features


def extract_conversation_daily(conv_df: pl.DataFrame, user_id: str) -> List[Dict]:
    """
    Extract daily conversation features.

    Features per day:
    - conversation_count: number of conversations
    - conversation_duration: total hours in conversation
    - conversation_duration_mean: average conversation length (minutes)
    """
    user_conv = conv_df.filter(pl.col("user_id") == user_id)

    if len(user_conv) < 1:
        return []

    # Find start and end timestamp columns (handle CSV column name variations)
    start_col = None
    for col in ["start_timestamp", " start_timestamp"]:
        if col in user_conv.columns:
            start_col = col
            break

    end_col = None
    for col in ["end_timestamp", " end_timestamp"]:
        if col in user_conv.columns:
            end_col = col
            break

    if start_col is None or end_col is None:
        return []

    daily_features = []

    starts = user_conv[start_col].to_numpy()
    ends = user_conv[end_col].to_numpy()

    # Filter valid
    valid = (starts > 0) & (ends > starts) & ~np.isnan(starts) & ~np.isnan(ends)
    starts = starts[valid]
    ends = ends[valid]

    if len(starts) == 0:
        return []

    # Group by date
    dates = [timestamp_to_date(s) for s in starts]
    unique_dates = sorted(set(d for d in dates if d is not None))

    for date in unique_dates:
        mask = [d == date for d in dates]
        day_starts = starts[mask]
        day_ends = ends[mask]

        if len(day_starts) == 0:
            continue

        try:
            durations = day_ends - day_starts

            daily_features.append({
                "user_id": user_id,
                "date": date,
                "conversation_count": len(day_starts),
                "conversation_duration": np.sum(durations) / 3600,  # hours
                "conversation_duration_mean": np.mean(durations) / 60  # minutes
            })

        except Exception as e:
            logger.debug(f"Conversation daily extraction failed for {user_id} on {date}: {e}")
            continue

    return daily_features


# =============================================================================
# Sliding Window Feature Extraction
# =============================================================================

def compute_sliding_window_features(
    daily_df: pl.DataFrame,
    feature_cols: List[str],
    modality: str,
    window_sizes: List[int] = WINDOW_SIZES,
    aggregations: List[str] = AGGREGATIONS
) -> pl.DataFrame:
    """
    Compute sliding window features from daily data.

    For each user and date, compute statistics over the past N days.

    Args:
        daily_df: DataFrame with columns [user_id, date, feature1, feature2, ...]
        feature_cols: List of feature columns to aggregate
        modality: Modality name (gps, phone, activity, conversation)
        window_sizes: List of window sizes in days
        aggregations: List of aggregation functions

    Returns:
        DataFrame with sliding window features
    """
    if daily_df is None or len(daily_df) == 0:
        return pl.DataFrame()

    all_rows = []

    users = daily_df["user_id"].unique().to_list()

    for user_id in users:
        user_df = daily_df.filter(pl.col("user_id") == user_id).sort("date")

        if len(user_df) < 2:
            continue

        dates = user_df["date"].to_list()

        for i, current_date in enumerate(dates):
            row = {"user_id": user_id, "date": current_date}

            for window in window_sizes:
                # Get data from past `window` days (including current)
                window_start = current_date - timedelta(days=window - 1)

                window_df = user_df.filter(
                    (pl.col("date") >= window_start) &
                    (pl.col("date") <= current_date)
                )

                if len(window_df) < max(1, window // 2):  # Require at least half the window
                    continue

                for feat in feature_cols:
                    if feat not in window_df.columns:
                        continue

                    values = window_df[feat].drop_nulls().to_numpy()

                    if len(values) == 0:
                        continue

                    prefix = f"{modality}_{feat}_{window}d"

                    for agg in aggregations:
                        if agg == 'mean':
                            row[f"{prefix}_mean"] = np.mean(values)
                        elif agg == 'std':
                            row[f"{prefix}_std"] = np.std(values) if len(values) > 1 else 0
                        elif agg == 'min':
                            row[f"{prefix}_min"] = np.min(values)
                        elif agg == 'max':
                            row[f"{prefix}_max"] = np.max(values)
                        elif agg == 'trend':
                            row[f"{prefix}_trend"] = compute_trend(values)

            if len(row) > 2:  # Has more than just user_id and date
                all_rows.append(row)

    if not all_rows:
        return pl.DataFrame()

    return pl.DataFrame(all_rows)


# =============================================================================
# User-Level Aggregation (Original Functionality)
# =============================================================================

def aggregate_daily_to_user(daily_df: pl.DataFrame, feature_cols: List[str]) -> pl.DataFrame:
    """Aggregate daily features to user level."""
    if daily_df is None or len(daily_df) == 0:
        return pl.DataFrame()

    agg_exprs = []
    for col in feature_cols:
        if col in daily_df.columns:
            agg_exprs.extend([
                pl.col(col).mean().alias(f"{col}_mean"),
                pl.col(col).std().alias(f"{col}_std"),
            ])

    if not agg_exprs:
        return pl.DataFrame()

    return daily_df.group_by("user_id").agg(agg_exprs)


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Preprocess StudentLife data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/datasets/studentlife.yaml",
        help="Dataset configuration file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/dataset",
        help="Raw data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/processed",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup
    np.random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_dir = output_dir / "daily_features"
    daily_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("StudentLife Data Preprocessing (with Temporal Features)")
    logger.info("=" * 60)

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = OmegaConf.load(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        config = OmegaConf.create({})
        logger.warning(f"Config not found: {config_path}, using defaults")

    # Initialize loader
    data_dir = Path(args.data_dir)
    loader = StudentLifeLoader(data_dir, config)

    # Get dataset info
    info = loader.get_dataset_info()
    logger.info(f"Dataset: {info.name}")
    logger.info(f"Participants: {info.n_participants}")
    logger.info(f"Time range: {info.time_range}")

    # Load data
    logger.info("Loading survey data...")
    surveys = loader.load_surveys()

    logger.info("Loading sensor data...")
    sensors = loader.load_sensors()

    logger.info("Loading education data...")
    education = loader.load_education()

    # Get user list
    user_ids = loader.get_user_ids()
    logger.info(f"Processing {len(user_ids)} users")

    # ==========================================================================
    # Step 1: Extract Daily Features
    # ==========================================================================
    logger.info("\n--- Extracting Daily Features ---")

    # GPS daily
    gps_daily_all = []
    if "gps" in sensors and sensors["gps"] is not None:
        for uid in user_ids:
            gps_daily_all.extend(extract_gps_daily(sensors["gps"], uid))
        logger.info(f"GPS: {len(gps_daily_all)} daily records")

    gps_daily_df = pl.DataFrame(gps_daily_all) if gps_daily_all else None
    if gps_daily_df is not None and len(gps_daily_df) > 0:
        gps_daily_df.write_parquet(daily_dir / "gps_daily.parquet")
        logger.info(f"  Saved to {daily_dir / 'gps_daily.parquet'}")

    # Phone daily
    phone_daily_all = []
    if "phonelock" in sensors and sensors["phonelock"] is not None:
        for uid in user_ids:
            phone_daily_all.extend(extract_phone_daily(sensors["phonelock"], uid))
        logger.info(f"Phone: {len(phone_daily_all)} daily records")

    phone_daily_df = pl.DataFrame(phone_daily_all) if phone_daily_all else None
    if phone_daily_df is not None and len(phone_daily_df) > 0:
        phone_daily_df.write_parquet(daily_dir / "phone_daily.parquet")
        logger.info(f"  Saved to {daily_dir / 'phone_daily.parquet'}")

    # Activity daily
    activity_daily_all = []
    if "activity" in sensors and sensors["activity"] is not None:
        for uid in user_ids:
            activity_daily_all.extend(extract_activity_daily(sensors["activity"], uid))
        logger.info(f"Activity: {len(activity_daily_all)} daily records")

    activity_daily_df = pl.DataFrame(activity_daily_all) if activity_daily_all else None
    if activity_daily_df is not None and len(activity_daily_df) > 0:
        activity_daily_df.write_parquet(daily_dir / "activity_daily.parquet")
        logger.info(f"  Saved to {daily_dir / 'activity_daily.parquet'}")

    # Conversation daily
    conv_daily_all = []
    if "conversation" in sensors and sensors["conversation"] is not None:
        for uid in user_ids:
            conv_daily_all.extend(extract_conversation_daily(sensors["conversation"], uid))
        logger.info(f"Conversation: {len(conv_daily_all)} daily records")

    conv_daily_df = pl.DataFrame(conv_daily_all) if conv_daily_all else None
    if conv_daily_df is not None and len(conv_daily_df) > 0:
        conv_daily_df.write_parquet(daily_dir / "conversation_daily.parquet")
        logger.info(f"  Saved to {daily_dir / 'conversation_daily.parquet'}")

    # ==========================================================================
    # Step 2: Compute Sliding Window Features
    # ==========================================================================
    logger.info("\n--- Computing Sliding Window Features ---")

    temporal_dfs = []

    # GPS sliding window
    if gps_daily_df is not None and len(gps_daily_df) > 0:
        gps_features = ["location_entropy", "home_stay_ratio", "n_locations",
                        "total_distance", "radius_of_gyration"]
        gps_temporal = compute_sliding_window_features(
            gps_daily_df, gps_features, "gps"
        )
        if len(gps_temporal) > 0:
            temporal_dfs.append(gps_temporal)
            logger.info(f"GPS temporal: {len(gps_temporal)} records, {len(gps_temporal.columns)} features")

    # Phone sliding window
    if phone_daily_df is not None and len(phone_daily_df) > 0:
        phone_features = ["unlock_count", "screen_time", "session_duration_mean",
                          "session_duration_std", "night_usage_ratio"]
        phone_temporal = compute_sliding_window_features(
            phone_daily_df, phone_features, "phone"
        )
        if len(phone_temporal) > 0:
            temporal_dfs.append(phone_temporal)
            logger.info(f"Phone temporal: {len(phone_temporal)} records, {len(phone_temporal.columns)} features")

    # Activity sliding window
    if activity_daily_df is not None and len(activity_daily_df) > 0:
        activity_features = ["still_ratio", "walking_ratio", "running_ratio",
                             "activity_transitions"]
        activity_temporal = compute_sliding_window_features(
            activity_daily_df, activity_features, "activity"
        )
        if len(activity_temporal) > 0:
            temporal_dfs.append(activity_temporal)
            logger.info(f"Activity temporal: {len(activity_temporal)} records, {len(activity_temporal.columns)} features")

    # Conversation sliding window
    if conv_daily_df is not None and len(conv_daily_df) > 0:
        conv_features = ["conversation_count", "conversation_duration",
                         "conversation_duration_mean"]
        conv_temporal = compute_sliding_window_features(
            conv_daily_df, conv_features, "conversation"
        )
        if len(conv_temporal) > 0:
            temporal_dfs.append(conv_temporal)
            logger.info(f"Conversation temporal: {len(conv_temporal)} records, {len(conv_temporal.columns)} features")

    # Merge all temporal features
    if temporal_dfs:
        logger.info(f"Merging {len(temporal_dfs)} temporal DataFrames...")
        for i, df in enumerate(temporal_dfs):
            logger.info(f"  DataFrame {i}: {df.shape[0]} rows, {df.shape[1]} cols")

        # Join on user_id and date using coalesce to handle full join properly
        temporal_merged = temporal_dfs[0]
        for i, df in enumerate(temporal_dfs[1:], start=1):
            logger.info(f"  Joining DataFrame {i}...")
            # Use outer join with suffix, then drop duplicates
            temporal_merged = temporal_merged.join(
                df, on=["user_id", "date"], how="full", suffix="_dup"
            )
            # Remove any duplicate columns created by join
            dup_cols = [c for c in temporal_merged.columns if c.endswith("_dup")]
            if dup_cols:
                temporal_merged = temporal_merged.drop(dup_cols)
            logger.info(f"  After join: {temporal_merged.shape}")

        temporal_merged.write_parquet(output_dir / "temporal_features.parquet")
        logger.info(f"Saved temporal features: {temporal_merged.shape}")
        logger.info(f"  Columns: {len(temporal_merged.columns)}")

    # ==========================================================================
    # Step 3: Create User-Level Aggregated Features (for backward compatibility)
    # ==========================================================================
    logger.info("\n--- Creating User-Level Features ---")

    user_features_dfs = []

    # GPS user-level
    if gps_daily_df is not None and len(gps_daily_df) > 0:
        gps_features = ["location_entropy", "home_stay_ratio", "n_locations",
                        "total_distance", "radius_of_gyration"]
        gps_user = aggregate_daily_to_user(gps_daily_df, gps_features)
        if len(gps_user) > 0:
            # Add validity flag
            gps_user = gps_user.with_columns(pl.lit(True).alias("gps_valid"))
            user_features_dfs.append(gps_user)

    # Phone user-level
    if phone_daily_df is not None and len(phone_daily_df) > 0:
        phone_features = ["unlock_count", "screen_time", "session_duration_mean",
                          "session_duration_std", "night_usage_ratio"]
        phone_user = aggregate_daily_to_user(phone_daily_df, phone_features)
        if len(phone_user) > 0:
            phone_user = phone_user.with_columns(pl.lit(True).alias("phone_valid"))
            user_features_dfs.append(phone_user)

    # Activity user-level
    if activity_daily_df is not None and len(activity_daily_df) > 0:
        activity_features = ["still_ratio", "walking_ratio", "running_ratio",
                             "activity_transitions"]
        activity_user = aggregate_daily_to_user(activity_daily_df, activity_features)
        if len(activity_user) > 0:
            activity_user = activity_user.with_columns(pl.lit(True).alias("activity_valid"))
            user_features_dfs.append(activity_user)

    # Conversation user-level
    if conv_daily_df is not None and len(conv_daily_df) > 0:
        conv_features = ["conversation_count", "conversation_duration",
                         "conversation_duration_mean"]
        conv_user = aggregate_daily_to_user(conv_daily_df, conv_features)
        if len(conv_user) > 0:
            conv_user = conv_user.with_columns(pl.lit(True).alias("conv_valid"))
            user_features_dfs.append(conv_user)

    # Merge user-level features
    if user_features_dfs:
        user_merged = user_features_dfs[0]
        for df in user_features_dfs[1:]:
            user_merged = user_merged.join(df, on="user_id", how="full", suffix="_dup")
            # Remove any duplicate columns
            dup_cols = [c for c in user_merged.columns if c.endswith("_dup")]
            if dup_cols:
                user_merged = user_merged.drop(dup_cols)

        # Ensure all users are present
        all_users_df = pl.DataFrame({"user_id": user_ids})
        user_merged = all_users_df.join(user_merged, on="user_id", how="left", suffix="_dup")
        # Remove any duplicate columns
        dup_cols = [c for c in user_merged.columns if c.endswith("_dup")]
        if dup_cols:
            user_merged = user_merged.drop(dup_cols)

        user_merged.write_parquet(output_dir / "sensor_features.parquet")
        logger.info(f"Saved user-level features: {user_merged.shape}")

    # ==========================================================================
    # Step 4: Save Survey and Education Data
    # ==========================================================================
    logger.info("\n--- Saving Survey and Education Data ---")

    if "phq9" in surveys:
        phq9_df = surveys["phq9"]
        survey_scores_path = output_dir / "survey_scores.parquet"
        phq9_df.write_parquet(survey_scores_path)
        logger.info(f"Saved survey scores: {phq9_df.shape}")

    if "grades" in education:
        grades_df = education["grades"]
        grades_path = output_dir / "grades.parquet"
        grades_df.write_parquet(grades_path)
        logger.info(f"Saved grades: {grades_df.shape}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"\nGenerated files:")
    logger.info(f"  - sensor_features.parquet (user-level aggregates)")
    logger.info(f"  - temporal_features.parquet (sliding window features)")
    logger.info(f"  - daily_features/*.parquet (daily features per modality)")
    logger.info(f"  - survey_scores.parquet")
    if (output_dir / "grades.parquet").exists():
        logger.info(f"  - grades.parquet")


if __name__ == "__main__":
    main()
