"""
GPS Mobility Feature Extractor

Extracts digital biomarkers from GPS data following Saeb et al. (2015):
- Location variance (strongest depression predictor)
- Location diversity (number of significant locations)
- Distance traveled
- Home stay time
- Movement entropy
- Regularity metrics

Reference:
Saeb, S., et al. (2015). Mobile phone sensor correlates of depressive
symptom severity in daily-life behavior. Journal of Medical Internet Research.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
from sklearn.cluster import DBSCAN
from scipy.stats import entropy
from datetime import datetime


class GPSMobilityExtractor:
    """Extract mobility features from GPS data."""

    def __init__(self,
                 data_dir: Path,
                 timelines_file: Path,
                 accuracy_threshold: float = 100.0,
                 min_points_per_day: int = 5):
        """
        Initialize GPS feature extractor.

        Args:
            data_dir: Directory containing GPS files (gps_u*.csv)
            timelines_file: Path to user_timelines.json
            accuracy_threshold: Maximum acceptable GPS accuracy (meters)
            min_points_per_day: Minimum GPS points required per valid day
        """
        self.data_dir = Path(data_dir)
        self.accuracy_threshold = accuracy_threshold
        self.min_points_per_day = min_points_per_day

        # Load user timelines
        with open(timelines_file, 'r') as f:
            self.timelines = json.load(f)

    def extract_features(self, user_id: str) -> Optional[Dict[str, float]]:
        """
        Extract all GPS mobility features for a user.

        Args:
            user_id: User identifier (e.g., 'u00')

        Returns:
            Dictionary of features or None if insufficient data
        """
        # Load and preprocess GPS data
        gps_data = self._load_gps_data(user_id)

        if gps_data is None or len(gps_data) == 0:
            return None

        # Filter by timeline (use data before pre-assessment)
        timeline = self.timelines.get(user_id)
        if timeline is None:
            return None

        data_end = timeline['data_end']
        gps_data = gps_data[gps_data['timestamp'] <= data_end]

        if len(gps_data) < self.min_points_per_day * 7:  # At least 7 valid days
            return None

        # Aggregate to daily level
        daily_features = self._compute_daily_features(gps_data)

        if daily_features is None or len(daily_features) < 7:
            return None

        # Compute user-level statistics
        features = {}

        # 1. Location variance (Saeb et al. - strongest predictor)
        features['location_variance_mean'] = daily_features['location_variance'].mean()
        features['location_variance_std'] = daily_features['location_variance'].std()

        # 2. Distance traveled
        features['distance_traveled_mean'] = daily_features['distance_km'].mean()
        features['distance_traveled_std'] = daily_features['distance_km'].std()

        # 3. Movement metrics
        features['radius_of_gyration_mean'] = daily_features['radius_gyration'].mean()
        features['max_distance_from_home'] = daily_features['max_dist_home'].max()

        # 4. Home stay time (social withdrawal proxy)
        features['home_stay_ratio'] = daily_features['home_stay_hours'].sum() / (len(daily_features) * 24)

        # 5. Location diversity
        n_locations = self._compute_location_diversity(gps_data)
        features['n_significant_locations'] = n_locations

        # 6. Movement entropy
        features['movement_entropy'] = self._compute_movement_entropy(gps_data)

        # 7. Regularity (coefficient of variation on raw variance, not log-transformed)
        # CV measures day-to-day consistency in movement patterns
        features['distance_traveled_cv'] = features['distance_traveled_std'] / (features['distance_traveled_mean'] + 1e-10)
        features['radius_gyration_cv'] = daily_features['radius_gyration'].std() / (daily_features['radius_gyration'].mean() + 1e-10)

        # 8. Data coverage metrics
        features['gps_valid_days'] = len(daily_features)
        features['gps_total_points'] = len(gps_data)
        features['gps_points_per_day'] = len(gps_data) / len(daily_features)

        return features

    def _load_gps_data(self, user_id: str) -> Optional[pd.DataFrame]:
        """Load and clean GPS data for a user."""
        gps_file = self.data_dir / f'gps_{user_id}.csv'

        if not gps_file.exists():
            return None

        try:
            # Read GPS file - handle trailing commas by reading first 4 columns by index
            # The CSV has trailing commas that shift column positions
            df = pd.read_csv(gps_file, header=None, skiprows=1,
                           usecols=[0, 3, 4, 5],
                           names=['timestamp', 'accuracy', 'latitude', 'longitude'])

            # Convert timestamp to numeric (handle any non-numeric values)
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

            # Drop rows with invalid timestamps
            df = df.dropna(subset=['timestamp'])

            # Filter by accuracy
            df = df[df['accuracy'] <= self.accuracy_threshold].copy()

            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df['date'] = df['datetime'].dt.date

            # Remove rows with missing coordinates
            df = df.dropna(subset=['latitude', 'longitude'])

            # Filter days with sufficient points
            daily_counts = df.groupby('date').size()
            valid_dates = daily_counts[daily_counts >= self.min_points_per_day].index
            df = df[df['date'].isin(valid_dates)]

            return df

        except Exception as e:
            print(f"  Warning: Could not process {user_id}: {e}")
            return None

    def _compute_daily_features(self, gps_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute daily-level GPS features."""
        daily_list = []

        # Identify home location (most frequent location in first week)
        home_coords = self._identify_home(gps_data)

        for date, day_data in gps_data.groupby('date'):
            if len(day_data) < self.min_points_per_day:
                continue

            # Location variance
            lat_var = day_data['latitude'].var()
            lon_var = day_data['longitude'].var()
            loc_variance = np.log(lat_var + lon_var + 1e-10)  # Log transform as in Saeb et al.

            # Distance traveled
            distance_km = self._compute_total_distance(day_data)

            # Radius of gyration (spatial spread)
            radius_gyration = self._compute_radius_of_gyration(day_data)

            # Home stay time
            if home_coords is not None:
                home_stay_hours = self._compute_home_stay_time(day_data, home_coords)
            else:
                home_stay_hours = 0.0

            # Max distance from home
            if home_coords is not None:
                max_dist_home = self._compute_max_distance_from_home(day_data, home_coords)
            else:
                max_dist_home = 0.0

            daily_list.append({
                'date': date,
                'location_variance': loc_variance,
                'distance_km': distance_km,
                'radius_gyration': radius_gyration,
                'home_stay_hours': home_stay_hours,
                'max_dist_home': max_dist_home
            })

        if len(daily_list) == 0:
            return None

        return pd.DataFrame(daily_list)

    def _identify_home(self, gps_data: pd.DataFrame) -> Optional[Tuple[float, float]]:
        """Identify home location (most frequent cluster in first 7 days)."""
        # Use first week of data
        first_week = gps_data.nsmallest(min(len(gps_data), 2000), 'timestamp')

        if len(first_week) < 20:
            return None

        # Cluster locations
        coords = first_week[['latitude', 'longitude']].values
        clustering = DBSCAN(eps=0.001, min_samples=5).fit(coords)

        # Find largest cluster
        labels = clustering.labels_
        if len(labels[labels != -1]) == 0:
            # No clusters, use median
            return (first_week['latitude'].median(), first_week['longitude'].median())

        unique, counts = np.unique(labels[labels != -1], return_counts=True)
        largest_cluster = unique[np.argmax(counts)]

        # Home = centroid of largest cluster
        home_points = first_week[labels == largest_cluster]
        home_lat = home_points['latitude'].median()
        home_lon = home_points['longitude'].median()

        return (home_lat, home_lon)

    def _compute_total_distance(self, day_data: pd.DataFrame) -> float:
        """Compute total distance traveled in km using Haversine formula."""
        coords = day_data[['latitude', 'longitude']].values

        if len(coords) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(len(coords) - 1):
            dist = self._haversine_distance(coords[i], coords[i+1])
            # Filter unrealistic movements (>50 km between consecutive points)
            if dist < 50.0:
                total_distance += dist

        return total_distance

    def _haversine_distance(self, coord1: np.ndarray, coord2: np.ndarray) -> float:
        """Calculate distance between two GPS coordinates in km."""
        lat1, lon1 = np.radians(coord1)
        lat2, lon2 = np.radians(coord2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        # Earth radius in km
        r = 6371.0

        return c * r

    def _compute_radius_of_gyration(self, day_data: pd.DataFrame) -> float:
        """Compute radius of gyration (spatial spread)."""
        # Centroid
        center_lat = day_data['latitude'].mean()
        center_lon = day_data['longitude'].mean()

        # Mean distance from centroid
        distances = []
        for _, row in day_data.iterrows():
            dist = self._haversine_distance(
                np.array([center_lat, center_lon]),
                np.array([row['latitude'], row['longitude']])
            )
            distances.append(dist)

        return np.mean(distances)

    def _compute_home_stay_time(self, day_data: pd.DataFrame,
                                 home_coords: Tuple[float, float]) -> float:
        """Compute hours spent at home (within 100m)."""
        home_threshold_km = 0.1  # 100 meters

        at_home = 0
        for _, row in day_data.iterrows():
            dist = self._haversine_distance(
                np.array(home_coords),
                np.array([row['latitude'], row['longitude']])
            )
            if dist <= home_threshold_km:
                at_home += 1

        # Convert to hours (assuming ~1 point per 20 minutes average)
        hours = (at_home / len(day_data)) * 24

        return hours

    def _compute_max_distance_from_home(self, day_data: pd.DataFrame,
                                        home_coords: Tuple[float, float]) -> float:
        """Compute maximum distance from home in km."""
        max_dist = 0.0
        for _, row in day_data.iterrows():
            dist = self._haversine_distance(
                np.array(home_coords),
                np.array([row['latitude'], row['longitude']])
            )
            max_dist = max(max_dist, dist)

        return max_dist

    def _compute_location_diversity(self, gps_data: pd.DataFrame) -> int:
        """Compute number of significant locations using DBSCAN clustering."""
        coords = gps_data[['latitude', 'longitude']].values

        # Sample if too many points (for performance)
        if len(coords) > 5000:
            sample_indices = np.random.choice(len(coords), 5000, replace=False)
            coords = coords[sample_indices]

        # Cluster with DBSCAN (eps=0.001 ≈ 100m)
        clustering = DBSCAN(eps=0.001, min_samples=5).fit(coords)

        # Count clusters (excluding noise points)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        return n_clusters

    def _compute_movement_entropy(self, gps_data: pd.DataFrame) -> float:
        """Compute entropy of hourly movement patterns."""
        # Extract hour of day
        gps_data = gps_data.copy()
        gps_data['hour'] = gps_data['datetime'].dt.hour

        # Count points per hour
        hour_counts = gps_data.groupby('hour').size()

        # Normalize to probability distribution
        hour_probs = hour_counts / hour_counts.sum()

        # Compute entropy
        return entropy(hour_probs)

    def extract_all_users(self, user_ids: list) -> pd.DataFrame:
        """
        Extract GPS features for all users.

        Args:
            user_ids: List of user IDs to process

        Returns:
            DataFrame with features (rows=users, columns=features)
        """
        results = []

        for user_id in user_ids:
            print(f"Processing {user_id}...", end=' ')
            features = self.extract_features(user_id)

            if features is not None:
                features['uid'] = user_id
                results.append(features)
                print(f"✓ ({features['gps_valid_days']:.0f} days)")
            else:
                print("✗ (insufficient data)")

        if len(results) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Reorder columns (uid first)
        cols = ['uid'] + [c for c in df.columns if c != 'uid']
        df = df[cols]

        return df
