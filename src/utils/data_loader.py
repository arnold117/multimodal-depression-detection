"""
Data Loading Utilities

Functions for loading and preprocessing features and labels for modeling.
Handles loading from parquet files, train/test splitting, and feature scaling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import yaml


def load_config(config_path: str = 'configs/model_configs.yaml') -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config YAML file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_features_labels(
    features_path: str = 'data/processed/features/combined_features.parquet',
    labels_path: str = 'data/processed/labels/item9_labels_pre.csv',
    return_feature_names: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    """
    Load features and labels from files.

    Args:
        features_path: Path to combined features parquet file
        labels_path: Path to labels CSV file
        return_feature_names: Whether to return feature names

    Returns:
        Tuple of (X, y, feature_names) where:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (n_samples,)
            feature_names: List of feature names (optional)
    """
    # Load features
    features_df = pd.read_parquet(features_path)

    # Load labels
    labels_df = pd.read_csv(labels_path)

    # Ensure user_id alignment
    if 'user_id' in features_df.columns:
        features_df = features_df.set_index('user_id')
    if 'user_id' in labels_df.columns:
        labels_df = labels_df.set_index('user_id')

    # Align indices
    common_users = features_df.index.intersection(labels_df.index)
    features_df = features_df.loc[common_users]
    labels_df = labels_df.loc[common_users]

    # Select only numeric columns
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df = features_df[numeric_cols]

    # Extract numpy arrays
    X = features_df.values.astype(np.float64)
    y = labels_df['item9_binary'].values

    # Handle missing values with median imputation
    if np.isnan(X).any():
        n_missing = np.isnan(X).sum()
        print(f"⚠️  Found {n_missing} missing values, imputing with median...")
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        print(f"✓ Missing values imputed")

    # Get feature names
    feature_names = features_df.columns.tolist() if return_feature_names else None

    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)} (0: negative, 1: positive)")
    print(f"Class imbalance ratio: {np.bincount(y)[0] / np.bincount(y)[1]:.2f}:1")

    return X, y, feature_names


def apply_feature_scaling(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    method: str = 'standard'
) -> Tuple[np.ndarray, Optional[np.ndarray], object]:
    """
    Apply feature scaling to training (and optionally test) data.

    Args:
        X_train: Training features
        X_test: Test features (optional)
        method: Scaling method ('standard', 'minmax', 'robust')

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler)
    """
    # Select scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform test data if provided
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def train_test_split_stratified(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train/test split maintaining class distribution.

    Args:
        X: Feature matrix
        y: Labels
        test_size: Proportion of test set
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"  Class distribution: {np.bincount(y_train)}")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"  Class distribution: {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test


def get_stratified_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
) -> StratifiedKFold:
    """
    Get stratified k-fold cross-validation splits.

    Args:
        X: Feature matrix
        y: Labels
        n_splits: Number of folds
        random_state: Random seed

    Returns:
        StratifiedKFold object
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    print(f"\nCreated {n_splits}-fold stratified CV")
    print(f"Total samples: {X.shape[0]}")
    print(f"Approximate samples per fold: {X.shape[0] // n_splits}")

    return skf


def get_modality_features(
    X: np.ndarray,
    feature_names: List[str],
    config: Optional[Dict] = None
) -> Dict[str, np.ndarray]:
    """
    Split features by modality (GPS, App, Communication, Activity).

    Args:
        X: Feature matrix (n_samples, 44)
        feature_names: List of feature names
        config: Configuration dict (optional, loads from file if None)

    Returns:
        Dictionary with modality names as keys and feature matrices as values
    """
    if config is None:
        config = load_config()

    feature_indices = config['features']

    modalities = {
        'gps': X[:, feature_indices['gps_indices']],
        'app': X[:, feature_indices['app_indices']],
        'communication': X[:, feature_indices['communication_indices']],
        'activity': X[:, feature_indices['activity_indices']]
    }

    print("\nModality feature counts:")
    for name, features in modalities.items():
        print(f"  {name}: {features.shape[1]} features")

    return modalities


def check_data_quality(X: np.ndarray, y: np.ndarray, feature_names: List[str]):
    """
    Check data quality: missing values, infinite values, feature variance.

    Args:
        X: Feature matrix
        y: Labels
        feature_names: List of feature names
    """
    print("\n=== Data Quality Check ===")

    # Check for missing values
    n_missing = np.isnan(X).sum()
    if n_missing > 0:
        print(f"⚠️  Found {n_missing} missing values")
        missing_by_feature = np.isnan(X).sum(axis=0)
        for i, count in enumerate(missing_by_feature):
            if count > 0:
                print(f"  {feature_names[i]}: {count} missing")
    else:
        print("✓ No missing values")

    # Check for infinite values
    n_inf = np.isinf(X).sum()
    if n_inf > 0:
        print(f"⚠️  Found {n_inf} infinite values")
    else:
        print("✓ No infinite values")

    # Check feature variance
    feature_var = X.var(axis=0)
    zero_var_features = [feature_names[i] for i, var in enumerate(feature_var) if var == 0]
    if zero_var_features:
        print(f"⚠️  Found {len(zero_var_features)} features with zero variance:")
        for feat in zero_var_features:
            print(f"  {feat}")
    else:
        print("✓ All features have non-zero variance")

    # Check label distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nLabel distribution:")
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples ({count/len(y)*100:.1f}%)")

    # Feature statistics
    print(f"\nFeature statistics:")
    print(f"  Mean range: [{X.mean(axis=0).min():.3f}, {X.mean(axis=0).max():.3f}]")
    print(f"  Std range: [{X.std(axis=0).min():.3f}, {X.std(axis=0).max():.3f}]")
    print(f"  Min value: {X.min():.3f}")
    print(f"  Max value: {X.max():.3f}")


# Example usage
if __name__ == '__main__':
    import sys
    sys.path.append('/Users/arnold/Documents/Dev/multimodal-depression-detection')

    # Load configuration
    config = load_config()

    # Load data
    X, y, feature_names = load_features_labels(
        features_path=config['paths']['data']['features'],
        labels_path=config['paths']['data']['labels']
    )

    # Check data quality
    check_data_quality(X, y, feature_names)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X, y,
        test_size=config['common']['test_size'],
        random_state=config['common']['random_seed']
    )

    # Apply scaling
    X_train_scaled, X_test_scaled, scaler = apply_feature_scaling(
        X_train, X_test,
        method=config['features']['scaling_method']
    )

    print(f"\nScaled features:")
    print(f"  Train: {X_train_scaled.shape}")
    print(f"  Test: {X_test_scaled.shape}")

    # Get CV splits
    cv = get_stratified_cv_splits(
        X, y,
        n_splits=config['common']['cv_folds'],
        random_state=config['common']['random_seed']
    )

    # Split by modality
    modalities = get_modality_features(X, feature_names, config)

    print("\n✓ Data loading and preprocessing complete!")
