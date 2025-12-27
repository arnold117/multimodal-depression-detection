"""
Baseline Machine Learning Models

Implements traditional ML models for suicide risk prediction:
- Logistic Regression (interpretable linear baseline)
- Random Forest (non-linear ensemble)
- XGBoost (gradient boosting)

All models handle class imbalance (4 positive vs 42 negative samples).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
import json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler


class BaselineModel:
    """
    Wrapper for baseline classification models with standardized interface.
    """

    def __init__(self, model_type: str = 'logistic', config: Optional[Dict] = None):
        """
        Initialize baseline model.

        Args:
            model_type: Type of model ('logistic', 'random_forest', 'xgboost')
            config: Configuration dictionary (optional)
        """
        self.model_type = model_type
        self.config = config or {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

        # Initialize model based on type
        self._init_model()

    def _init_model(self):
        """Initialize the specific model based on model_type."""
        if self.model_type == 'logistic':
            self.model = self._create_logistic_regression()
        elif self.model_type == 'random_forest':
            self.model = self._create_random_forest()
        elif self.model_type == 'xgboost':
            self.model = self._create_xgboost()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _create_logistic_regression(self) -> LogisticRegression:
        """
        Create Logistic Regression model.

        L2 regularization with balanced class weights to handle imbalance.
        """
        params = self.config.get('baseline', {}).get('logistic_regression', {})

        return LogisticRegression(
            C=params.get('C', 1.0),
            penalty=params.get('penalty', 'l2'),
            solver=params.get('solver', 'lbfgs'),
            max_iter=params.get('max_iter', 1000),
            class_weight=params.get('class_weight', 'balanced'),
            random_state=self.config.get('common', {}).get('random_seed', 42)
        )

    def _create_random_forest(self) -> RandomForestClassifier:
        """
        Create Random Forest model.

        Shallow trees (max_depth=3) to prevent overfitting on small dataset.
        """
        params = self.config.get('baseline', {}).get('random_forest', {})

        return RandomForestClassifier(
            n_estimators=params.get('n_estimators', 500),
            max_depth=params.get('max_depth', 3),
            min_samples_split=params.get('min_samples_split', 5),
            min_samples_leaf=params.get('min_samples_leaf', 2),
            class_weight=params.get('class_weight', 'balanced'),
            random_state=params.get('random_state', 42),
            n_jobs=params.get('n_jobs', -1)
        )

    def _create_xgboost(self) -> XGBClassifier:
        """
        Create XGBoost model.

        Uses scale_pos_weight to handle 10.5:1 class imbalance.
        """
        params = self.config.get('baseline', {}).get('xgboost', {})

        return XGBClassifier(
            n_estimators=params.get('n_estimators', 300),
            max_depth=params.get('max_depth', 3),
            learning_rate=params.get('learning_rate', 0.1),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            scale_pos_weight=params.get('scale_pos_weight', 10.5),
            random_state=params.get('random_state', 42),
            n_jobs=params.get('n_jobs', -1),
            eval_metric='logloss',
            use_label_encoder=False
        )

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list] = None):
        """
        Fit the model.

        Args:
            X: Training features
            y: Training labels
            feature_names: List of feature names (optional)
        """
        # Store feature names
        self.feature_names = feature_names

        # Fit scaler and transform features
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        print(f"\nTraining {self.model_type} model...")
        print(f"  Training samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Class distribution: {np.bincount(y)}")

        self.model.fit(X_scaled, y)
        self.is_fitted = True

        print(f"✓ {self.model_type} model trained successfully")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features

        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features

        Returns:
            Predicted probabilities (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Returns:
            DataFrame with features and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")

        # Extract importance based on model type
        if self.model_type == 'logistic':
            # Logistic regression coefficients
            importance = np.abs(self.model.coef_[0])
            importance_type = 'coefficient_magnitude'
        elif self.model_type in ['random_forest', 'xgboost']:
            # Tree-based feature importance
            importance = self.model.feature_importances_
            importance_type = 'gini_importance' if self.model_type == 'random_forest' else 'gain'
        else:
            raise ValueError(f"Feature importance not implemented for {self.model_type}")

        # Create DataFrame
        feature_names = self.feature_names or [f'feature_{i}' for i in range(len(importance))]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
            'importance_type': importance_type
        })

        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df = importance_df.reset_index(drop=True)

        return importance_df

    def save(self, save_dir: str, model_name: Optional[str] = None):
        """
        Save model and scaler to disk.

        Args:
            save_dir: Directory to save model
            model_name: Custom model name (optional)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        if model_name is None:
            model_name = f'{self.model_type}_baseline'

        # Save model
        model_path = save_dir / f'{model_name}.pkl'
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'config': self.config
        }, model_path)

        print(f"✓ Model saved to {model_path}")

        # Save feature importance
        try:
            importance_df = self.get_feature_importance()
            importance_path = save_dir / f'{model_name}_feature_importance.csv'
            importance_df.to_csv(importance_path, index=False)
            print(f"✓ Feature importance saved to {importance_path}")
        except Exception as e:
            print(f"⚠️  Could not save feature importance: {e}")

    @classmethod
    def load(cls, model_path: str) -> 'BaselineModel':
        """
        Load model from disk.

        Args:
            model_path: Path to saved model file

        Returns:
            Loaded BaselineModel instance
        """
        # Load saved data
        saved_data = joblib.load(model_path)

        # Create instance
        instance = cls(
            model_type=saved_data['model_type'],
            config=saved_data.get('config', {})
        )

        # Restore model and scaler
        instance.model = saved_data['model']
        instance.scaler = saved_data['scaler']
        instance.feature_names = saved_data.get('feature_names')
        instance.is_fitted = True

        print(f"✓ Model loaded from {model_path}")

        return instance

    def __repr__(self) -> str:
        """String representation."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        return f"BaselineModel(type='{self.model_type}', {fitted_status})"


def train_all_baseline_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[list] = None,
    config: Optional[Dict] = None,
    save_dir: Optional[str] = None
) -> Dict[str, BaselineModel]:
    """
    Train all three baseline models.

    Args:
        X: Training features
        y: Training labels
        feature_names: List of feature names (optional)
        config: Configuration dictionary (optional)
        save_dir: Directory to save models (optional)

    Returns:
        Dictionary of trained models
    """
    models = {}
    model_types = ['logistic', 'random_forest', 'xgboost']

    print("=" * 60)
    print("Training All Baseline Models")
    print("=" * 60)

    for model_type in model_types:
        print(f"\n{'=' * 60}")
        print(f"Model: {model_type.upper()}")
        print(f"{'=' * 60}")

        # Create and train model
        model = BaselineModel(model_type=model_type, config=config)
        model.fit(X, y, feature_names=feature_names)

        # Save if directory provided
        if save_dir:
            model.save(save_dir)

        # Store model
        models[model_type] = model

        # Print top features
        print(f"\nTop 10 features for {model_type}:")
        importance_df = model.get_feature_importance()
        for i, row in importance_df.head(10).iterrows():
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")

    print("\n" + "=" * 60)
    print("✓ All baseline models trained successfully")
    print("=" * 60)

    return models


# Example usage
if __name__ == '__main__':
    import sys
    sys.path.append('/Users/arnold/Documents/Dev/multimodal-depression-detection')
    from src.utils.data_loader import load_config, load_features_labels

    # Load configuration
    config = load_config()

    # Load data
    X, y, feature_names = load_features_labels()

    # Train all models
    models = train_all_baseline_models(
        X, y,
        feature_names=feature_names,
        config=config,
        save_dir='results/models'
    )

    # Test predictions
    print("\n" + "=" * 60)
    print("Testing Predictions")
    print("=" * 60)

    for model_type, model in models.items():
        print(f"\n{model_type}:")
        y_pred = model.predict(X[:5])
        y_pred_proba = model.predict_proba(X[:5])
        print(f"  Predictions: {y_pred}")
        print(f"  Probabilities (positive class): {y_pred_proba[:, 1]}")
