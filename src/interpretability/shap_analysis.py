"""
SHAP Analysis for Model Interpretability

Uses SHAP (SHapley Additive exPlanations) to:
1. Identify most important features globally
2. Understand feature interactions
3. Explain individual predictions
4. Generate feature importance rankings
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Optional, List, Tuple
from pathlib import Path

sns.set_style('whitegrid')


class SHAPAnalyzer:
    """SHAP-based interpretability analysis."""

    def __init__(self, model, X_train: np.ndarray, feature_names: List[str]):
        """
        Initialize SHAP analyzer.

        Args:
            model: Trained model (scikit-learn compatible)
            X_train: Training data for background
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names

        # Create explainer
        print("Creating SHAP explainer...")
        try:
            # Try TreeExplainer for tree-based models
            self.explainer = shap.TreeExplainer(model)
            self.explainer_type = 'tree'
        except:
            # Fall back to KernelExplainer
            self.explainer = shap.KernelExplainer(
                model.predict_proba,
                shap.sample(X_train, 100)
            )
            self.explainer_type = 'kernel'

        print(f"✓ Created {self.explainer_type} explainer")

        self.shap_values = None

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for dataset.

        Args:
            X: Feature matrix

        Returns:
            SHAP values
        """
        print("Computing SHAP values...")

        if self.explainer_type == 'tree':
            shap_values = self.explainer.shap_values(X)
            # For binary classification, take positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        self.shap_values = shap_values
        print(f"✓ Computed SHAP values: {shap_values.shape}")

        return shap_values

    def plot_summary(
        self,
        X: np.ndarray,
        save_path: Optional[str] = None,
        max_display: int = 20
    ):
        """
        Plot SHAP summary (beeswarm plot).

        Args:
            X: Feature matrix
            save_path: Path to save figure
            max_display: Maximum features to display
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        fig = plt.figure(figsize=(12, 8))

        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved SHAP summary to {save_path}")

        return fig

    def plot_bar(
        self,
        X: np.ndarray,
        save_path: Optional[str] = None,
        max_display: int = 20
    ):
        """
        Plot SHAP feature importance bar chart.

        Args:
            X: Feature matrix
            save_path: Path to save figure
            max_display: Maximum features to display
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        fig = plt.figure(figsize=(10, 8))

        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_names,
            plot_type='bar',
            max_display=max_display,
            show=False
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved SHAP bar plot to {save_path}")

        return fig

    def get_feature_importance(
        self,
        X: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Get feature importance ranking.

        Args:
            X: Feature matrix (optional, uses cached SHAP values if available)

        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            if X is None:
                raise ValueError("Must provide X if SHAP values not computed")
            self.compute_shap_values(X)

        # Calculate mean absolute SHAP value per feature
        importance = np.abs(self.shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return importance_df

    def plot_dependence(
        self,
        feature: str,
        X: np.ndarray,
        interaction_feature: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot SHAP dependence plot for a feature.

        Args:
            feature: Feature name
            X: Feature matrix
            interaction_feature: Feature to color by (auto if None)
            save_path: Path to save figure
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        fig = plt.figure(figsize=(10, 6))

        feature_idx = self.feature_names.index(feature)
        interaction_idx = (
            self.feature_names.index(interaction_feature)
            if interaction_feature else 'auto'
        )

        shap.dependence_plot(
            feature_idx,
            self.shap_values,
            X,
            feature_names=self.feature_names,
            interaction_index=interaction_idx,
            show=False
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved dependence plot to {save_path}")

        return fig

    def plot_waterfall(
        self,
        sample_idx: int,
        X: np.ndarray,
        save_path: Optional[str] = None,
        max_display: int = 10
    ):
        """
        Plot SHAP waterfall for individual prediction.

        Args:
            sample_idx: Sample index
            X: Feature matrix
            save_path: Path to save figure
            max_display: Maximum features to display
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        fig = plt.figure(figsize=(10, 8))

        # Create explanation object
        if self.explainer_type == 'tree':
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1]
        else:
            expected_value = self.explainer.expected_value

        shap_explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=expected_value,
            data=X[sample_idx],
            feature_names=self.feature_names
        )

        shap.plots.waterfall(shap_explanation, max_display=max_display, show=False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved waterfall plot to {save_path}")

        return fig

    def explain_prediction(
        self,
        sample_idx: int,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Explain individual prediction with top contributing features.

        Args:
            sample_idx: Sample index
            X: Feature matrix
            y: Labels (optional)
            top_k: Number of top features

        Returns:
            DataFrame with feature contributions
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        # Get SHAP values for this sample
        sample_shap = self.shap_values[sample_idx]
        sample_features = X[sample_idx]

        # Create explanation DataFrame
        explanation_df = pd.DataFrame({
            'feature': self.feature_names,
            'value': sample_features,
            'shap_value': sample_shap,
            'abs_shap': np.abs(sample_shap)
        }).sort_values('abs_shap', ascending=False)

        # Add direction
        explanation_df['direction'] = explanation_df['shap_value'].apply(
            lambda x: 'Increases risk' if x > 0 else 'Decreases risk'
        )

        if y is not None:
            print(f"\nSample {sample_idx}:")
            print(f"  True label: {y[sample_idx]}")
            print(f"  Prediction: {self.model.predict([X[sample_idx]])[0]}")

        print(f"\nTop {top_k} contributing features:")
        print(explanation_df[['feature', 'value', 'shap_value', 'direction']].head(top_k).to_string(index=False))

        return explanation_df.head(top_k)


# Example usage
if __name__ == '__main__':
    import sys
    import joblib
    sys.path.append('/Users/arnold/Documents/Dev/multimodal-depression-detection')
    from src.utils.data_loader import load_features_labels

    print("=== Testing SHAP Analysis ===\n")

    # Load data
    X, y, feature_names = load_features_labels()
    print(f"Data: X={X.shape}, y={y.shape}")

    # Load XGBoost model
    model_path = 'results/models/xgboost_baseline.pkl'
    model_dict = joblib.load(model_path)
    model = model_dict['model']

    print(f"Model loaded: {type(model)}")

    # Create analyzer
    analyzer = SHAPAnalyzer(model, X, feature_names)

    # Compute SHAP values
    shap_values = analyzer.compute_shap_values(X)

    # Get feature importance
    print("\nFeature Importance:")
    importance_df = analyzer.get_feature_importance()
    print(importance_df.head(10).to_string(index=False))

    # Explain a positive sample
    if y.sum() > 0:
        pos_idx = np.where(y == 1)[0][0]
        print(f"\n{'='*80}")
        print("Explaining Positive Sample")
        print(f"{'='*80}")
        analyzer.explain_prediction(pos_idx, X, y, top_k=10)

    print("\n✓ SHAP analysis test complete!")
