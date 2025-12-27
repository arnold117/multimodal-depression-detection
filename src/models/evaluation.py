"""
Model Evaluation Utilities

Functions for evaluating classification models:
- Cross-validation
- Performance metrics (AUC-ROC, PR-AUC, sensitivity, specificity, F1)
- Permutation tests for statistical significance
- ROC curves and confusion matrices
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, classification_report,
    roc_curve, average_precision_score,
    balanced_accuracy_score, f1_score
)
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        y_pred_proba: Predicted probabilities for positive class (optional)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Handle edge cases (when only one class is present)
    if cm.shape == (1, 1):
        if y_true[0] == 0:  # Only negative samples
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:  # Only positive samples
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        tn, fp, fn, tp = cm.ravel()

    # Basic metrics
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    # Sensitivity (Recall for positive class) - Clinical priority
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['recall'] = metrics['sensitivity']  # Alias

    # Specificity (Recall for negative class)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Precision
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # F1 score
    metrics['f1_score'] = f1_score(y_true, y_pred)

    # Confusion matrix elements
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)

    # Probability-based metrics (if probabilities provided)
    if y_pred_proba is not None:
        # ROC-AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['roc_auc'] = np.nan

        # PR-AUC (better for imbalanced data)
        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
        except:
            metrics['pr_auc'] = np.nan

    return metrics


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    return_predictions: bool = False
) -> Tuple[Dict[str, List[float]], Optional[np.ndarray]]:
    """
    Perform stratified cross-validation and calculate metrics.

    Args:
        model: Sklearn-compatible model
        X: Feature matrix
        y: Labels
        cv: Cross-validation splitter
        return_predictions: Whether to return out-of-fold predictions

    Returns:
        Tuple of (metrics_dict, predictions)
            metrics_dict: Dictionary with lists of per-fold metrics
            predictions: Out-of-fold predictions (if return_predictions=True)
    """
    metrics_per_fold = {
        'roc_auc': [],
        'pr_auc': [],
        'sensitivity': [],
        'specificity': [],
        'f1_score': [],
        'balanced_accuracy': []
    }

    all_predictions = np.zeros(len(y))
    all_predictions_proba = np.zeros(len(y))

    print(f"\nPerforming {cv.n_splits}-fold cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Clone and train model
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        # Predictions
        y_pred = fold_model.predict(X_val)
        y_pred_proba = fold_model.predict_proba(X_val)[:, 1]

        # Store predictions
        all_predictions[val_idx] = y_pred
        all_predictions_proba[val_idx] = y_pred_proba

        # Calculate metrics
        fold_metrics = calculate_metrics(y_val, y_pred, y_pred_proba)

        # Store metrics
        for key in metrics_per_fold.keys():
            if key in fold_metrics:
                metrics_per_fold[key].append(fold_metrics[key])

        # Print fold results
        print(f"  Fold {fold}: "
              f"AUC={fold_metrics.get('roc_auc', np.nan):.3f}, "
              f"Sensitivity={fold_metrics['sensitivity']:.3f}, "
              f"F1={fold_metrics['f1_score']:.3f}")

    # Print mean ± std across folds
    print(f"\nCross-validation results (mean ± std):")
    for metric_name, values in metrics_per_fold.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {metric_name}: {mean_val:.3f} ± {std_val:.3f}")

    predictions = all_predictions_proba if return_predictions else None
    return metrics_per_fold, predictions


def permutation_test(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 1000,
    metric: str = 'roc_auc',
    cv: Optional[StratifiedKFold] = None,
    random_state: int = 42
) -> Tuple[float, float, np.ndarray]:
    """
    Permutation test to assess statistical significance of model performance.

    Args:
        model: Sklearn-compatible model
        X: Feature matrix
        y: Labels
        n_permutations: Number of random permutations
        metric: Metric to test ('roc_auc', 'balanced_accuracy', 'f1')
        cv: Cross-validation splitter (if None, uses 5-fold)
        random_state: Random seed

    Returns:
        Tuple of (true_score, p_value, permuted_scores)
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # True score with real labels
    print(f"\nPerforming permutation test ({n_permutations} permutations)...")
    print(f"Metric: {metric}")

    scoring = metric
    if metric == 'sensitivity':
        scoring = 'recall'  # sklearn uses 'recall' for sensitivity

    true_score = cross_val_score(model, X, y, cv=cv, scoring=scoring).mean()
    print(f"True {metric}: {true_score:.3f}")

    # Permutation scores
    rng = np.random.RandomState(random_state)
    permuted_scores = []

    for i in range(n_permutations):
        # Permute labels
        y_permuted = rng.permutation(y)

        # Calculate score with permuted labels
        score = cross_val_score(model, X, y_permuted, cv=cv, scoring=scoring).mean()
        permuted_scores.append(score)

        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{n_permutations} permutations")

    permuted_scores = np.array(permuted_scores)

    # Calculate p-value
    p_value = (np.sum(permuted_scores >= true_score) + 1) / (n_permutations + 1)

    print(f"\nPermutation test results:")
    print(f"  True score: {true_score:.3f}")
    print(f"  Permuted scores: {permuted_scores.mean():.3f} ± {permuted_scores.std():.3f}")
    print(f"  p-value: {p_value:.4f}")

    if p_value < 0.05:
        print(f"  ✓ Statistically significant (p < 0.05)")
    else:
        print(f"  ✗ Not statistically significant (p ≥ 0.05)")

    return true_score, p_value, permuted_scores


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = 'Model',
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class
        model_name: Name of model for legend
        save_path: Path to save figure (optional)
        ax: Matplotlib axes (optional)

    Returns:
        Matplotlib axes object
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot ROC curve
    ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')

    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    # Save if path provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")

    return ax


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = 'Model',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix with annotations.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of model for title
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure object
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                square=True, linewidths=1, linecolor='gray',
                xticklabels=['Negative (0)', 'Positive (1)'],
                yticklabels=['Negative (0)', 'Positive (1)'],
                ax=ax)

    # Add percentages
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / cm.sum() * 100
            ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                   ha='center', va='center', fontsize=10, color='gray')

    # Formatting
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')

    # Save if path provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = 'Model',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Precision-Recall curve (better for imbalanced data than ROC).

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class
        model_name: Name of model for legend
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure object
    """
    # Calculate PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    ap = average_precision_score(y_true, y_pred_proba)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot PR curve
    ax.plot(recall, precision, linewidth=2,
            label=f'{model_name} (AP = {ap:.3f}, AUC = {pr_auc:.3f})')

    # Plot baseline (random classifier for imbalanced data)
    baseline = y_true.sum() / len(y_true)
    ax.plot([0, 1], [baseline, baseline], 'k--', linewidth=1,
            label=f'Random (AP = {baseline:.3f})')

    # Formatting
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    # Save if path provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PR curve to {save_path}")

    return fig


def optimize_threshold_for_sensitivity(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    target_sensitivity: float = 0.80,
    min_specificity: float = 0.30
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal classification threshold to achieve target sensitivity.

    Clinical priority: Don't miss positive cases (high sensitivity).

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class
        target_sensitivity: Target sensitivity (e.g., 0.80 for 80%)
        min_specificity: Minimum acceptable specificity

    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    # Sensitivity = TPR, Specificity = 1 - FPR
    sensitivities = tpr
    specificities = 1 - fpr

    # Find thresholds that meet sensitivity target
    valid_indices = np.where(
        (sensitivities >= target_sensitivity) &
        (specificities >= min_specificity)
    )[0]

    if len(valid_indices) == 0:
        print(f"⚠️  Cannot achieve sensitivity ≥ {target_sensitivity:.2f} "
              f"with specificity ≥ {min_specificity:.2f}")
        print(f"Using threshold for max sensitivity")
        optimal_idx = np.argmax(sensitivities)
    else:
        # Among valid thresholds, choose one with highest specificity
        optimal_idx = valid_indices[np.argmax(specificities[valid_indices])]

    optimal_threshold = thresholds[optimal_idx]

    # Calculate metrics at optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    metrics = calculate_metrics(y_true, y_pred_optimal, y_pred_proba)
    metrics['threshold'] = optimal_threshold

    print(f"\nOptimal threshold: {optimal_threshold:.3f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
    print(f"  Specificity: {metrics['specificity']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  F1: {metrics['f1_score']:.3f}")

    return optimal_threshold, metrics


# Example usage
if __name__ == '__main__':
    import sys
    sys.path.append('/Users/arnold/Documents/Dev/multimodal-depression-detection')
    from src.utils.data_loader import load_features_labels, train_test_split_stratified
    from sklearn.linear_model import LogisticRegression

    print("=== Testing Evaluation Functions ===\n")

    # Load data
    X, y, feature_names = load_features_labels()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)

    # Train simple model
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    print("\n=== Test Set Metrics ===")
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Cross-validation
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_metrics, _ = cross_validate_model(model, X, y, cv)

    # Optimize threshold
    threshold, opt_metrics = optimize_threshold_for_sensitivity(
        y_test, y_pred_proba, target_sensitivity=0.80
    )

    print("\n✓ Evaluation functions test complete!")
