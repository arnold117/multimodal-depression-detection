#!/usr/bin/env python3
"""
Train Baseline Models (Phase 3)

Trains three baseline machine learning models for suicide risk prediction:
1. Logistic Regression (L2 regularization, balanced class weights)
2. Random Forest (shallow trees, n=500)
3. XGBoost (scale_pos_weight for imbalance)

Performs 5-fold stratified cross-validation and permutation tests.
Saves trained models, metrics, and visualizations.

Usage:
    mamba activate qbio
    python scripts/07_train_baseline.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.data_loader import (
    load_config, load_features_labels,
    apply_feature_scaling, get_stratified_cv_splits,
    check_data_quality
)
from src.models.baseline import BaselineModel
from src.models.evaluation import (
    cross_validate_model, calculate_metrics,
    permutation_test, plot_roc_curve,
    plot_confusion_matrix, plot_precision_recall_curve,
    optimize_threshold_for_sensitivity
)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("BASELINE MODEL TRAINING - PHASE 3")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # -----------------------------
    # 1. Load Configuration
    # -----------------------------
    print("Step 1: Loading configuration...")
    config = load_config('configs/model_configs.yaml')
    print(f"✓ Configuration loaded")

    # Create output directories
    results_dir = Path(config['paths']['results']['models'])
    metrics_dir = Path(config['paths']['results']['metrics'])
    figures_dir = Path(config['paths']['results']['figures'])

    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 2. Load Data
    # -----------------------------
    print("\nStep 2: Loading features and labels...")
    X, y, feature_names = load_features_labels(
        features_path=config['paths']['data']['features'],
        labels_path=config['paths']['data']['labels']
    )

    # Check data quality
    check_data_quality(X, y, feature_names)

    # -----------------------------
    # 3. Prepare Cross-Validation
    # -----------------------------
    print("\nStep 3: Preparing cross-validation...")
    cv = get_stratified_cv_splits(
        X, y,
        n_splits=config['common']['cv_folds'],
        random_state=config['common']['random_seed']
    )

    # -----------------------------
    # 4. Train Models
    # -----------------------------
    print("\n" + "=" * 80)
    print("Step 4: Training Baseline Models")
    print("=" * 80)

    model_types = ['logistic', 'random_forest', 'xgboost']
    model_results = {}

    for model_type in model_types:
        print(f"\n{'=' * 80}")
        print(f"Training: {model_type.upper().replace('_', ' ')}")
        print(f"{'=' * 80}")

        # Create model
        model = BaselineModel(model_type=model_type, config=config)

        # Fit on full dataset (scaler will be fitted inside)
        model.fit(X, y, feature_names=feature_names)

        # Cross-validation evaluation
        print(f"\nEvaluating {model_type} with cross-validation...")
        cv_metrics, cv_predictions = cross_validate_model(
            model.model,  # Use the underlying sklearn model for CV
            model.scaler.transform(X),  # Pre-scaled features
            y,
            cv,
            return_predictions=True
        )

        # Permutation test
        print(f"\nPerforming permutation test for {model_type}...")
        true_auc, p_value, perm_scores = permutation_test(
            model.model,
            model.scaler.transform(X),
            y,
            n_permutations=config['evaluation']['permutation_test']['n_permutations'],
            metric='roc_auc',
            cv=cv,
            random_state=config['common']['random_seed']
        )

        # Optimize threshold for sensitivity
        print(f"\nOptimizing classification threshold for {model_type}...")
        optimal_threshold, opt_metrics = optimize_threshold_for_sensitivity(
            y,
            cv_predictions,
            target_sensitivity=config['evaluation']['target_sensitivity']
        )

        # Store results
        model_results[model_type] = {
            'model': model,
            'cv_metrics': cv_metrics,
            'cv_predictions': cv_predictions,
            'permutation_test': {
                'true_auc': float(true_auc),
                'p_value': float(p_value),
                'permuted_scores_mean': float(perm_scores.mean()),
                'permuted_scores_std': float(perm_scores.std())
            },
            'optimal_threshold': float(optimal_threshold),
            'optimized_metrics': {k: float(v) if isinstance(v, (np.float32, np.float64, np.int64)) else v
                                 for k, v in opt_metrics.items()}
        }

        # Save model
        print(f"\nSaving {model_type} model...")
        model.save(results_dir, model_name=f'{model_type}_baseline')

    # -----------------------------
    # 5. Generate Visualizations
    # -----------------------------
    print("\n" + "=" * 80)
    print("Step 5: Generating Visualizations")
    print("=" * 80)

    # ROC curves (all models on one plot)
    fig, ax = plt.subplots(figsize=(10, 8))
    for model_type in model_types:
        plot_roc_curve(
            y,
            model_results[model_type]['cv_predictions'],
            model_name=model_type.replace('_', ' ').title(),
            ax=ax
        )
    plt.tight_layout()
    roc_path = figures_dir / 'baseline_roc_curves_comparison.png'
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved ROC curves to {roc_path}")
    plt.close()

    # Individual confusion matrices
    for model_type in model_types:
        y_pred_opt = (model_results[model_type]['cv_predictions'] >=
                      model_results[model_type]['optimal_threshold']).astype(int)

        fig = plot_confusion_matrix(
            y,
            y_pred_opt,
            model_name=model_type.replace('_', ' ').title(),
            save_path=figures_dir / f'{model_type}_confusion_matrix.png'
        )
        plt.close()

    # Precision-Recall curves
    fig, ax = plt.subplots(figsize=(10, 8))
    for model_type in model_types:
        precision, recall, _ = precision_recall_curve(
            y, model_results[model_type]['cv_predictions']
        )
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, linewidth=2,
                label=f"{model_type.replace('_', ' ').title()} (AUC={pr_auc:.3f})")

    # Baseline
    baseline = y.sum() / len(y)
    ax.plot([0, 1], [baseline, baseline], 'k--', linewidth=1,
            label=f'Random (AP={baseline:.3f})')

    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves - All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    pr_path = figures_dir / 'baseline_pr_curves_comparison.png'
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PR curves to {pr_path}")
    plt.close()

    # Feature importance comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, model_type in enumerate(model_types):
        importance_df = model_results[model_type]['model'].get_feature_importance()
        top_10 = importance_df.head(10)

        axes[idx].barh(range(len(top_10)), top_10['importance'])
        axes[idx].set_yticks(range(len(top_10)))
        axes[idx].set_yticklabels(top_10['feature'], fontsize=9)
        axes[idx].set_xlabel('Importance', fontsize=10)
        axes[idx].set_title(f"{model_type.replace('_', ' ').title()}",
                           fontsize=12, fontweight='bold')
        axes[idx].invert_yaxis()
        axes[idx].grid(alpha=0.3, axis='x')

    plt.tight_layout()
    importance_path = figures_dir / 'baseline_feature_importance_comparison.png'
    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved feature importance to {importance_path}")
    plt.close()

    # -----------------------------
    # 6. Save Metrics Summary
    # -----------------------------
    print("\n" + "=" * 80)
    print("Step 6: Saving Metrics Summary")
    print("=" * 80)

    # Prepare summary table
    summary_data = []
    for model_type in model_types:
        cv_metrics = model_results[model_type]['cv_metrics']
        opt_metrics = model_results[model_type]['optimized_metrics']
        perm_test = model_results[model_type]['permutation_test']

        summary_data.append({
            'model': model_type,
            'cv_roc_auc_mean': np.mean(cv_metrics['roc_auc']),
            'cv_roc_auc_std': np.std(cv_metrics['roc_auc']),
            'cv_pr_auc_mean': np.mean(cv_metrics['pr_auc']),
            'cv_pr_auc_std': np.std(cv_metrics['pr_auc']),
            'cv_sensitivity_mean': np.mean(cv_metrics['sensitivity']),
            'cv_sensitivity_std': np.std(cv_metrics['sensitivity']),
            'cv_specificity_mean': np.mean(cv_metrics['specificity']),
            'cv_specificity_std': np.std(cv_metrics['specificity']),
            'cv_f1_mean': np.mean(cv_metrics['f1_score']),
            'cv_f1_std': np.std(cv_metrics['f1_score']),
            'permutation_p_value': perm_test['p_value'],
            'statistically_significant': perm_test['p_value'] < 0.05,
            'optimal_threshold': model_results[model_type]['optimal_threshold'],
            'optimized_sensitivity': opt_metrics['sensitivity'],
            'optimized_specificity': opt_metrics['specificity'],
            'optimized_f1': opt_metrics['f1_score']
        })

    summary_df = pd.DataFrame(summary_data)

    # Save as CSV
    summary_csv = metrics_dir / 'baseline_models_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Saved summary CSV to {summary_csv}")

    # Save detailed JSON
    detailed_results = {}
    for model_type in model_types:
        detailed_results[model_type] = {
            'cv_metrics': {k: [float(v) for v in vals]
                          for k, vals in model_results[model_type]['cv_metrics'].items()},
            'permutation_test': model_results[model_type]['permutation_test'],
            'optimal_threshold': model_results[model_type]['optimal_threshold'],
            'optimized_metrics': model_results[model_type]['optimized_metrics']
        }

    detailed_json = metrics_dir / 'baseline_models_detailed.json'
    with open(detailed_json, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"✓ Saved detailed JSON to {detailed_json}")

    # -----------------------------
    # 7. Print Final Summary
    # -----------------------------
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    print("\nCross-Validation Performance (mean ± std):")
    print("-" * 80)
    print(f"{'Model':<20} {'AUC-ROC':<15} {'Sensitivity':<15} {'Specificity':<15} {'F1':<15}")
    print("-" * 80)

    for model_type in model_types:
        cv_metrics = model_results[model_type]['cv_metrics']
        print(f"{model_type:<20} "
              f"{np.mean(cv_metrics['roc_auc']):.3f}±{np.std(cv_metrics['roc_auc']):.3f}    "
              f"{np.mean(cv_metrics['sensitivity']):.3f}±{np.std(cv_metrics['sensitivity']):.3f}    "
              f"{np.mean(cv_metrics['specificity']):.3f}±{np.std(cv_metrics['specificity']):.3f}    "
              f"{np.mean(cv_metrics['f1_score']):.3f}±{np.std(cv_metrics['f1_score']):.3f}")

    print("\n" + "-" * 80)
    print("Permutation Test (Statistical Significance):")
    print("-" * 80)
    print(f"{'Model':<20} {'True AUC':<15} {'p-value':<15} {'Significant':<15}")
    print("-" * 80)

    for model_type in model_types:
        perm = model_results[model_type]['permutation_test']
        sig = "✓ Yes" if perm['p_value'] < 0.05 else "✗ No"
        print(f"{model_type:<20} {perm['true_auc']:.3f}          "
              f"{perm['p_value']:.4f}          {sig}")

    print("\n" + "-" * 80)
    print("Optimized Thresholds (Target Sensitivity ≥ 0.80):")
    print("-" * 80)
    print(f"{'Model':<20} {'Threshold':<15} {'Sensitivity':<15} {'Specificity':<15}")
    print("-" * 80)

    for model_type in model_types:
        thresh = model_results[model_type]['optimal_threshold']
        opt = model_results[model_type]['optimized_metrics']
        print(f"{model_type:<20} {thresh:.3f}          "
              f"{opt['sensitivity']:.3f}          {opt['specificity']:.3f}")

    print("\n" + "=" * 80)
    print(f"✓ BASELINE TRAINING COMPLETE")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print(f"\nOutputs saved to:")
    print(f"  Models: {results_dir}")
    print(f"  Metrics: {metrics_dir}")
    print(f"  Figures: {figures_dir}")


if __name__ == '__main__':
    # Import additional dependencies
    from sklearn.metrics import precision_recall_curve, auc

    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
