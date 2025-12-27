"""
Model Comparison Visualization Tools

Creates publication-quality figures comparing all models:
- ROC curves
- Performance bar charts
- Confusion matrices
- Model comparison tables
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from typing import Dict, List, Optional
from pathlib import Path

sns.set_style('whitegrid')


def plot_roc_curves(
    models_results: Dict[str, Dict],
    save_path: Optional[str] = None
):
    """
    Plot ROC curves for all models on same figure.

    Args:
        models_results: Dict mapping model name to results dict with 'y_true' and 'y_proba'
        save_path: Path to save figure

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_results)))

    for (name, results), color in zip(models_results.items(), colors):
        y_true = results['y_true']
        y_proba = results['y_proba']

        # Calculate ROC curve
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)

            ax.plot(
                fpr, tpr,
                color=color,
                linewidth=2.5,
                label=f'{name} (AUC = {roc_auc:.3f})',
                alpha=0.8
            )
        except (ValueError, Exception) as e:
            # Skip if ROC can't be calculated
            print(f"Warning: Could not calculate ROC for {name}: {e}")
            continue

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')

    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curves - Model Comparison', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved ROC curves to {save_path}")

    return fig


def plot_performance_comparison(
    metrics_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot performance metrics comparison as grouped bar chart.

    Args:
        metrics_df: DataFrame with models as rows and metrics as columns
        save_path: Path to save figure

    Returns:
        Figure object
    """
    # Select key metrics for visualization
    metrics_to_plot = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']
    available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]

    if not available_metrics:
        print("Warning: No metrics available for plotting")
        return None

    fig, ax = plt.subplots(figsize=(14, 6))

    # Prepare data
    plot_data = metrics_df[available_metrics].copy()

    # Bar plot
    x = np.arange(len(plot_data))
    width = 0.15
    multiplier = 0

    for metric in available_metrics:
        offset = width * multiplier
        values = plot_data[metric].values

        # Replace NaN with 0 for visualization
        values = np.nan_to_num(values, nan=0.0)

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=metric,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5
        )

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, plot_data[metric].values)):
            height = bar.get_height()
            if not np.isnan(val) and val > 0.05:  # Only show if value > 5%
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{val:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )

        multiplier += 1

    ax.set_xlabel('Model', fontsize=13)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Model Performance Comparison', fontsize=15, fontweight='bold')
    ax.set_xticks(x + width * (len(available_metrics) - 1) / 2)
    ax.set_xticklabels(plot_data.index, rotation=45, ha='right')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim([0, 1.1])
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved performance comparison to {save_path}")

    return fig


def plot_confusion_matrices(
    models_results: Dict[str, Dict],
    save_path: Optional[str] = None
):
    """
    Plot confusion matrices for all models in grid layout.

    Args:
        models_results: Dict mapping model name to results dict with 'y_true' and 'y_pred'
        save_path: Path to save figure

    Returns:
        Figure object
    """
    n_models = len(models_results)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))

    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (name, results) in enumerate(models_results.items()):
        y_true = results['y_true']
        y_pred = results['y_pred']

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot
        sns.heatmap(
            cm_norm,
            annot=np.array([[f'{val}\n({norm:.1%})' for val, norm in zip(row_cm, row_norm)]
                           for row_cm, row_norm in zip(cm, cm_norm)]),
            fmt='',
            cmap='Blues',
            cbar=True,
            ax=axes[idx],
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            vmin=0,
            vmax=1
        )

        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrices to {save_path}")

    return fig


def create_model_comparison_table(
    metrics_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create formatted comparison table.

    Args:
        metrics_df: DataFrame with model metrics
        save_path: Path to save CSV

    Returns:
        Formatted DataFrame
    """
    # Format percentages
    formatted_df = metrics_df.copy()

    percentage_cols = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'AUC-ROC', 'PR-AUC']
    for col in percentage_cols:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f'{x:.1%}' if not np.isnan(x) else 'N/A'
            )

    # Format parameter count
    if 'Parameters' in formatted_df.columns:
        formatted_df['Parameters'] = formatted_df['Parameters'].apply(
            lambda x: f'{int(x):,}' if not np.isnan(x) else 'N/A'
        )

    if save_path:
        formatted_df.to_csv(save_path)
        print(f"✓ Saved comparison table to {save_path}")

    return formatted_df


def plot_model_ranking(
    metrics_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot model ranking across different metrics.

    Args:
        metrics_df: DataFrame with model metrics
        save_path: Path to save figure

    Returns:
        Figure object
    """
    metrics_to_rank = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']
    available_metrics = [m for m in metrics_to_rank if m in metrics_df.columns]

    if not available_metrics:
        return None

    # Calculate ranks (1 = best)
    ranks_df = pd.DataFrame()
    for metric in available_metrics:
        # Rank in descending order (higher is better), NaN gets worst rank
        ranks_df[metric] = metrics_df[metric].rank(ascending=False, na_option='bottom')

    # Average rank
    ranks_df['Average Rank'] = ranks_df.mean(axis=1)
    ranks_df = ranks_df.sort_values('Average Rank')

    fig, ax = plt.subplots(figsize=(12, 6))

    # Heatmap
    sns.heatmap(
        ranks_df,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn_r',
        cbar_kws={'label': 'Rank (1 = Best)'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )

    ax.set_title('Model Ranking Across Metrics', fontsize=15, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved model ranking to {save_path}")

    return fig


# Example usage
if __name__ == '__main__':
    print("=== Testing Model Comparison Visualization ===\n")

    # Create dummy data
    np.random.seed(42)

    models = ['Logistic Regression', 'Random Forest', 'XGBoost']
    n_samples = 100

    models_results = {}
    for model in models:
        y_true = np.random.randint(0, 2, n_samples)
        y_proba = np.random.rand(n_samples)
        y_pred = (y_proba > 0.5).astype(int)

        models_results[model] = {
            'y_true': y_true,
            'y_proba': y_proba,
            'y_pred': y_pred
        }

    # Test ROC curves
    print("Testing ROC curves...")
    fig1 = plot_roc_curves(models_results)
    plt.close()

    # Test performance comparison
    print("\nTesting performance comparison...")
    metrics_df = pd.DataFrame({
        'Accuracy': [0.85, 0.82, 0.88],
        'Sensitivity': [0.75, 0.70, 0.80],
        'Specificity': [0.90, 0.85, 0.92],
        'F1-Score': [0.80, 0.75, 0.83],
        'AUC-ROC': [0.87, 0.83, 0.90]
    }, index=models)

    fig2 = plot_performance_comparison(metrics_df)
    plt.close()

    # Test confusion matrices
    print("\nTesting confusion matrices...")
    fig3 = plot_confusion_matrices(models_results)
    plt.close()

    # Test ranking
    print("\nTesting model ranking...")
    fig4 = plot_model_ranking(metrics_df)
    plt.close()

    print("\n✓ Visualization tests complete!")
