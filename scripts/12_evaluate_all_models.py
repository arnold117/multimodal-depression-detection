#!/usr/bin/env python3
"""
Evaluate and Compare All Models (Phase 6)

Compares performance across:
- Baseline models (Logistic, RF, XGBoost)
- Deep learning models (VAE, GNN, Contrastive, Transformer)

Generates:
- ROC curve comparison
- Performance metrics table
- Confusion matrices
- Model ranking

Usage:
    mamba activate qbio
    python scripts/12_evaluate_all_models.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# Add project root
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.data_loader import load_config, load_features_labels
from src.utils.pytorch_utils import set_seed, get_device
from src.models.gnn_model import DepGraphNet
from src.models.contrastive_model import ContrastiveClassifier, ContrastiveEncoder
from src.models.multimodal_transformer import MultimodalTransformer, ModalityFeatureSplitter
from src.features.graph_builder import UserSimilarityGraph
from src.visualization.compare_models import (
    plot_roc_curves,
    plot_performance_comparison,
    plot_confusion_matrices,
    create_model_comparison_table,
    plot_model_ranking
)

sns.set_style('whitegrid')


def evaluate_baseline_models(X, y, models_dir):
    """Evaluate baseline models with cross-validation."""
    print("\n" + "=" * 80)
    print("Evaluating Baseline Models")
    print("=" * 80)

    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    baseline_names = ['logistic', 'random_forest', 'xgboost']

    for name in baseline_names:
        model_path = models_dir / f'{name}_baseline.pkl'

        if not model_path.exists():
            print(f"Warning: {model_path} not found, skipping {name}")
            continue

        print(f"\nEvaluating {name}...")

        all_preds = []
        all_probs = []
        all_labels = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Load model dictionary (contains 'model', 'scaler', etc.)
            model_dict = joblib.load(model_path)
            model = model_dict['model']

            # Retrain model (baseline models are lightweight)
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_val)
            try:
                y_proba = model.predict_proba(X_val)[:, 1]
            except:
                y_proba = y_pred.astype(float)

            all_preds.extend(y_pred)
            all_probs.extend(y_proba)
            all_labels.extend(y_val)

        results[name.replace('_', ' ').title()] = {
            'y_true': np.array(all_labels),
            'y_pred': np.array(all_preds),
            'y_proba': np.array(all_probs)
        }

    return results


def evaluate_deep_learning_models(X, y, models_dir, config, device):
    """Evaluate deep learning models."""
    print("\n" + "=" * 80)
    print("Evaluating Deep Learning Models")
    print("=" * 80)

    results = {}

    # GNN
    print("\nEvaluating GNN...")
    gnn_path = models_dir / 'gnn_final.pth'
    if gnn_path.exists():
        try:
            # Build graph
            graph_builder = UserSimilarityGraph(metric='cosine')
            edge_index, _ = graph_builder.build_knn_graph(X, k=5)

            # Create model
            gnn = DepGraphNet(
                in_channels=X.shape[1],
                hidden_channels=16,
                num_layers=2,
                device=str(device)
            ).to(device)
            gnn.load(str(gnn_path))

            # Predict
            X_tensor = torch.FloatTensor(X).to(device)
            edge_index = edge_index.to(device)

            gnn.eval()
            with torch.no_grad():
                logits = gnn(X_tensor, edge_index)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            y_proba = probs[:, 1]
            y_pred = np.argmax(probs, axis=1)

            results['GNN'] = {
                'y_true': y,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
        except Exception as e:
            print(f"  Error evaluating GNN: {e}")

    # Contrastive Learning
    print("\nEvaluating Contrastive Learning...")
    contrastive_encoder_path = models_dir / 'contrastive_encoder.pth'
    contrastive_classifier_path = models_dir / 'contrastive_classifier.pth'

    if contrastive_encoder_path.exists():
        try:
            # Load encoder
            encoder = ContrastiveEncoder(
                input_dim=X.shape[1],
                hidden_dim=64,
                embedding_dim=32,
                device=str(device)
            ).to(device)
            encoder.load(str(contrastive_encoder_path))

            # Create and load classifier
            classifier = ContrastiveClassifier(
                encoder, num_classes=2, freeze_encoder=True
            ).to(device)
            classifier.load(str(contrastive_classifier_path))

            # Predict
            X_tensor = torch.FloatTensor(X).to(device)

            classifier.eval()
            with torch.no_grad():
                logits = classifier(X_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            y_proba = probs[:, 1]
            y_pred = np.argmax(probs, axis=1)

            results['Contrastive'] = {
                'y_true': y,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
        except Exception as e:
            print(f"  Error evaluating Contrastive: {e}")

    # Transformer
    print("\nEvaluating Transformer...")
    transformer_path = models_dir / 'transformer_best.pth'

    if transformer_path.exists():
        try:
            modality_dims = ModalityFeatureSplitter.get_modality_dims()

            transformer = MultimodalTransformer(
                modality_dims=modality_dims,
                d_model=16,
                nhead=4,
                num_layers=2,
                device=str(device)
            ).to(device)
            transformer.load(str(transformer_path))

            # Predict
            X_tensor = torch.FloatTensor(X).to(device)

            transformer.eval()
            with torch.no_grad():
                logits = transformer(X_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            y_proba = probs[:, 1]
            y_pred = np.argmax(probs, axis=1)

            results['Transformer'] = {
                'y_true': y,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
        except Exception as e:
            print(f"  Error evaluating Transformer: {e}")

    return results


def calculate_metrics(results_dict):
    """Calculate all metrics for each model."""
    metrics_data = []

    for name, results in results_dict.items():
        y_true = results['y_true']
        y_pred = results['y_pred']
        y_proba = results['y_proba']

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Handle zero division
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Specificity (TN / (TN + FP))
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            specificity = np.nan

        # AUC metrics
        try:
            auc_roc = roc_auc_score(y_true, y_proba)
        except (ValueError, Exception):
            auc_roc = np.nan

        try:
            pr_auc = average_precision_score(y_true, y_proba)
        except (ValueError, Exception):
            pr_auc = np.nan

        metrics_data.append({
            'Model': name,
            'Accuracy': accuracy,
            'Sensitivity': recall,
            'Specificity': specificity,
            'Precision': precision,
            'F1-Score': f1,
            'AUC-ROC': auc_roc,
            'PR-AUC': pr_auc
        })

    return pd.DataFrame(metrics_data).set_index('Model')


def main():
    print("=" * 80)
    print("MODEL COMPARISON & EVALUATION - PHASE 6")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load config
    config = load_config()
    common_config = config['common']

    # Set seed
    set_seed(common_config['random_seed'])

    # Get device
    device = get_device(config['device'].get('force_device'))

    # Paths
    models_dir = Path(config['paths']['results']['models'])
    figures_dir = Path(config['paths']['results']['figures'])
    tables_dir = Path(config['paths']['results'].get('tables', 'results/tables'))
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    X, y, feature_names = load_features_labels()
    print(f"✓ Loaded {X.shape[0]} users with {X.shape[1]} features")

    # Evaluate all models
    all_results = {}

    # Baseline models
    baseline_results = evaluate_baseline_models(X, y, models_dir)
    all_results.update(baseline_results)

    # Deep learning models
    dl_results = evaluate_deep_learning_models(X, y, models_dir, config, device)
    all_results.update(dl_results)

    # Calculate metrics
    print("\n" + "=" * 80)
    print("Calculating Metrics")
    print("=" * 80)

    metrics_df = calculate_metrics(all_results)
    print("\n" + metrics_df.to_string())

    # Create visualizations
    print("\n" + "=" * 80)
    print("Creating Visualizations")
    print("=" * 80)

    # ROC curves
    print("\nGenerating ROC curves...")
    plot_roc_curves(
        all_results,
        save_path=figures_dir / 'all_models_roc_comparison.png'
    )

    # Performance comparison
    print("Generating performance comparison...")
    plot_performance_comparison(
        metrics_df,
        save_path=figures_dir / 'all_models_performance_comparison.png'
    )

    # Confusion matrices
    print("Generating confusion matrices...")
    plot_confusion_matrices(
        all_results,
        save_path=figures_dir / 'all_models_confusion_matrices.png'
    )

    # Model ranking
    print("Generating model ranking...")
    plot_model_ranking(
        metrics_df,
        save_path=figures_dir / 'all_models_ranking.png'
    )

    # Save comparison table
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    # Save raw metrics
    metrics_df.to_csv(tables_dir / 'model_comparison_raw.csv')
    print(f"✓ Saved raw metrics to {tables_dir / 'model_comparison_raw.csv'}")

    # Save formatted table
    formatted_df = create_model_comparison_table(
        metrics_df,
        save_path=tables_dir / 'model_comparison_formatted.csv'
    )

    # Print formatted table
    print("\n" + "=" * 80)
    print("FORMATTED COMPARISON TABLE")
    print("=" * 80)
    print(formatted_df.to_string())

    # Identify best models per metric
    print("\n" + "=" * 80)
    print("Best Models Per Metric")
    print("=" * 80)

    for metric in ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']:
        if metric in metrics_df.columns:
            best_model = metrics_df[metric].idxmax()
            best_value = metrics_df.loc[best_model, metric]
            if not np.isnan(best_value):
                print(f"  {metric}: {best_model} ({best_value:.3f})")

    # Save summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'n_models': len(all_results),
        'models_evaluated': list(all_results.keys()),
        'best_models': {
            metric: {
                'model': str(metrics_df[metric].idxmax()),
                'value': float(metrics_df[metric].max())
            }
            for metric in ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']
            if metric in metrics_df.columns and not metrics_df[metric].isna().all()
        },
        'metrics': metrics_df.to_dict()
    }

    with open(tables_dir / 'model_comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Saved summary to {tables_dir / 'model_comparison_summary.json'}")

    print("\n" + "=" * 80)
    print("✓ MODEL COMPARISON COMPLETE")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print(f"\nOutputs:")
    print(f"  Figures: {figures_dir}")
    print(f"  Tables: {tables_dir}")
    print(f"\nKey takeaways:")
    print(f"  - {len(all_results)} models compared")
    print(f"  - Best overall accuracy: {metrics_df['Accuracy'].max():.1%}")
    if 'Sensitivity' in metrics_df.columns:
        print(f"  - Best sensitivity: {metrics_df['Sensitivity'].max():.1%} ({metrics_df['Sensitivity'].idxmax()})")
    if 'AUC-ROC' in metrics_df.columns and not metrics_df['AUC-ROC'].isna().all():
        print(f"  - Best AUC-ROC: {metrics_df['AUC-ROC'].max():.3f} ({metrics_df['AUC-ROC'].idxmax()})")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
