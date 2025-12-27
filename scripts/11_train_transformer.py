#!/usr/bin/env python3
"""
Train Multimodal Transformer (Phase 5D)

Treats 4 behavioral modalities as tokens with cross-modal attention:
- GPS/Location
- App Usage
- Communication
- Activity/Motion

Usage:
    mamba activate qbio
    python scripts/11_train_transformer.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Add project root
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.data_loader import load_config, load_features_labels
from src.utils.pytorch_utils import set_seed, get_device, get_dataloaders
from src.models.multimodal_transformer import (
    MultimodalTransformer,
    ModalityFeatureSplitter
)

sns.set_style('whitegrid')


def plot_modality_importance(
    importance: np.ndarray,
    labels: np.ndarray,
    modality_names: list,
    save_path: str = None
):
    """Plot modality importance by class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Average importance per modality
    avg_importance = importance.mean(axis=0)

    axes[0].bar(range(len(modality_names)), avg_importance, color='steelblue', edgecolor='black')
    axes[0].set_xticks(range(len(modality_names)))
    axes[0].set_xticklabels(modality_names, rotation=45, ha='right')
    axes[0].set_ylabel('Average Importance', fontsize=12)
    axes[0].set_title('Modality Importance (Overall)', fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='y')

    # Importance by class
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_importance = importance[pos_mask].mean(axis=0) if pos_mask.sum() > 0 else np.zeros(len(modality_names))
    neg_importance = importance[neg_mask].mean(axis=0) if neg_mask.sum() > 0 else np.zeros(len(modality_names))

    x = np.arange(len(modality_names))
    width = 0.35

    axes[1].bar(x - width/2, neg_importance, width, label='Negative', color='blue', alpha=0.7, edgecolor='black')
    axes[1].bar(x + width/2, pos_importance, width, label='Positive', color='red', alpha=0.7, edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(modality_names, rotation=45, ha='right')
    axes[1].set_ylabel('Average Importance', fontsize=12)
    axes[1].set_title('Modality Importance by Class', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved modality importance plot to {save_path}")

    return fig


def evaluate_transformer(
    model: MultimodalTransformer,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    verbose: bool = True
) -> dict:
    """
    Evaluate transformer with stratified k-fold CV.

    Args:
        model: Transformer model
        X: Features
        y: Labels
        n_splits: Number of CV folds
        verbose: Print progress

    Returns:
        Dictionary with predictions and metrics
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_preds = []
    all_probs = []
    all_labels = []

    if verbose:
        print(f"\nRunning {n_splits}-fold CV...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        if verbose:
            print(f"Fold {fold+1}/{n_splits}...")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create dataloaders
        train_loader, val_loader = get_dataloaders(
            X_train, y_train, X_val, y_val, batch_size=16
        )

        # Reinitialize model for each fold
        model_fold = MultimodalTransformer(
            modality_dims=model.modality_dims,
            d_model=model.d_model,
            nhead=model.nhead,
            num_layers=2,
            dropout=0.2,
            num_classes=2,
            device=str(model.device)
        ).to(model.device)

        # Train
        optimizer = torch.optim.Adam(
            model_fold.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        criterion = nn.CrossEntropyLoss()

        model_fold.fit(
            train_loader,
            val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=100,
            patience=20,
            verbose=False
        )

        # Evaluate
        model_fold.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val).to(model_fold.device)
            logits = model_fold(X_val_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

        all_preds.extend(preds)
        all_probs.extend(probs[:, 1])
        all_labels.extend(y_val)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = np.nan

    if verbose:
        print(f"\n{'='*80}")
        print("CV Results:")
        print(f"{'='*80}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc:.4f}" if not np.isnan(auc) else "AUC-ROC: N/A")
        print(f"\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=['Negative', 'Positive']))

    return {
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels,
        'accuracy': accuracy,
        'auc': auc
    }


def main():
    print("=" * 80)
    print("MULTIMODAL TRANSFORMER TRAINING - PHASE 5D")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load config
    config = load_config()
    transformer_config = config.get('transformer', {
        'd_model': 16,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 64,
        'dropout': 0.2
    })
    common_config = config['common']

    # Set seed
    set_seed(common_config['random_seed'])

    # Get device
    device = get_device(config['device'].get('force_device'))

    # Create output directories
    models_dir = Path(config['paths']['results']['models'])
    figures_dir = Path(config['paths']['results']['figures'])
    metrics_dir = Path(config['paths']['results']['metrics'])

    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    X, y, feature_names = load_features_labels()
    print(f"✓ Loaded {X.shape[0]} users with {X.shape[1]} features")
    print(f"  Positive samples: {y.sum()}")
    print(f"  Negative samples: {len(y) - y.sum()}")

    # Get modality structure
    modality_dims = ModalityFeatureSplitter.get_modality_dims()
    modality_names = ModalityFeatureSplitter.get_modality_names()

    print(f"\nModality structure:")
    for name, dim in zip(modality_names, modality_dims):
        print(f"  {name}: {dim} features")
    print(f"  Total: {sum(modality_dims)} features")

    # Create model
    print("\n" + "=" * 80)
    print("Creating Multimodal Transformer")
    print("=" * 80)

    model = MultimodalTransformer(
        modality_dims=modality_dims,
        d_model=transformer_config['d_model'],
        nhead=transformer_config['nhead'],
        num_layers=transformer_config['num_layers'],
        dim_feedforward=transformer_config.get('dim_feedforward', 64),
        dropout=transformer_config['dropout'],
        num_classes=2,
        device=str(device)
    ).to(device)

    print(f"\nModel device: {model.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nArchitecture:")
    print(f"  Input: 4 modalities → {transformer_config['d_model']}-dim tokens")
    print(f"  Transformer: {transformer_config['num_layers']} layers, {transformer_config['nhead']} heads")
    print(f"  Output: 2 classes")

    # Cross-validation
    print("\n" + "=" * 80)
    print("Cross-Validation Evaluation")
    print("=" * 80)

    cv_results = evaluate_transformer(
        model, X, y,
        n_splits=5,
        verbose=True
    )

    # Train final model
    print("\n" + "=" * 80)
    print("Training Final Model")
    print("=" * 80)

    train_loader, _ = get_dataloaders(X, y, batch_size=common_config['batch_size'])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=common_config['learning_rate'],
        weight_decay=common_config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()

    history = model.fit(
        train_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=common_config['max_epochs'],
        patience=common_config['early_stopping_patience'],
        verbose=True,
        save_path=str(models_dir / 'transformer_best.pth')
    )

    # Analyze modality importance
    print("\n" + "=" * 80)
    print("Analyzing Modality Importance")
    print("=" * 80)

    X_tensor = torch.FloatTensor(X)
    importance = model.get_modality_importance(X_tensor)

    print(f"\nModality importance shape: {importance.shape}")
    print(f"\nAverage importance per modality:")
    for name, imp in zip(modality_names, importance.mean(axis=0)):
        print(f"  {name}: {imp:.4f}")

    # Plot importance
    plot_modality_importance(
        importance, y, modality_names,
        save_path=figures_dir / 'transformer_modality_importance.png'
    )

    # Analyze by class
    print(f"\nImportance by class:")
    print(f"  Negative class:")
    neg_mask = y == 0
    for name, imp in zip(modality_names, importance[neg_mask].mean(axis=0)):
        print(f"    {name}: {imp:.4f}")

    if y.sum() > 0:
        print(f"  Positive class:")
        pos_mask = y == 1
        for name, imp in zip(modality_names, importance[pos_mask].mean(axis=0)):
            print(f"    {name}: {imp:.4f}")

    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    results = {
        'config': transformer_config,
        'modality_structure': {
            name: int(dim) for name, dim in zip(modality_names, modality_dims)
        },
        'cv_metrics': {
            'accuracy': float(cv_results['accuracy']),
            'auc': float(cv_results['auc']) if not np.isnan(cv_results['auc']) else None
        },
        'training_history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'epochs': len(history['train_loss'])
        },
        'modality_importance': {
            'overall': {
                name: float(imp) for name, imp in zip(modality_names, importance.mean(axis=0))
            },
            'negative_class': {
                name: float(imp) for name, imp in zip(modality_names, importance[neg_mask].mean(axis=0))
            },
            'positive_class': {
                name: float(imp) for name, imp in zip(modality_names, importance[pos_mask].mean(axis=0))
            } if y.sum() > 0 else {}
        }
    }

    import json
    with open(metrics_dir / 'transformer_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved results to {metrics_dir / 'transformer_results.json'}")

    # Plot training history
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['train_loss'], linewidth=2, label='Training Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Transformer Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'transformer_training_history.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved training history to {figures_dir / 'transformer_training_history.png'}")

    print("\n" + "=" * 80)
    print("✓ MULTIMODAL TRANSFORMER COMPLETE")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print(f"\nOutputs:")
    print(f"  Models: {models_dir}")
    print(f"  Figures: {figures_dir}")
    print(f"  Metrics: {metrics_dir}")

    print(f"\nKey Results:")
    print(f"  CV Accuracy: {cv_results['accuracy']:.4f}")
    print(f"  CV AUC-ROC: {cv_results['auc']:.4f}" if not np.isnan(cv_results['auc']) else "  CV AUC-ROC: N/A")
    print(f"  Most important modality: {modality_names[np.argmax(importance.mean(axis=0))]}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
