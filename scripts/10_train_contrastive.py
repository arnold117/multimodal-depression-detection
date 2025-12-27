#!/usr/bin/env python3
"""
Train Contrastive Learning Model (Phase 5C)

Two-stage training:
1. Self-supervised pretraining with NT-Xent loss
2. Supervised fine-tuning with downstream classifier

Usage:
    mamba activate qbio
    python scripts/10_train_contrastive.py
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
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import umap

# Add project root
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.data_loader import load_config, load_features_labels
from src.utils.pytorch_utils import set_seed, get_device, get_dataloaders
from src.models.contrastive_model import (
    ContrastiveEncoder,
    ContrastiveClassifier,
    pretrain_contrastive
)

sns.set_style('whitegrid')


def plot_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = 'tsne',
    save_path: str = None
):
    """Plot embeddings with dimensionality reduction."""
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
        title = 't-SNE Projection of Contrastive Embeddings'
    else:  # umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
        title = 'UMAP Projection of Contrastive Embeddings'

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Negative class
    neg_mask = labels == 0
    ax.scatter(
        embedding_2d[neg_mask, 0], embedding_2d[neg_mask, 1],
        c='blue', label=f'Negative (n={neg_mask.sum()})',
        alpha=0.6, s=100, edgecolors='black', linewidth=0.5
    )

    # Positive class
    pos_mask = labels == 1
    ax.scatter(
        embedding_2d[pos_mask, 0], embedding_2d[pos_mask, 1],
        c='red', label=f'Positive (n={pos_mask.sum()})',
        alpha=0.8, s=150, marker='*', edgecolors='black', linewidth=1
    )

    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved embeddings plot to {save_path}")

    return fig


def evaluate_classifier(
    classifier: ContrastiveClassifier,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    verbose: bool = True
) -> dict:
    """
    Evaluate classifier with stratified k-fold CV.

    Args:
        classifier: Trained classifier
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

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)

        # Create dataloaders
        train_loader, val_loader = get_dataloaders(
            X_train, y_train, X_val, y_val, batch_size=16
        )

        # Train classifier (keep encoder frozen)
        optimizer = torch.optim.Adam(
            classifier.classifier.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        criterion = nn.CrossEntropyLoss()

        classifier.fit(
            train_loader,
            val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=100,
            patience=20,
            verbose=False
        )

        # Evaluate
        classifier.eval()
        with torch.no_grad():
            X_val_device = X_val_t.to(classifier.device)
            logits = classifier(X_val_device)
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
    print("CONTRASTIVE LEARNING TRAINING - PHASE 5C")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load config
    config = load_config()
    contrastive_config = config.get('contrastive', {
        'temperature': 0.5,
        'projection_dim': 32,
        'augmentation_strength': 0.2,
        'pretrain_epochs': 200
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

    # Stage 1: Self-supervised pretraining
    print("\n" + "=" * 80)
    print("Stage 1: Self-Supervised Pretraining")
    print("=" * 80)

    encoder = ContrastiveEncoder(
        input_dim=X.shape[1],
        hidden_dim=64,
        embedding_dim=32,
        projection_dim=contrastive_config['projection_dim'],
        dropout=0.2,
        device=str(device)
    ).to(device)

    print(f"\nEncoder device: {encoder.device}")
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # Pretrain
    X_tensor = torch.FloatTensor(X)

    print(f"\nPretraining with NT-Xent loss...")
    print(f"  Temperature: {contrastive_config['temperature']}")
    print(f"  Augmentation strength: {contrastive_config['augmentation_strength']}")

    pretrain_history = pretrain_contrastive(
        encoder,
        X_tensor,
        epochs=contrastive_config.get('pretrain_epochs', 200),
        batch_size=common_config['batch_size'],
        learning_rate=common_config['learning_rate'],
        temperature=contrastive_config['temperature'],
        verbose=True
    )

    # Save pretrained encoder
    encoder.save(str(models_dir / 'contrastive_encoder.pth'))

    # Plot pretraining history
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pretrain_history['loss'], linewidth=2, label='Contrastive Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Contrastive Pretraining History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'contrastive_pretrain_history.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved pretraining history to {figures_dir / 'contrastive_pretrain_history.png'}")

    # Visualize embeddings after pretraining
    print("\n" + "=" * 80)
    print("Analyzing Learned Embeddings")
    print("=" * 80)

    embeddings = encoder.get_embeddings(X_tensor)
    print(f"Embedding shape: {embeddings.shape}")

    # t-SNE
    print("\nGenerating t-SNE visualization...")
    plot_embeddings(
        embeddings, y, method='tsne',
        save_path=figures_dir / 'contrastive_embeddings_tsne.png'
    )

    # UMAP
    print("Generating UMAP visualization...")
    plot_embeddings(
        embeddings, y, method='umap',
        save_path=figures_dir / 'contrastive_embeddings_umap.png'
    )

    # Stage 2: Supervised fine-tuning
    print("\n" + "=" * 80)
    print("Stage 2: Supervised Fine-Tuning")
    print("=" * 80)

    # Create classifier with frozen encoder
    classifier = ContrastiveClassifier(
        encoder,
        num_classes=2,
        freeze_encoder=True,  # Only train classifier head
        dropout=0.3
    ).to(device)

    print(f"\nClassifier device: {classifier.device}")
    print(f"Trainable parameters: {sum(p.numel() for p in classifier.parameters() if p.requires_grad):,}")
    print(f"Total parameters: {sum(p.numel() for p in classifier.parameters()):,}")

    # Evaluate with CV
    print("\nEvaluating downstream classifier...")
    cv_results = evaluate_classifier(
        classifier, X, y,
        n_splits=5,
        verbose=True
    )

    # Train final model on all data
    print("\n" + "=" * 80)
    print("Training Final Classifier")
    print("=" * 80)

    train_loader, _ = get_dataloaders(X, y, batch_size=common_config['batch_size'])

    optimizer = torch.optim.Adam(
        classifier.classifier.parameters(),
        lr=common_config['learning_rate'],
        weight_decay=common_config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()

    final_history = classifier.fit(
        train_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=100,
        patience=20,
        verbose=True,
        save_path=str(models_dir / 'contrastive_classifier.pth')
    )

    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    results = {
        'config': contrastive_config,
        'pretrain_history': {
            'loss': [float(x) for x in pretrain_history['loss']],
            'epochs': len(pretrain_history['loss'])
        },
        'cv_metrics': {
            'accuracy': float(cv_results['accuracy']),
            'auc': float(cv_results['auc']) if not np.isnan(cv_results['auc']) else None
        },
        'final_training': {
            'train_loss': [float(x) for x in final_history['train_loss']],
            'epochs': len(final_history['train_loss'])
        }
    }

    import json
    with open(metrics_dir / 'contrastive_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved results to {metrics_dir / 'contrastive_results.json'}")

    # Plot final training history
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(final_history['train_loss'], linewidth=2, label='Training Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Classifier Fine-Tuning History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'contrastive_finetune_history.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved fine-tuning history to {figures_dir / 'contrastive_finetune_history.png'}")

    print("\n" + "=" * 80)
    print("✓ CONTRASTIVE LEARNING COMPLETE")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print(f"\nOutputs:")
    print(f"  Models: {models_dir}")
    print(f"  Figures: {figures_dir}")
    print(f"  Metrics: {metrics_dir}")

    print(f"\nKey Results:")
    print(f"  Pretraining final loss: {pretrain_history['loss'][-1]:.4f}")
    print(f"  CV Accuracy: {cv_results['accuracy']:.4f}")
    print(f"  CV AUC-ROC: {cv_results['auc']:.4f}" if not np.isnan(cv_results['auc']) else "  CV AUC-ROC: N/A")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
