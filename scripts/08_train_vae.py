#!/usr/bin/env python3
"""
Train Variational Autoencoder (Phase 5A)

Trains VAE for:
1. Learning low-dimensional latent representation
2. Anomaly detection via reconstruction error
3. Data augmentation - generate synthetic positive samples
4. Latent space visualization

Usage:
    mamba activate qbio
    python scripts/08_train_vae.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datetime import datetime
from sklearn.manifold import TSNE
import umap

# Add project root
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.data_loader import load_config, load_features_labels
from src.utils.pytorch_utils import get_dataloaders, set_seed, get_device
from src.models.vae_model import MultimodalVAE

sns.set_style('whitegrid')


def plot_latent_space(
    latent_mean: np.ndarray,
    labels: np.ndarray,
    method: str = 'tsne',
    save_path: str = None
):
    """Plot 2D visualization of latent space."""
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        embedding = reducer.fit_transform(latent_mean)
        title = 't-SNE Projection of VAE Latent Space'
    else:  # umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(latent_mean)
        title = 'UMAP Projection of VAE Latent Space'

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot negative class
    neg_mask = labels == 0
    ax.scatter(embedding[neg_mask, 0], embedding[neg_mask, 1],
              c='blue', label='Negative (n={})'.format(neg_mask.sum()),
              alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

    # Plot positive class
    pos_mask = labels == 1
    ax.scatter(embedding[pos_mask, 0], embedding[pos_mask, 1],
              c='red', label='Positive (n={})'.format(pos_mask.sum()),
              alpha=0.8, s=150, marker='*', edgecolors='black', linewidth=1)

    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved latent space plot to {save_path}")

    return fig


def plot_reconstruction_error(
    errors: np.ndarray,
    labels: np.ndarray,
    save_path: str = None
):
    """Plot reconstruction error distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(errors[labels == 0], bins=20, alpha=0.6,
                label='Negative', color='blue', edgecolor='black')
    axes[0].hist(errors[labels == 1], bins=10, alpha=0.6,
                label='Positive', color='red', edgecolor='black')
    axes[0].set_xlabel('Reconstruction Error', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Reconstruction Error Distribution', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # Box plot
    data_to_plot = [errors[labels == 0], errors[labels == 1]]
    bp = axes[1].boxplot(data_to_plot, labels=['Negative', 'Positive'],
                         patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    axes[1].set_ylabel('Reconstruction Error', fontsize=12)
    axes[1].set_title('Reconstruction Error by Class', fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')

    # Add means
    means = [errors[labels == 0].mean(), errors[labels == 1].mean()]
    axes[1].plot([1, 2], means, 'go', markersize=8, label='Mean')
    axes[1].legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved reconstruction error plot to {save_path}")

    return fig


def main():
    print("=" * 80)
    print("VAE TRAINING - PHASE 5A")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load config
    config = load_config()
    vae_config = config['vae']
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

    # Create dataloaders
    train_loader, _ = get_dataloaders(X, y, batch_size=common_config['batch_size'])

    # Create VAE
    print("\nCreating VAE model...")
    vae = MultimodalVAE(
        input_dim=X.shape[1],
        latent_dim=vae_config['latent_dim'],
        hidden_dims=vae_config['hidden_dims'],
        beta=vae_config['beta'],
        dropout=vae_config['dropout'],
        device=str(device)
    )

    print(f"Model device: {vae.device}")
    print(f"Parameters: {sum(p.numel() for p in vae.parameters()):,}")
    print(f"Architecture:")
    print(f"  Input: {X.shape[1]} dims")
    print(f"  Encoder: {X.shape[1]} → {' → '.join(map(str, vae_config['hidden_dims']))} → {vae_config['latent_dim']}")
    print(f"  Decoder: {vae_config['latent_dim']} → {' → '.join(map(str, reversed(vae_config['hidden_dims'])))} → {X.shape[1]}")

    # Train
    print("\n" + "=" * 80)
    print("Training VAE")
    print("=" * 80)

    optimizer = torch.optim.Adam(
        vae.parameters(),
        lr=vae_config['learning_rate'],
        weight_decay=vae_config['weight_decay']
    )

    history = vae.fit(
        train_loader,
        optimizer=optimizer,
        epochs=common_config['max_epochs'],
        patience=common_config['early_stopping_patience'],
        verbose=True,
        save_path=str(models_dir / 'vae_best.pth')
    )

    # Save final model
    vae.save(str(models_dir / 'vae_final.pth'))

    # Get latent representations
    print("\n" + "=" * 80)
    print("Analyzing Latent Space")
    print("=" * 80)

    X_tensor = torch.FloatTensor(X)
    latent_mean, latent_logvar = vae.get_latent_representation(X_tensor)

    print(f"Latent space shape: {latent_mean.shape}")

    # t-SNE visualization
    print("\nGenerating t-SNE visualization...")
    plot_latent_space(
        latent_mean, y, method='tsne',
        save_path=figures_dir / 'vae_latent_tsne.png'
    )

    # UMAP visualization
    print("Generating UMAP visualization...")
    plot_latent_space(
        latent_mean, y, method='umap',
        save_path=figures_dir / 'vae_latent_umap.png'
    )

    # Reconstruction error analysis
    print("\n" + "=" * 80)
    print("Anomaly Detection (Reconstruction Error)")
    print("=" * 80)

    errors = vae.get_reconstruction_error(X_tensor)

    print(f"\nReconstruction error statistics:")
    print(f"  Overall: {errors.mean():.4f} ± {errors.std():.4f}")
    print(f"  Negative class: {errors[y==0].mean():.4f} ± {errors[y==0].std():.4f}")
    print(f"  Positive class: {errors[y==1].mean():.4f} ± {errors[y==1].std():.4f}")

    # Statistical test
    from scipy.stats import mannwhitneyu
    stat, p_value = mannwhitneyu(errors[y==0], errors[y==1], alternative='two-sided')
    print(f"\nMann-Whitney U test:")
    print(f"  Statistic: {stat:.2f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"  ✓ Significant difference in reconstruction error")
    else:
        print(f"  ✗ No significant difference")

    plot_reconstruction_error(
        errors, y,
        save_path=figures_dir / 'vae_reconstruction_error.png'
    )

    # Generate synthetic samples
    print("\n" + "=" * 80)
    print("Data Augmentation (Synthetic Sample Generation)")
    print("=" * 80)

    n_synthetic = vae_config['n_synthetic_samples']
    print(f"\nGenerating {n_synthetic} synthetic positive samples...")

    y_tensor = torch.LongTensor(y)
    synthetic_samples = vae.generate_synthetic_samples(
        n_samples=n_synthetic,
        class_label=1,  # Generate positive samples
        X=X_tensor,
        y=y_tensor
    )

    print(f"✓ Generated {synthetic_samples.shape[0]} synthetic samples")
    print(f"  Shape: {synthetic_samples.shape}")
    print(f"  Mean: {synthetic_samples.mean():.4f}")
    print(f"  Std: {synthetic_samples.std():.4f}")

    # Save synthetic samples
    synthetic_df = pd.DataFrame(synthetic_samples, columns=feature_names)
    synthetic_path = Path('data/processed/features/vae_synthetic_samples.parquet')
    synthetic_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_parquet(synthetic_path, index=False)
    print(f"✓ Saved synthetic samples to {synthetic_path}")

    # Save metrics
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    results = {
        'config': vae_config,
        'training_history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'epochs_trained': len(history['train_loss'])
        },
        'reconstruction_error': {
            'overall_mean': float(errors.mean()),
            'overall_std': float(errors.std()),
            'negative_mean': float(errors[y==0].mean()),
            'negative_std': float(errors[y==0].std()),
            'positive_mean': float(errors[y==1].mean()),
            'positive_std': float(errors[y==1].std()),
            'mann_whitney_u': float(stat),
            'p_value': float(p_value)
        },
        'synthetic_generation': {
            'n_samples': n_synthetic,
            'save_path': str(synthetic_path)
        }
    }

    import json
    with open(metrics_dir / 'vae_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved results to {metrics_dir / 'vae_results.json'}")

    # Plot training history
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['train_loss'], linewidth=2, label='Training Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('VAE Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'vae_training_history.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved training history to {figures_dir / 'vae_training_history.png'}")

    print("\n" + "=" * 80)
    print("✓ VAE TRAINING COMPLETE")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print(f"\nOutputs:")
    print(f"  Models: {models_dir}")
    print(f"  Figures: {figures_dir}")
    print(f"  Metrics: {metrics_dir}")
    print(f"  Synthetic data: {synthetic_path}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
