"""
Variational Autoencoder (VAE) for Multimodal Features

Uses:
1. Learn low-dimensional latent representation of 44-dim features
2. Anomaly detection via reconstruction error
3. Data augmentation - generate synthetic positive samples
4. Latent space visualization with t-SNE/UMAP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path

from src.models.pytorch_base import BaseDeepModel


class MultimodalVAE(BaseDeepModel):
    """Variational Autoencoder for tabular multimodal features."""

    def __init__(
        self,
        input_dim: int = 52,
        latent_dim: int = 8,
        hidden_dims: list = [32, 16],
        beta: float = 1.0,
        dropout: float = 0.2,
        device: Optional[str] = None
    ):
        """
        Initialize VAE.

        Args:
            input_dim: Input feature dimension
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions for encoder/decoder
            beta: Beta-VAE weight for KL divergence
            dropout: Dropout rate
            device: Device ('mps', 'cuda', 'cpu')
        """
        super().__init__(device)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space (mean and log variance)
        self.fc_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder layers (reverse of encoder)
        decoder_layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            Tuple of (mean, logvar) [batch, latent_dim]
        """
        h = self.encoder(x)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mean + std * epsilon.

        Args:
            mean: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            Sampled latent vector z
        """
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruction.

        Args:
            z: Latent vector [batch, latent_dim]

        Returns:
            Reconstructed features [batch, input_dim]
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            Tuple of (reconstruction, mean, logvar)
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mean, logvar

    def loss_function(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        VAE loss = Reconstruction loss + Beta * KL divergence.

        Args:
            x: Original input
            reconstruction: Reconstructed input
            mean: Latent mean
            logvar: Latent log variance

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x, reduction='sum')

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        # Loss breakdown
        loss_dict = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }

        return total_loss, loss_dict

    def training_step(self, batch, criterion=None):
        """Override training step for VAE."""
        features, _ = batch  # Ignore labels for VAE
        features = features.to(self.device)

        # Forward pass
        reconstruction, mean, logvar = self(features)

        # Compute loss
        loss, _ = self.loss_function(features, reconstruction, mean, logvar)

        return loss / features.size(0)  # Normalize by batch size

    def get_reconstruction_error(
        self,
        x: torch.Tensor
    ) -> np.ndarray:
        """
        Calculate reconstruction error for anomaly detection.

        Args:
            x: Input features

        Returns:
            Reconstruction error per sample
        """
        self.eval()
        x = x.to(self.device)

        with torch.no_grad():
            reconstruction, _, _ = self(x)
            errors = torch.mean((x - reconstruction) ** 2, dim=1)

        return errors.cpu().numpy()

    def get_latent_representation(
        self,
        x: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get latent space representation.

        Args:
            x: Input features

        Returns:
            Tuple of (mean, logvar) as numpy arrays
        """
        self.eval()
        x = x.to(self.device)

        with torch.no_grad():
            mean, logvar = self.encode(x)

        return mean.cpu().numpy(), logvar.cpu().numpy()

    def generate_synthetic_samples(
        self,
        n_samples: int,
        class_label: int,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> np.ndarray:
        """
        Generate synthetic samples for data augmentation.

        Args:
            n_samples: Number of samples to generate
            class_label: Class to generate (0 or 1)
            X: Full feature matrix
            y: Full label vector

        Returns:
            Generated samples [n_samples, input_dim]
        """
        self.eval()

        # Get latent representations for target class
        class_indices = (y == class_label).nonzero(as_tuple=True)[0]
        class_samples = X[class_indices].to(self.device)

        with torch.no_grad():
            mean, logvar = self.encode(class_samples)

        # Calculate class distribution statistics
        class_mean = mean.mean(dim=0)
        class_std = torch.exp(0.5 * logvar.mean(dim=0))

        # Sample from class distribution
        z_samples = class_mean + class_std * torch.randn(n_samples, self.latent_dim).to(self.device)

        # Decode to generate samples
        with torch.no_grad():
            generated = self.decode(z_samples)

        return generated.cpu().numpy()


# Example usage and testing
if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    sys.path.append('/Users/arnold/Documents/Dev/multimodal-depression-detection')
    from src.utils.data_loader import load_features_labels, load_config
    from src.utils.pytorch_utils import get_dataloaders, set_seed

    print("=== Testing VAE Model ===\n")

    # Set seed
    set_seed(42)

    # Load data
    X, y, _ = load_features_labels()
    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Load config
    config = load_config()
    vae_config = config['vae']

    # Create dataloaders
    train_loader, _ = get_dataloaders(X, y, batch_size=16)

    # Create VAE
    vae = MultimodalVAE(
        input_dim=X.shape[1],
        latent_dim=vae_config['latent_dim'],
        hidden_dims=vae_config['hidden_dims'],
        beta=vae_config['beta'],
        dropout=vae_config['dropout']
    )

    print(f"VAE created on device: {vae.device}")
    print(f"Parameters: {sum(p.numel() for p in vae.parameters()):,}")

    # Test forward pass
    print("\nTesting forward pass...")
    sample_batch = next(iter(train_loader))
    sample_x, _ = sample_batch
    sample_x = sample_x[:4]  # Take 4 samples

    vae.to(vae.device)
    recon, mean, logvar = vae(sample_x.to(vae.device))

    print(f"Input shape: {sample_x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent mean shape: {mean.shape}")
    print(f"Latent logvar shape: {logvar.shape}")

    # Test loss
    loss, loss_dict = vae.loss_function(
        sample_x.to(vae.device), recon, mean, logvar
    )
    print(f"\nLoss breakdown:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    # Test synthetic generation
    print("\nGenerating synthetic samples...")
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    synthetic = vae.generate_synthetic_samples(
        n_samples=10,
        class_label=1,  # Generate positive samples
        X=X_tensor,
        y=y_tensor
    )
    print(f"Generated {synthetic.shape[0]} synthetic samples")

    print("\n✓ VAE model test complete!")
