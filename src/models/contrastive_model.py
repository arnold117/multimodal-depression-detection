"""
Contrastive Learning for Depression Detection

Implements SimCLR-style contrastive learning:
1. Self-supervised pretraining on augmented views
2. NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
3. Downstream classifier fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path

from src.models.pytorch_base import BaseDeepModel
from src.utils.augmentation import TabularAugmentation


class ContrastiveEncoder(BaseDeepModel):
    """SimCLR-style contrastive learning encoder."""

    def __init__(
        self,
        input_dim: int = 52,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        projection_dim: int = 32,
        dropout: float = 0.2,
        device: Optional[str] = None
    ):
        """
        Initialize contrastive encoder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            embedding_dim: Embedding dimension (encoder output)
            projection_dim: Projection head dimension (for contrastive loss)
            dropout: Dropout rate
            device: Device ('mps', 'cuda', 'cpu')
        """
        super().__init__(device)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )

        # Projection head (for contrastive learning)
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

        # Augmenter
        self.augmenter = TabularAugmentation(augmentation_strength=0.2)

    def forward(
        self,
        x: torch.Tensor,
        return_projection: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input features [batch, input_dim]
            return_projection: Whether to return projection

        Returns:
            Tuple of (embedding, projection) if return_projection=True
            Otherwise just embedding
        """
        # Encoder
        h = self.encoder(x)  # Representation/embedding

        if return_projection:
            # Projection (for contrastive loss)
            z = self.projector(h)
            return h, z
        else:
            return h

    def get_embeddings(self, x: torch.Tensor) -> np.ndarray:
        """Get embeddings for visualization."""
        self.eval()
        x = x.to(self.device)

        with torch.no_grad():
            h = self.forward(x, return_projection=False)

        return h.cpu().numpy()


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)."""

    def __init__(self, temperature: float = 0.5, batch_size: int = 16):
        """
        Initialize NT-Xent loss.

        Args:
            temperature: Temperature parameter
            batch_size: Batch size
        """
        super().__init__()
        self.temperature = temperature
        self.batch_size = batch_size

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss.

        Args:
            z_i: First view projections [batch, projection_dim]
            z_j: Second view projections [batch, projection_dim]

        Returns:
            Loss value
        """
        batch_size = z_i.shape[0]

        # Normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate views
        z = torch.cat([z_i, z_j], dim=0)  # [2*batch, projection_dim]

        # Compute similarity matrix
        similarity_matrix = torch.mm(z, z.T)  # [2*batch, 2*batch]
        similarity_matrix = similarity_matrix / self.temperature

        # Create mask for positive pairs
        # For each sample i, its positive pair is at index i+batch (and vice versa)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)

        # Positive pairs
        positives = torch.cat([
            torch.diag(similarity_matrix, batch_size),  # z_i vs z_j
            torch.diag(similarity_matrix, -batch_size)  # z_j vs z_i
        ], dim=0)

        # Negatives (all except self and positive pair)
        negatives = similarity_matrix[~mask].view(2 * batch_size, -1)

        # Compute loss
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)

        loss = F.cross_entropy(logits, labels)

        return loss


class ContrastiveClassifier(BaseDeepModel):
    """Classifier built on top of contrastive encoder."""

    def __init__(
        self,
        encoder: ContrastiveEncoder,
        num_classes: int = 2,
        freeze_encoder: bool = False,
        dropout: float = 0.3
    ):
        """
        Initialize classifier.

        Args:
            encoder: Pretrained contrastive encoder
            num_classes: Number of classes
            freeze_encoder: Whether to freeze encoder weights
            dropout: Dropout rate
        """
        super().__init__(device=str(encoder.device))

        self.encoder = encoder

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.encoder.parameters():
                param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(encoder.embedding_dim, encoder.embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder.embedding_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Get embeddings from encoder
        h = self.encoder.forward(x, return_projection=False)

        # Classify
        logits = self.classifier(h)

        return logits

    def get_embeddings(self, x: torch.Tensor) -> np.ndarray:
        """Get embeddings."""
        return self.encoder.get_embeddings(x)


def pretrain_contrastive(
    encoder: ContrastiveEncoder,
    X: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    temperature: float = 0.5,
    verbose: bool = True
) -> Dict:
    """
    Pretrain encoder with contrastive learning.

    Args:
        encoder: Contrastive encoder model
        X: Input features [n_samples, n_features]
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        temperature: NT-Xent temperature
        verbose: Print progress

    Returns:
        Training history
    """
    from torch.utils.data import DataLoader, TensorDataset

    # Create dataset
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = NTXentLoss(temperature=temperature, batch_size=batch_size)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    # Training
    encoder.to(encoder.device)
    encoder.train()

    history = {'loss': []}

    for epoch in range(epochs):
        epoch_losses = []

        for batch in dataloader:
            x = batch[0].to(encoder.device)

            # Create two augmented views
            view1, view2 = encoder.augmenter.create_positive_pairs(x)

            # Forward pass
            _, z1 = encoder(view1, return_projection=True)
            _, z2 = encoder(view2, return_projection=True)

            # Compute loss
            loss = criterion(z1, z2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        history['loss'].append(avg_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    return history


# Example usage and testing
if __name__ == '__main__':
    import sys
    sys.path.append('/Users/arnold/Documents/Dev/multimodal-depression-detection')
    from src.utils.data_loader import load_features_labels, load_config
    from src.utils.pytorch_utils import set_seed

    print("=== Testing Contrastive Learning Model ===\n")

    # Set seed
    set_seed(42)

    # Load data
    X, y, _ = load_features_labels()
    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Load config
    config = load_config()
    contrastive_config = config.get('contrastive', {})

    # Create encoder
    encoder = ContrastiveEncoder(
        input_dim=X.shape[1],
        hidden_dim=64,
        embedding_dim=32,
        projection_dim=contrastive_config.get('projection_dim', 32),
        dropout=0.2
    )

    print(f"\nEncoder device: {encoder.device}")
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # Test forward pass
    print("\nTesting forward pass...")
    x_sample = torch.FloatTensor(X[:4]).to(encoder.device)

    h, z = encoder(x_sample, return_projection=True)
    print(f"Input shape: {x_sample.shape}")
    print(f"Embedding shape: {h.shape}")
    print(f"Projection shape: {z.shape}")

    # Test augmentation
    print("\nTesting augmentation...")
    view1, view2 = encoder.augmenter.create_positive_pairs(x_sample)
    print(f"View 1 shape: {view1.shape}")
    print(f"View 2 shape: {view2.shape}")

    # Test NT-Xent loss
    print("\nTesting NT-Xent loss...")
    _, z1 = encoder(view1, return_projection=True)
    _, z2 = encoder(view2, return_projection=True)

    criterion = NTXentLoss(temperature=0.5, batch_size=4)
    loss = criterion(z1, z2)
    print(f"NT-Xent loss: {loss.item():.4f}")

    # Test pretraining (small test)
    print("\nTesting pretraining (10 epochs)...")
    X_tensor = torch.FloatTensor(X)
    history = pretrain_contrastive(
        encoder, X_tensor,
        epochs=10,
        batch_size=16,
        learning_rate=0.001,
        temperature=0.5,
        verbose=False
    )
    print(f"Pretraining losses: {history['loss'][:3]}... {history['loss'][-3:]}")

    # Test classifier
    print("\nTesting classifier...")
    classifier = ContrastiveClassifier(
        encoder, num_classes=2, freeze_encoder=True
    )
    print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")

    logits = classifier(x_sample)
    print(f"Logits shape: {logits.shape}")

    print("\n✓ Contrastive learning test complete!")
