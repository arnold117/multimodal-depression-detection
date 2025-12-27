"""
Multimodal Transformer for Depression Detection

Treats 4 behavioral modalities as separate tokens:
1. GPS/Location features
2. App usage features
3. Communication features
4. Activity/Motion features

Uses Transformer encoder to learn cross-modal interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path

from src.models.pytorch_base import BaseDeepModel


class MultimodalTransformer(BaseDeepModel):
    """Transformer model for multimodal behavioral features."""

    def __init__(
        self,
        modality_dims: List[int] = [11, 10, 11, 20],  # GPS, App, Comm, Activity
        d_model: int = 16,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.2,
        num_classes: int = 2,
        device: Optional[str] = None
    ):
        """
        Initialize multimodal transformer.

        Args:
            modality_dims: Dimensions of each modality [GPS, App, Comm, Activity]
            d_model: Transformer embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            num_classes: Number of output classes
            device: Device ('mps', 'cuda', 'cpu')
        """
        super().__init__(device)

        self.modality_dims = modality_dims
        self.num_modalities = len(modality_dims)
        self.d_model = d_model
        self.nhead = nhead

        # Modality embedding layers (project each modality to d_model)
        self.modality_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout)
            )
            for dim in modality_dims
        ])

        # Learnable positional encoding for modalities
        self.modality_pos_encoding = nn.Parameter(
            torch.randn(1, self.num_modalities, d_model)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * self.num_modalities, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        # For attention weight extraction
        self.attention_weights = None

    def split_modalities(
        self,
        x: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Split concatenated features into modality-specific features.

        Args:
            x: Concatenated features [batch, total_features]

        Returns:
            List of modality tensors
        """
        modality_features = []
        start_idx = 0

        for dim in self.modality_dims:
            modality_features.append(x[:, start_idx:start_idx + dim])
            start_idx += dim

        return modality_features

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input features [batch, total_features]
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (logits, attention_weights) if return_attention=True
            Otherwise just logits
        """
        batch_size = x.shape[0]

        # Split into modalities
        modality_features = self.split_modalities(x)

        # Embed each modality
        tokens = []
        for i, (embed, feat) in enumerate(zip(self.modality_embeddings, modality_features)):
            token = embed(feat)  # [batch, d_model]
            tokens.append(token)

        # Stack to form token sequence [batch, num_modalities, d_model]
        tokens = torch.stack(tokens, dim=1)

        # Add positional encoding
        tokens = tokens + self.modality_pos_encoding

        # Transformer encoding
        encoded = self.transformer(tokens)  # [batch, num_modalities, d_model]

        # Pool all modality tokens (concatenate)
        pooled = encoded.flatten(1)  # [batch, num_modalities * d_model]

        # Classification
        logits = self.classifier(pooled)

        if return_attention:
            # Extract attention weights from first layer
            attention = self._extract_attention_weights(tokens)
            return logits, attention
        else:
            return logits

    def _extract_attention_weights(
        self,
        tokens: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Extract attention weights from transformer.
        Note: This is a simplified version. For full attention extraction,
        would need to modify TransformerEncoder to return attention weights.

        Args:
            tokens: Input tokens

        Returns:
            Attention weights [batch, num_heads, num_modalities, num_modalities]
        """
        # This is a placeholder - actual implementation would require
        # modifying the transformer to expose attention weights
        return None

    def get_modality_importance(
        self,
        x: torch.Tensor
    ) -> np.ndarray:
        """
        Compute modality importance scores.
        Uses norm of attended modality representations.

        Args:
            x: Input features [batch, total_features]

        Returns:
            Modality importance [batch, num_modalities]
        """
        self.eval()
        x = x.to(self.device)

        with torch.no_grad():
            # Split into modalities
            modality_features = self.split_modalities(x)

            # Embed each modality
            tokens = []
            for i, (embed, feat) in enumerate(zip(self.modality_embeddings, modality_features)):
                token = embed(feat)
                tokens.append(token)

            tokens = torch.stack(tokens, dim=1)
            tokens = tokens + self.modality_pos_encoding

            # Pass through transformer
            encoded = self.transformer(tokens)

            # Compute importance as L2 norm of each modality's encoding
            importance = torch.norm(encoded, dim=2)  # [batch, num_modalities]

        return importance.cpu().numpy()

    def get_modality_embeddings(
        self,
        x: torch.Tensor
    ) -> np.ndarray:
        """
        Get embeddings for each modality.

        Args:
            x: Input features [batch, total_features]

        Returns:
            Modality embeddings [batch, num_modalities, d_model]
        """
        self.eval()
        x = x.to(self.device)

        with torch.no_grad():
            modality_features = self.split_modalities(x)

            tokens = []
            for i, (embed, feat) in enumerate(zip(self.modality_embeddings, modality_features)):
                token = embed(feat)
                tokens.append(token)

            tokens = torch.stack(tokens, dim=1)
            tokens = tokens + self.modality_pos_encoding

            encoded = self.transformer(tokens)

        return encoded.cpu().numpy()


class ModalityFeatureSplitter:
    """Helper class to manage modality feature indices."""

    # Default feature split (based on combined_features.parquet structure)
    # Total: 52 features
    DEFAULT_SPLITS = {
        'gps': (0, 11),        # 11 features: location variance, entropy, etc.
        'app': (11, 21),       # 10 features: app usage patterns
        'communication': (21, 32),  # 11 features: call/SMS patterns
        'activity': (32, 52)   # 20 features: motion + phone lock
    }

    @staticmethod
    def get_modality_dims() -> List[int]:
        """Get dimensions of each modality."""
        splits = ModalityFeatureSplitter.DEFAULT_SPLITS
        return [
            splits['gps'][1] - splits['gps'][0],
            splits['app'][1] - splits['app'][0],
            splits['communication'][1] - splits['communication'][0],
            splits['activity'][1] - splits['activity'][0]
        ]

    @staticmethod
    def get_modality_names() -> List[str]:
        """Get names of modalities."""
        return ['GPS/Location', 'App Usage', 'Communication', 'Activity/Motion']


# Example usage and testing
if __name__ == '__main__':
    import sys
    sys.path.append('/Users/arnold/Documents/Dev/multimodal-depression-detection')
    from src.utils.data_loader import load_features_labels, load_config
    from src.utils.pytorch_utils import set_seed

    print("=== Testing Multimodal Transformer ===\n")

    # Set seed
    set_seed(42)

    # Load data
    X, y, feature_names = load_features_labels()
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Total features: {X.shape[1]}")

    # Get modality dimensions
    modality_dims = ModalityFeatureSplitter.get_modality_dims()
    modality_names = ModalityFeatureSplitter.get_modality_names()

    print(f"\nModality structure:")
    for name, dim in zip(modality_names, modality_dims):
        print(f"  {name}: {dim} features")
    print(f"  Total: {sum(modality_dims)} features")

    # Load config
    config = load_config()
    transformer_config = config.get('transformer', {})

    # Create model
    model = MultimodalTransformer(
        modality_dims=modality_dims,
        d_model=transformer_config.get('d_model', 16),
        nhead=transformer_config.get('nhead', 4),
        num_layers=transformer_config.get('num_layers', 2),
        dim_feedforward=transformer_config.get('dim_feedforward', 64),
        dropout=transformer_config.get('dropout', 0.2),
        num_classes=2
    )

    print(f"\nModel device: {model.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    print("\nTesting forward pass...")
    x_sample = torch.FloatTensor(X[:4]).to(model.device)

    logits = model(x_sample)
    print(f"Input shape: {x_sample.shape}")
    print(f"Output logits shape: {logits.shape}")

    # Test modality splitting
    print("\nTesting modality splitting...")
    modality_features = model.split_modalities(x_sample)
    for i, (name, feat) in enumerate(zip(modality_names, modality_features)):
        print(f"  {name}: {feat.shape}")

    # Test modality importance
    print("\nTesting modality importance...")
    importance = model.get_modality_importance(x_sample)
    print(f"Importance shape: {importance.shape}")
    print(f"Average importance per modality:")
    for name, imp in zip(modality_names, importance.mean(axis=0)):
        print(f"  {name}: {imp:.4f}")

    # Test modality embeddings
    print("\nTesting modality embeddings...")
    embeddings = model.get_modality_embeddings(x_sample)
    print(f"Embeddings shape: {embeddings.shape}")

    # Test predictions
    print("\nTesting predictions...")
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    print(f"Predictions: {preds.cpu().numpy()}")
    print(f"Probabilities (first sample): {probs[0].cpu().numpy()}")

    print("\n✓ Multimodal Transformer test complete!")
