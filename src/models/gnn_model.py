"""
Graph Neural Network for Depression Prediction

Uses Graph Attention Network (GAT) to leverage user similarity structure.
Key features:
- Semi-supervised learning (useful for small labeled dataset)
- Attention mechanism for interpretability
- Node embeddings visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from typing import Tuple, Optional
import numpy as np

from src.models.pytorch_base import BaseDeepModel


class DepGraphNet(BaseDeepModel):
    """Graph Attention Network for depression prediction."""

    def __init__(
        self,
        in_channels: int = 52,
        hidden_channels: int = 16,
        num_classes: int = 2,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        device: Optional[str] = None
    ):
        """
        Initialize GNN model.

        Args:
            in_channels: Input feature dimension (52 features)
            hidden_channels: Hidden layer dimension
            num_classes: Number of output classes (2 for binary)
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout rate
            device: Device ('mps', 'cuda', 'cpu')
        """
        super().__init__(device)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout

        # Build GAT layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=heads,
                dropout=dropout,
                concat=True
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    concat=True
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        # Last layer (average attention heads)
        if num_layers > 1:
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=1,
                    dropout=dropout,
                    concat=False
                )
            )
        else:
            # Single layer case
            self.convs[0] = GATConv(
                in_channels,
                hidden_channels,
                heads=1,
                dropout=dropout,
                concat=False
            )
            self.bns[0] = nn.BatchNorm1d(hidden_channels)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )

        # Store attention weights for visualization
        self.last_attention_weights = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            return_embeddings: Whether to return node embeddings

        Returns:
            Tuple of (logits, embeddings) if return_embeddings=True
            Otherwise just logits
        """
        # Graph convolutions
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if i < len(self.bns):
                x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Last layer (with attention weights)
        x, (edge_index_att, attention_weights) = self.convs[-1](
            x, edge_index, return_attention_weights=True
        )

        # Store attention weights
        self.last_attention_weights = attention_weights

        # Node embeddings (for visualization)
        embeddings = x

        # Classification
        logits = self.classifier(x)

        if return_embeddings:
            return logits, embeddings
        else:
            return logits

    def get_attention_weights(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get last layer attention weights.

        Returns:
            Tuple of (edge_index, attention_weights) or None
        """
        return self.last_attention_weights

    def get_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> np.ndarray:
        """
        Get node embeddings for visualization.

        Args:
            x: Node features
            edge_index: Edge connectivity

        Returns:
            Node embeddings as numpy array
        """
        self.eval()
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        with torch.no_grad():
            _, embeddings = self(x, edge_index, return_embeddings=True)

        return embeddings.cpu().numpy()

    def training_step(self, batch, criterion=None):
        """Override training step for graph data."""
        # Unpack graph batch
        x, edge_index, y, mask = batch

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        y = y.to(self.device)
        mask = mask.to(self.device)

        # Forward pass (only on training nodes)
        logits = self(x, edge_index)

        # Compute loss only on masked nodes
        loss = criterion(logits[mask], y[mask])

        return loss

    def validation_step(self, batch, criterion=None):
        """Override validation step for graph data."""
        x, edge_index, y, mask = batch

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        y = y.to(self.device)
        mask = mask.to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self(x, edge_index)
            loss = criterion(logits[mask], y[mask])

            # Get predictions
            preds = torch.argmax(logits[mask], dim=-1)

        return loss.item(), preds, y[mask]

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            x: Node features
            edge_index: Edge connectivity
            mask: Optional mask for subset of nodes

        Returns:
            Class probabilities [num_nodes, num_classes]
        """
        self.eval()
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        with torch.no_grad():
            logits = self(x, edge_index)
            probs = F.softmax(logits, dim=-1)

        if mask is not None:
            probs = probs[mask]

        return probs.cpu().numpy()


# Example usage
if __name__ == '__main__':
    import sys
    sys.path.append('/Users/arnold/Documents/Dev/multimodal-depression-detection')
    from src.utils.data_loader import load_features_labels, load_config
    from src.features.graph_builder import UserSimilarityGraph

    print("=== Testing GNN Model ===\n")

    # Load data
    X, y, _ = load_features_labels()
    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Build graph
    config = load_config()
    gnn_config = config['gnn']

    graph_builder = UserSimilarityGraph(
        metric=gnn_config['similarity_metric']
    )
    edge_index, edge_weights = graph_builder.build_knn_graph(
        X, k=gnn_config['k_neighbors']
    )

    print(f"\nGraph: {X.shape[0]} nodes, {edge_index.shape[1]} edges")

    # Create model
    model = DepGraphNet(
        in_channels=X.shape[1],
        hidden_channels=gnn_config['hidden_channels'],
        num_layers=gnn_config['num_layers'],
        heads=gnn_config.get('heads', 4),
        dropout=gnn_config['dropout']
    )

    print(f"\nModel device: {model.device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    print("\nTesting forward pass...")
    x_tensor = torch.FloatTensor(X).to(model.device)
    edge_index_tensor = edge_index.to(model.device)

    logits, embeddings = model(
        x_tensor, edge_index_tensor, return_embeddings=True
    )

    print(f"Input shape: {x_tensor.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Embeddings shape: {embeddings.shape}")

    # Test predictions
    probs = model.predict_proba(x_tensor, edge_index_tensor)
    print(f"\nPrediction probabilities shape: {probs.shape}")
    print(f"Sample probabilities (first 3 nodes):")
    print(probs[:3])

    print("\n✓ GNN model test complete!")
