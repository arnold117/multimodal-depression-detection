"""
Graph Neural Network Encoders for Mental Health Prediction.

Implements:
1. TemporalGAT - Time-aware Graph Attention Network
2. MultiModalGNN - Cross-modal fusion with attention
3. UserGraphEncoder - User-level aggregation

Design choices for small sample (n≈50):
- Strong regularization (dropout=0.3)
- Shallow networks (2 layers)
- Attention for interpretability
- Multi-task learning for regularization
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Geometric imports
try:
    from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool
    from torch_geometric.data import Data, Batch

    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


class FourierTimeEncoder(nn.Module):
    """
    Fourier-based time encoding for continuous timestamps.

    Encodes time as sum of sinusoids at different frequencies,
    similar to positional encoding in Transformers.

    Reference: Xu et al. "Inductive Representation Learning on Temporal Graphs" (ICLR 2020)
    """

    def __init__(self, dim: int = 16):
        """
        Args:
            dim: Output dimension (should be even)
        """
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)

        # Initialize with different frequencies
        with torch.no_grad():
            freq = torch.exp(
                torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)
            )
            self.w.weight[::2, 0] = freq
            self.w.weight[1::2, 0] = freq

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values (batch_size,) or (batch_size, 1)

        Returns:
            Time encoding (batch_size, dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Linear projection
        t_proj = self.w(t.float())

        # Apply sin/cos
        t_proj[:, ::2] = torch.sin(t_proj[:, ::2])
        t_proj[:, 1::2] = torch.cos(t_proj[:, 1::2])

        return t_proj


class TemporalGAT(nn.Module):
    """
    Temporal Graph Attention Network.

    Features:
    1. Multi-head attention for interpretable edge weights
    2. Time encoding to capture temporal dynamics
    3. Residual connections to prevent over-smoothing
    4. Strong dropout for small sample regularization

    Suitable for graphs where edges have timestamps (e.g., user visits location at time t).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        heads: int = 4,
        dropout: float = 0.3,
        time_dim: int = 16,
        use_time_encoding: bool = True,
    ):
        """
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output embedding dimension
            heads: Number of attention heads
            dropout: Dropout rate
            time_dim: Time encoding dimension
            use_time_encoding: Whether to use temporal encoding
        """
        super().__init__()

        if not PYG_AVAILABLE:
            raise ImportError("torch_geometric required for TemporalGAT")

        self.use_time_encoding = use_time_encoding
        self.time_dim = time_dim if use_time_encoding else 0

        # Time encoder
        if use_time_encoding:
            self.time_encoder = FourierTimeEncoder(dim=time_dim)

        # Input projection (handles time encoding concatenation)
        input_dim = in_channels + self.time_dim
        self.input_proj = nn.Linear(input_dim, in_channels)

        # GAT layers
        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,  # Concatenate heads
        )

        self.gat2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            dropout=dropout,
            concat=False,
        )

        # Layer norm for stability
        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        self.norm2 = nn.LayerNorm(out_channels)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Store attention weights for interpretability
        self._attention_weights = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_time: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge connectivity (2, num_edges)
            edge_time: Edge timestamps (num_edges,) - optional
            return_attention: Whether to return attention weights

        Returns:
            Node embeddings (num_nodes, out_channels)
            Optionally: attention weights
        """
        # Time encoding
        if self.use_time_encoding and edge_time is not None:
            # Aggregate time info to nodes (mean of incident edge times)
            # This is a simplification; more sophisticated methods exist
            time_enc = self.time_encoder(edge_time)

            # Scatter time encoding to source nodes
            src_nodes = edge_index[0]
            node_time = torch.zeros(x.size(0), self.time_dim, device=x.device)
            node_time.scatter_reduce_(
                0,
                src_nodes.unsqueeze(-1).expand(-1, self.time_dim),
                time_enc,
                reduce="mean",
            )

            # Concatenate and project
            x = torch.cat([x, node_time], dim=-1)
            x = self.input_proj(x)

        # First GAT layer
        h, attention1 = self.gat1(x, edge_index, return_attention_weights=True)
        h = F.elu(h)
        h = self.norm1(h)
        h = self.dropout(h)

        # Second GAT layer
        h, attention2 = self.gat2(h, edge_index, return_attention_weights=True)
        h = self.norm2(h)

        # Store attention for interpretability
        self._attention_weights = (attention1, attention2)

        if return_attention:
            return h, self._attention_weights

        return h

    def get_attention_weights(self) -> Optional[Tuple]:
        """Get stored attention weights from last forward pass."""
        return self._attention_weights


class MultiModalFusion(nn.Module):
    """
    Cross-modal attention fusion for multiple modality embeddings.

    Given embeddings from different modalities (GPS, App, Communication, Activity),
    learns to weight and combine them using attention mechanism.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        n_modalities: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim: Dimension of each modality embedding
            n_modalities: Number of modalities to fuse
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.n_modalities = n_modalities

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Modality embeddings (learnable)
        self.modality_embed = nn.Embedding(n_modalities, embed_dim)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        modality_embeddings: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse multiple modality embeddings.

        Args:
            modality_embeddings: (batch_size, n_modalities, embed_dim)
            modality_mask: (batch_size, n_modalities) - True for missing modalities

        Returns:
            fused: (batch_size, embed_dim) - Fused representation
            attention_weights: (batch_size, n_modalities) - Modality importance
        """
        batch_size = modality_embeddings.size(0)

        # Add modality type embeddings
        modality_ids = torch.arange(self.n_modalities, device=modality_embeddings.device)
        modality_ids = modality_ids.unsqueeze(0).expand(batch_size, -1)
        modality_type_embed = self.modality_embed(modality_ids)

        x = modality_embeddings + modality_type_embed

        # Self-attention across modalities
        # Query: mean of all modalities; Key/Value: individual modalities
        query = x.mean(dim=1, keepdim=True)  # (batch, 1, embed_dim)

        attn_output, attn_weights = self.cross_attention(
            query, x, x,
            key_padding_mask=modality_mask,
        )

        # Output
        fused = self.output_proj(attn_output.squeeze(1))
        fused = self.norm(fused)

        return fused, attn_weights.squeeze(1)


class MentalHealthGNN(nn.Module):
    """
    Main GNN model for mental health prediction.

    Architecture:
    1. Modality-specific TemporalGAT encoders
    2. Cross-modal attention fusion
    3. Multi-task prediction heads (PHQ-9, Big Five, GPA)

    Designed for small sample learning with strong regularization.
    """

    def __init__(
        self,
        node_features: int,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        n_modalities: int = 4,
        n_heads: int = 4,
        dropout: float = 0.3,
        n_phq9_classes: int = 5,  # Severity levels
        n_bigfive_dims: int = 5,
    ):
        """
        Args:
            node_features: Input node feature dimension
            hidden_dim: Hidden layer dimension
            embed_dim: Output embedding dimension
            n_modalities: Number of sensor modalities
            n_heads: Attention heads
            dropout: Dropout rate
            n_phq9_classes: PHQ-9 severity classes
            n_bigfive_dims: Big Five personality dimensions
        """
        super().__init__()

        self.n_modalities = n_modalities

        # Modality encoders
        self.encoders = nn.ModuleDict({
            "gps": TemporalGAT(node_features, hidden_dim, embed_dim, n_heads, dropout),
            "activity": TemporalGAT(node_features, hidden_dim, embed_dim, n_heads, dropout),
            "phone": TemporalGAT(node_features, hidden_dim, embed_dim, n_heads, dropout),
            "social": TemporalGAT(node_features, hidden_dim, embed_dim, n_heads, dropout),
        })

        # Cross-modal fusion
        self.fusion = MultiModalFusion(embed_dim, n_modalities, n_heads, dropout)

        # Prediction heads
        self.phq9_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),  # Regression
        )

        self.phq9_severity_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, n_phq9_classes),  # Classification
        )

        self.bigfive_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, n_bigfive_dims),  # 5 dimensions
        )

        self.gpa_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),  # Regression
        )

        # Uncertainty weights for multi-task learning
        self.log_vars = nn.Parameter(torch.zeros(4))

    def forward(
        self,
        data_dict: dict,
        return_attention: bool = False,
    ) -> dict:
        """
        Forward pass.

        Args:
            data_dict: Dictionary containing modality-specific graph data
                - 'gps': Data object with x, edge_index, edge_time
                - 'activity': ...
                - 'phone': ...
                - 'social': ...
                - 'batch': Batch indices for pooling
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with predictions and optionally attention weights
        """
        modality_embeds = []
        attention_weights = {}

        # Encode each modality
        for modality, encoder in self.encoders.items():
            if modality in data_dict:
                mod_data = data_dict[modality]
                if return_attention:
                    h, attn = encoder(
                        mod_data.x,
                        mod_data.edge_index,
                        getattr(mod_data, "edge_time", None),
                        return_attention=True,
                    )
                    attention_weights[modality] = attn
                else:
                    h = encoder(
                        mod_data.x,
                        mod_data.edge_index,
                        getattr(mod_data, "edge_time", None),
                    )

                # Pool to user level (only if batch indices are explicitly provided)
                # Note: PyG Data objects return None for undefined attributes via __getattr__,
                # so we must check for None, not just hasattr()
                batch = getattr(mod_data, "batch", None)
                if batch is not None:
                    h = global_mean_pool(h, batch)

                modality_embeds.append(h)
            else:
                # Missing modality - use zeros with correct device
                batch_size = data_dict.get("batch_size", 1)
                # Get device from existing modality data or default to CPU
                device = next(
                    (data_dict[m].x.device for m in self.encoders.keys() if m in data_dict),
                    torch.device("cpu")
                )
                modality_embeds.append(
                    torch.zeros(batch_size, self.encoders[modality].gat2.out_channels, device=device)
                )

        # Stack modalities: (batch, n_modalities, embed_dim)
        modality_stack = torch.stack(modality_embeds, dim=1)

        # Cross-modal fusion
        fused, modal_attention = self.fusion(modality_stack)

        # Predictions
        outputs = {
            "phq9_score": self.phq9_head(fused).squeeze(-1),
            "phq9_severity": self.phq9_severity_head(fused),
            "bigfive": self.bigfive_head(fused),
            "gpa": self.gpa_head(fused).squeeze(-1),
            "embedding": fused,
            "modality_attention": modal_attention,
        }

        if return_attention:
            outputs["graph_attention"] = attention_weights

        return outputs

    def compute_loss(
        self,
        outputs: dict,
        targets: dict,
        task_weights: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute multi-task loss with uncertainty weighting.

        Args:
            outputs: Model predictions
            targets: Ground truth values
            task_weights: Optional manual task weights

        Returns:
            total_loss: Combined loss
            loss_dict: Individual task losses
        """
        losses = {}

        # PHQ-9 regression (MSE)
        if "phq9_score" in targets:
            losses["phq9_score"] = F.mse_loss(
                outputs["phq9_score"], targets["phq9_score"]
            )

        # PHQ-9 severity classification (CE)
        if "phq9_severity" in targets:
            losses["phq9_severity"] = F.cross_entropy(
                outputs["phq9_severity"], targets["phq9_severity"]
            )

        # Big Five regression (MSE)
        if "bigfive" in targets:
            losses["bigfive"] = F.mse_loss(outputs["bigfive"], targets["bigfive"])

        # GPA regression (MSE)
        if "gpa" in targets:
            losses["gpa"] = F.mse_loss(outputs["gpa"], targets["gpa"])

        # Uncertainty-weighted combination
        # Based on Kendall et al. "Multi-Task Learning Using Uncertainty"
        total_loss = 0
        for i, (task, loss) in enumerate(losses.items()):
            if task_weights is not None:
                total_loss += task_weights.get(task, 1.0) * loss
            else:
                # Learned uncertainty weighting
                precision = torch.exp(-self.log_vars[i])
                total_loss += precision * loss + self.log_vars[i]

        return total_loss, losses
