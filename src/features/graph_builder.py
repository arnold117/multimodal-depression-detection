"""
User Similarity Graph Builder

Constructs graphs for GNN based on:
- K-nearest neighbors (feature similarity)
- Fully connected graphs
- Threshold-based graphs
"""

import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.sparse import csr_matrix
from typing import Tuple, Optional
import networkx as nx
import matplotlib.pyplot as plt


class UserSimilarityGraph:
    """Build user similarity graphs for GNN."""

    def __init__(self, metric: str = 'cosine'):
        """
        Initialize graph builder.

        Args:
            metric: Distance metric ('cosine', 'euclidean', 'correlation')
        """
        self.metric = metric
        self.adjacency_matrix = None
        self.edge_index = None
        self.edge_weights = None

    def build_knn_graph(
        self,
        features: np.ndarray,
        k: int = 5,
        include_self: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build k-nearest neighbors graph.

        Args:
            features: Feature matrix (n_users, n_features)
            k: Number of neighbors
            include_self: Include self-loops

        Returns:
            Tuple of (edge_index, edge_weights)
        """
        print(f"\nBuilding {k}-NN graph with {self.metric} similarity...")

        # Build KNN graph
        A = kneighbors_graph(
            features,
            n_neighbors=k,
            metric=self.metric,
            mode='distance',
            include_self=include_self
        )

        # Convert distances to similarities
        A = A.toarray()

        if self.metric == 'cosine':
            # Cosine distance to similarity: sim = 1 - dist
            A = 1 - A
        elif self.metric == 'euclidean':
            # Euclidean distance to similarity: sim = 1 / (1 + dist)
            A = 1 / (1 + A)

        # Symmetrize (undirected graph)
        A = (A + A.T) / 2

        self.adjacency_matrix = A

        # Convert to edge_index format (PyTorch Geometric)
        edge_index, edge_weights = self._adj_to_edge_index(A)

        self.edge_index = edge_index
        self.edge_weights = edge_weights

        print(f"  Nodes: {features.shape[0]}")
        print(f"  Edges: {edge_index.shape[1]}")
        print(f"  Avg degree: {edge_index.shape[1] / features.shape[0]:.2f}")

        return edge_index, edge_weights

    def build_threshold_graph(
        self,
        features: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph with edges above similarity threshold.

        Args:
            features: Feature matrix
            threshold: Similarity threshold

        Returns:
            Tuple of (edge_index, edge_weights)
        """
        print(f"\nBuilding threshold graph (threshold={threshold})...")

        # Compute similarity matrix
        if self.metric == 'cosine':
            sim_matrix = cosine_similarity(features)
        elif self.metric == 'euclidean':
            dist = euclidean_distances(features)
            sim_matrix = 1 / (1 + dist)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # Threshold
        A = np.where(sim_matrix >= threshold, sim_matrix, 0)

        # Remove self-loops
        np.fill_diagonal(A, 0)

        self.adjacency_matrix = A

        # Convert to edge_index
        edge_index, edge_weights = self._adj_to_edge_index(A)

        self.edge_index = edge_index
        self.edge_weights = edge_weights

        print(f"  Nodes: {features.shape[0]}")
        print(f"  Edges: {edge_index.shape[1]}")
        print(f"  Density: {edge_index.shape[1] / (features.shape[0] ** 2):.4f}")

        return edge_index, edge_weights

    def build_fully_connected_graph(
        self,
        features: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build fully connected graph (complete graph).

        Args:
            features: Feature matrix

        Returns:
            Tuple of (edge_index, edge_weights)
        """
        print(f"\nBuilding fully connected graph...")

        # Compute similarity matrix
        if self.metric == 'cosine':
            sim_matrix = cosine_similarity(features)
        elif self.metric == 'euclidean':
            dist = euclidean_distances(features)
            sim_matrix = 1 / (1 + dist)

        # Remove self-loops
        np.fill_diagonal(sim_matrix, 0)

        self.adjacency_matrix = sim_matrix

        # Convert to edge_index
        edge_index, edge_weights = self._adj_to_edge_index(sim_matrix)

        self.edge_index = edge_index
        self.edge_weights = edge_weights

        print(f"  Nodes: {features.shape[0]}")
        print(f"  Edges: {edge_index.shape[1]}")

        return edge_index, edge_weights

    def _adj_to_edge_index(
        self,
        adj_matrix: np.ndarray,
        min_weight: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert adjacency matrix to edge_index format.

        Args:
            adj_matrix: Adjacency matrix
            min_weight: Minimum edge weight to include

        Returns:
            Tuple of (edge_index, edge_weights)
        """
        # Get non-zero edges
        rows, cols = np.where(adj_matrix > min_weight)
        weights = adj_matrix[rows, cols]

        # Create edge_index [2, num_edges]
        edge_index = torch.LongTensor(np.vstack([rows, cols]))

        # Edge weights
        edge_weights = torch.FloatTensor(weights)

        return edge_index, edge_weights

    def visualize_graph(
        self,
        labels: np.ndarray,
        save_path: Optional[str] = None,
        node_size: int = 300
    ):
        """
        Visualize graph with NetworkX.

        Args:
            labels: Node labels (for coloring)
            save_path: Path to save figure
            node_size: Node size
        """
        if self.adjacency_matrix is None:
            raise ValueError("Graph not built yet")

        # Create NetworkX graph
        G = nx.from_numpy_array(self.adjacency_matrix)

        # Layout
        pos = nx.spring_layout(G, seed=42)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))

        # Color nodes by label
        node_colors = ['red' if label == 1 else 'blue' for label in labels]

        # Draw
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors,
            node_size=node_size, alpha=0.7,
            ax=ax
        )

        nx.draw_networkx_edges(
            G, pos, alpha=0.2, width=0.5,
            ax=ax
        )

        # Labels
        nx.draw_networkx_labels(
            G, pos, font_size=8,
            ax=ax
        )

        ax.set_title('User Similarity Graph', fontsize=14, fontweight='bold')
        ax.axis('off')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Negative'),
            Patch(facecolor='red', label='Positive')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved graph visualization to {save_path}")

        return fig

    def get_graph_statistics(self) -> dict:
        """Get graph statistics."""
        if self.adjacency_matrix is None:
            raise ValueError("Graph not built yet")

        G = nx.from_numpy_array(self.adjacency_matrix)

        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_clustering': nx.average_clustering(G),
            'is_connected': nx.is_connected(G)
        }

        # Degree statistics
        degrees = [d for n, d in G.degree()]
        stats['avg_degree'] = np.mean(degrees)
        stats['max_degree'] = np.max(degrees)
        stats['min_degree'] = np.min(degrees)

        return stats


# Example usage
if __name__ == '__main__':
    import sys
    sys.path.append('/Users/arnold/Documents/Dev/multimodal-depression-detection')
    from src.utils.data_loader import load_features_labels, load_config

    print("=== Testing Graph Builder ===\n")

    # Load data
    X, y, _ = load_features_labels()

    # Load config
    config = load_config()
    gnn_config = config['gnn']

    # Create graph builder
    graph_builder = UserSimilarityGraph(
        metric=gnn_config['similarity_metric']
    )

    # Build KNN graph
    edge_index, edge_weights = graph_builder.build_knn_graph(
        X, k=gnn_config['k_neighbors']
    )

    print(f"\nEdge index shape: {edge_index.shape}")
    print(f"Edge weights shape: {edge_weights.shape}")
    print(f"Sample edges:")
    for i in range(min(5, edge_index.shape[1])):
        print(f"  {edge_index[0, i].item()} -> {edge_index[1, i].item()} "
              f"(weight={edge_weights[i].item():.3f})")

    # Graph statistics
    stats = graph_builder.get_graph_statistics()
    print(f"\nGraph statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Visualize (small sample)
    print(f"\nVisualizing graph...")
    graph_builder.visualize_graph(
        y,
        save_path='results/figures/user_similarity_graph_test.png',
        node_size=500
    )

    print("\n✓ Graph builder test complete!")
