#!/usr/bin/env python3
"""
Step 5: GNN Model Training

Trains a Graph Attention Network for mental health prediction.
Uses cross-validation for robust evaluation.

Supports: CUDA, MPS (Apple Silicon), CPU (auto-detected)

Usage:
    python scripts/05_train_gnn.py
    python scripts/05_train_gnn.py --device cuda
    python scripts/05_train_gnn.py --device mps

Outputs:
    - outputs/models/gnn_best.pt
    - outputs/models/cv_results.json
    - outputs/models/training_history.png
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gnn.encoders import MentalHealthGNN


def get_device(requested: str = "auto") -> torch.device:
    """
    Auto-detect best available device.

    Priority: CUDA > MPS > CPU

    Args:
        requested: "auto", "cuda", "mps", or "cpu"

    Returns:
        torch.device
    """
    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
    elif requested == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif requested == "mps":
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            logger.warning("MPS requested but not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device


def setup_logging(output_dir: Path) -> None:
    """Configure logging."""
    log_file = output_dir / "logs" / f"train_gnn_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_file, rotation="10 MB", level="DEBUG")


def load_data(processed_dir: Path, kg_path: Path) -> Tuple:
    """Load all required data including multi-modal daily features."""
    # Load user-level features
    features_df = pl.read_parquet(processed_dir / "sensor_features.parquet")
    survey_df = pl.read_parquet(processed_dir / "survey_scores.parquet")

    # Load daily features per modality
    daily_features = {}
    daily_dir = processed_dir / "daily_features"
    if daily_dir.exists():
        for modality in ["gps", "phone", "activity", "conv"]:
            path = daily_dir / f"{modality}_daily.parquet"
            if path.exists():
                daily_features[modality] = pl.read_parquet(path)
                logger.info(f"Loaded {modality} daily: {daily_features[modality].shape}")

    # Load temporal features (sliding window)
    temporal_df = None
    temporal_path = processed_dir / "temporal_features.parquet"
    if temporal_path.exists():
        temporal_df = pl.read_parquet(temporal_path)
        logger.info(f"Loaded temporal features: {temporal_df.shape}")

    # Load knowledge graph
    kg_data = None
    if kg_path.exists():
        with open(kg_path) as f:
            kg_data = json.load(f)
        logger.info(f"Loaded KG: {len(kg_data['nodes'])} nodes, {len(kg_data['edges'])} edges")

    logger.info(f"Loaded user features: {features_df.shape}")
    logger.info(f"Loaded surveys: {survey_df.shape}")

    return features_df, survey_df, kg_data, daily_features, temporal_df


def prepare_temporal_dataset(
    temporal_df: pl.DataFrame,
    survey_df: pl.DataFrame
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, List[str]], List[str]]:
    """
    Prepare multi-modal feature matrices from temporal (sliding window) features.

    Temporal features are aggregated per user (mean over study period).

    Returns:
        X_modalities: Dict of modality -> feature matrix
        X_combined: Combined feature matrix for graph building
        y: Labels (PHQ-9 scores)
        feature_cols: Dict of modality -> feature column names
        valid_user_ids: List of valid user IDs
    """
    # Modality prefixes in temporal features
    # Note: The preprocessing uses "conversation" prefix, we map it to "social" for the model
    modality_prefixes = {
        "gps": "gps_",
        "phone": "phone_",
        "activity": "activity_",
        "social": "conversation_"  # Model expects "social", features use "conversation_"
    }

    # Get numeric feature columns
    exclude_cols = ["user_id", "date"]
    numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32]

    all_feature_cols = [
        c for c in temporal_df.columns
        if c not in exclude_cols and temporal_df[c].dtype in numeric_types
    ]

    if not all_feature_cols:
        logger.warning("No numeric temporal features found")
        return {}, np.array([]), np.array([]), {}, []

    # Aggregate per user (mean over study period)
    user_temporal = temporal_df.group_by("user_id").agg([
        pl.col(c).mean().alias(c) for c in all_feature_cols
    ])
    user_temporal = user_temporal.filter(pl.col("user_id").is_not_null())

    user_ids = user_temporal["user_id"].to_list()

    # Separate features by modality
    X_modalities = {}
    feature_cols = {}

    for modality, prefix in modality_prefixes.items():
        mod_cols = [c for c in all_feature_cols if c.startswith(prefix)]
        if mod_cols:
            feature_cols[modality] = mod_cols
            X_modalities[modality] = user_temporal.select(mod_cols).to_numpy().astype(np.float64)
            # Replace NaN with 0
            X_modalities[modality] = np.nan_to_num(X_modalities[modality], nan=0.0)
            logger.info(f"  {modality}: {len(mod_cols)} temporal features")

    # Combined features for graph building
    X_combined = user_temporal.select(all_feature_cols).to_numpy().astype(np.float64)
    X_combined = np.nan_to_num(X_combined, nan=0.0)

    # Get labels (PHQ-9 total score)
    pre_df = survey_df.filter(pl.col("type") == "pre")

    y = []
    valid_indices = []
    for i, uid in enumerate(user_ids):
        user_survey = pre_df.filter(pl.col("uid") == uid)
        if len(user_survey) > 0 and "total_score" in user_survey.columns:
            score = user_survey["total_score"].to_list()[0]
            if score is not None and not np.isnan(score):
                y.append(score)
                valid_indices.append(i)

    # Filter to valid samples
    for modality in X_modalities:
        X_modalities[modality] = X_modalities[modality][valid_indices]
    X_combined = X_combined[valid_indices]
    y = np.array(y)
    valid_user_ids = [user_ids[i] for i in valid_indices]

    logger.info(f"Prepared temporal dataset: {len(valid_user_ids)} users, y range: [{y.min():.1f}, {y.max():.1f}]")
    for modality, X in X_modalities.items():
        logger.info(f"  {modality}: shape {X.shape}")

    return X_modalities, X_combined, y, feature_cols, valid_user_ids


def prepare_dataset(
    features_df: pl.DataFrame,
    survey_df: pl.DataFrame,
    daily_features: Dict = None
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, List[str]], List[str]]:
    """
    Prepare multi-modal feature matrices and labels from user-level features.

    Returns:
        X_modalities: Dict of modality -> feature matrix
        X_combined: Combined feature matrix for graph building
        y: Labels (PHQ-9 scores)
        feature_cols: Dict of modality -> feature column names
        valid_user_ids: List of valid user IDs
    """
    user_ids = features_df["user_id"].to_list()

    # Define modality-specific feature patterns
    modality_patterns = {
        "gps": ["location_entropy", "home_stay_ratio", "n_locations", "total_distance", "radius_of_gyration"],
        "phone": ["unlock_count", "screen_time", "session_duration", "night_usage_ratio"],
        "activity": ["still_ratio", "walking_ratio", "running_ratio", "transitions"],
        "social": ["conv_duration", "conv_count", "audio"]
    }

    # Separate features by modality from user-level aggregated features
    X_modalities = {}
    feature_cols = {}

    exclude_cols = ["user_id", "user_id_right", "gps_valid", "activity_valid", "phone_valid", "conv_valid"]
    numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32]

    for modality, patterns in modality_patterns.items():
        # Find columns matching this modality's patterns
        mod_cols = []
        for c in features_df.columns:
            if c in exclude_cols:
                continue
            if features_df[c].dtype not in numeric_types:
                continue
            # Check if column matches any pattern for this modality
            if any(p in c.lower() for p in patterns):
                mod_cols.append(c)

        if mod_cols:
            feature_cols[modality] = mod_cols
            X_modalities[modality] = features_df.select(mod_cols).to_numpy()
            logger.info(f"  {modality}: {len(mod_cols)} features")

    # Also prepare combined features for graph building
    all_cols = []
    for cols in feature_cols.values():
        all_cols.extend(cols)
    X_combined = features_df.select(all_cols).to_numpy() if all_cols else features_df.select(
        [c for c in features_df.columns if c not in exclude_cols and features_df[c].dtype in numeric_types]
    ).to_numpy()

    # Get labels (PHQ-9 total score)
    pre_df = survey_df.filter(pl.col("type") == "pre")

    y = []
    valid_indices = []
    for i, uid in enumerate(user_ids):
        user_survey = pre_df.filter(pl.col("uid") == uid)
        if len(user_survey) > 0 and "total_score" in user_survey.columns:
            score = user_survey["total_score"].to_list()[0]
            if score is not None and not np.isnan(score):
                y.append(score)
                valid_indices.append(i)

    # Filter to valid samples
    for modality in X_modalities:
        X_modalities[modality] = X_modalities[modality][valid_indices]
    X_combined = X_combined[valid_indices]
    y = np.array(y)
    valid_user_ids = [user_ids[i] for i in valid_indices]

    logger.info(f"Prepared dataset: {len(valid_user_ids)} users, y range: [{y.min():.1f}, {y.max():.1f}]")
    for modality, X in X_modalities.items():
        logger.info(f"  {modality}: shape {X.shape}")

    return X_modalities, X_combined, y, feature_cols, valid_user_ids


def build_user_graph(
    X: np.ndarray,
    user_ids: List[str],
    k_neighbors: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build user similarity graph using k-NN."""
    from sklearn.neighbors import NearestNeighbors

    # Standardize features
    X_scaled = StandardScaler().fit_transform(X)

    # Find k nearest neighbors
    k = min(k_neighbors, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)

    # Build edge index (excluding self-loops)
    edge_list = []
    edge_weights = []
    for i in range(len(X)):
        for j, dist in zip(indices[i][1:], distances[i][1:]):  # Skip self
            edge_list.append([i, j])
            edge_weights.append(1 - dist)  # Convert distance to similarity

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)

    return edge_index, edge_weight


class GNNTrainer:
    """Trainer for mental health GNN with multi-modal support."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 100,
        patience: int = 20
    ):
        self.model = model.to(device)
        self.device = device
        self.n_epochs = n_epochs
        self.patience = patience

        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=n_epochs)

        self.history = {"train_loss": [], "val_loss": [], "val_mae": []}

    def _create_data_dict(
        self,
        X_modalities: Dict[str, torch.Tensor],
        edge_index: torch.Tensor
    ) -> Dict:
        """
        Create multi-modal data dict for MentalHealthGNN.

        Provides all 4 modalities (gps, phone, activity, social).
        Missing modalities get zero tensors with correct shape and device.
        """
        # Get reference tensor for shape and device
        ref_tensor = list(X_modalities.values())[0]
        batch_size = ref_tensor.shape[0]
        n_features = ref_tensor.shape[1]
        device = ref_tensor.device

        data_dict = {"batch_size": batch_size}

        # All expected modalities
        all_modalities = ["gps", "phone", "activity", "social"]

        for modality in all_modalities:
            if modality in X_modalities:
                data_dict[modality] = Data(
                    x=X_modalities[modality],
                    edge_index=edge_index
                )
            else:
                # Create zero tensor for missing modality
                zero_x = torch.zeros(batch_size, n_features, device=device)
                data_dict[modality] = Data(
                    x=zero_x,
                    edge_index=edge_index
                )

        return data_dict

    def train_epoch(
        self,
        X_modalities: Dict[str, torch.Tensor],
        y: torch.Tensor,
        edge_index: torch.Tensor,
        train_mask: torch.Tensor
    ) -> float:
        """Train one epoch with multi-modal input."""
        self.model.train()
        self.optimizer.zero_grad()

        # Create multi-modal data dict
        data_dict = self._create_data_dict(X_modalities, edge_index)

        # Forward pass
        outputs = self.model(data_dict)
        predictions = outputs["phq9_score"].view(-1)

        # Loss only on training samples
        loss = F.mse_loss(predictions[train_mask], y[train_mask])

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def evaluate(
        self,
        X_modalities: Dict[str, torch.Tensor],
        y: torch.Tensor,
        edge_index: torch.Tensor,
        val_mask: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate on validation set with multi-modal input."""
        self.model.eval()
        with torch.no_grad():
            data_dict = self._create_data_dict(X_modalities, edge_index)

            outputs = self.model(data_dict)
            predictions = outputs["phq9_score"].view(-1)

            val_pred = predictions[val_mask].cpu().numpy()
            val_true = y[val_mask].cpu().numpy()

            mse = mean_squared_error(val_true, val_pred)
            mae = mean_absolute_error(val_true, val_pred)
            r2 = r2_score(val_true, val_pred)

            # Binary classification metrics (moderate depression: score >= 10)
            pred_binary = (val_pred >= 10).astype(int)
            true_binary = (val_true >= 10).astype(int)

            if len(np.unique(true_binary)) > 1:
                acc = accuracy_score(true_binary, pred_binary)
                f1 = f1_score(true_binary, pred_binary)
                try:
                    auc = roc_auc_score(true_binary, val_pred)
                except:
                    auc = 0.5
            else:
                acc = f1 = auc = 0.0

            # Get modality attention weights for interpretability
            modality_attention = outputs.get("modality_attention", None)

        return {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "accuracy": acc,
            "f1": f1,
            "auc": auc,
            "modality_attention": modality_attention
        }

    def fit(
        self,
        X_modalities: Dict[str, torch.Tensor],
        y: torch.Tensor,
        edge_index: torch.Tensor,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor
    ) -> Dict[str, List[float]]:
        """Full training loop with early stopping."""
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.n_epochs):
            train_loss = self.train_epoch(X_modalities, y, edge_index, train_mask)
            val_metrics = self.evaluate(X_modalities, y, edge_index, val_mask)

            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["mse"])
            self.history["val_mae"].append(val_metrics["mae"])

            if val_metrics["mse"] < best_val_loss:
                best_val_loss = val_metrics["mse"]
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.n_epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val MSE: {val_metrics['mse']:.4f}, "
                    f"Val MAE: {val_metrics['mae']:.4f}"
                )

            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.history


def cross_validate(
    X_modalities: Dict[str, np.ndarray],
    X_combined: np.ndarray,
    y: np.ndarray,
    edge_index: torch.Tensor,
    n_folds: int = 5,
    device: torch.device = None,
    **model_kwargs
) -> Dict:
    """Run cross-validation with multi-modal input."""
    if device is None:
        device = torch.device("cpu")

    # Create binary labels for stratification
    y_binary = (y >= 10).astype(int)

    # Handle case where one class has very few samples
    min_class_count = min(np.sum(y_binary == 0), np.sum(y_binary == 1))
    actual_folds = min(n_folds, min_class_count)

    if actual_folds < n_folds:
        logger.warning(f"Reducing folds from {n_folds} to {actual_folds} due to class imbalance")

    if actual_folds < 2:
        logger.warning("Not enough samples for cross-validation, using single split")
        actual_folds = 2
        # Use simple random split instead
        indices = np.random.permutation(len(y))
        split_idx = len(y) // 2
        folds = [(indices[:split_idx], indices[split_idx:])]
    else:
        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        folds = list(skf.split(X_combined, y_binary))

    # Standardize features per modality
    scalers = {}
    X_scaled_modalities = {}
    for modality, X in X_modalities.items():
        scalers[modality] = StandardScaler()
        X_scaled_modalities[modality] = scalers[modality].fit_transform(X)

    # Convert to tensors and move to device
    X_tensors = {
        modality: torch.tensor(X_scaled, dtype=torch.float32).to(device)
        for modality, X_scaled in X_scaled_modalities.items()
    }
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    edge_index = edge_index.to(device)

    # Determine node_features for model (use max across modalities for flexibility)
    max_features = max(X.shape[1] for X in X_modalities.values())

    fold_results = []
    best_fold_model = None
    best_fold_loss = float("inf")

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        logger.info(f"\n--- Fold {fold_idx + 1}/{len(folds)} ---")

        # Create masks
        train_mask = torch.zeros(len(y), dtype=torch.bool).to(device)
        val_mask = torch.zeros(len(y), dtype=torch.bool).to(device)
        train_mask[train_idx] = True
        val_mask[val_idx] = True

        # Initialize model with padded node_features
        # Each modality may have different feature dimensions, so we use max
        model = MentalHealthGNN(
            node_features=max_features,
            **model_kwargs
        )

        # Pad smaller modalities to match max_features
        X_tensors_padded = {}
        for modality, X_tensor in X_tensors.items():
            if X_tensor.shape[1] < max_features:
                padding = torch.zeros(X_tensor.shape[0], max_features - X_tensor.shape[1]).to(device)
                X_tensors_padded[modality] = torch.cat([X_tensor, padding], dim=1)
            else:
                X_tensors_padded[modality] = X_tensor

        # Train
        trainer = GNNTrainer(
            model,
            device=device,
            learning_rate=1e-3,
            weight_decay=1e-4,
            n_epochs=100,
            patience=20
        )

        trainer.fit(X_tensors_padded, y_tensor, edge_index, train_mask, val_mask)

        # Final evaluation
        metrics = trainer.evaluate(X_tensors_padded, y_tensor, edge_index, val_mask)

        # Remove non-serializable items for fold_results
        metrics_clean = {k: v for k, v in metrics.items() if k != "modality_attention"}
        fold_results.append(metrics_clean)

        logger.info(
            f"Fold {fold_idx + 1} Results - "
            f"MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, "
            f"R2: {metrics['r2']:.4f}, AUC: {metrics['auc']:.4f}"
        )

        # Track best model
        if metrics["mse"] < best_fold_loss:
            best_fold_loss = metrics["mse"]
            best_fold_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Aggregate results
    cv_results = {
        "n_folds": len(folds),
        "modalities": list(X_modalities.keys()),
        "metrics": {
            metric: {
                "mean": np.mean([r[metric] for r in fold_results]),
                "std": np.std([r[metric] for r in fold_results]),
                "values": [r[metric] for r in fold_results]
            }
            for metric in fold_results[0].keys()
        },
        "best_fold_mse": best_fold_loss
    }

    return cv_results, best_fold_model


def plot_training_history(history: Dict, output_path: Path) -> None:
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training History")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE curve
    axes[1].plot(history["val_mae"], label="Val MAE", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Validation MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved training history to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train GNN model")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/processed",
        help="Input directory with preprocessed data",
    )
    parser.add_argument(
        "--kg",
        type=str,
        default="outputs/graphs/knowledge_graph.json",
        help="Path to knowledge graph",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/models",
        help="Output directory",
    )
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument(
        "--use-temporal",
        action="store_true",
        help="Use temporal (sliding window) features instead of user-level aggregated features"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use: auto (default), cuda, mps, or cpu"
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    # Get device
    device = get_device(args.device)

    logger.info("=" * 60)
    logger.info("GNN Model Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")

    # Load data
    processed_dir = Path(args.input)
    kg_path = Path(args.kg)

    if not kg_path.exists():
        logger.warning(f"Knowledge graph not found: {kg_path}")
        logger.info("Training without KG (using feature-based graph only)")

    features_df, survey_df, _, daily_features, temporal_df = load_data(processed_dir, kg_path)

    # Prepare dataset (temporal or user-level)
    if args.use_temporal:
        logger.info("\n--- Preparing Temporal (Sliding Window) Dataset ---")
        if temporal_df is None:
            logger.error("Temporal features not found! Run 01_preprocess.py first.")
            return
        X_modalities, X_combined, y, feature_cols, user_ids = prepare_temporal_dataset(
            temporal_df, survey_df
        )
    else:
        logger.info("\n--- Preparing User-Level Dataset ---")
        X_modalities, X_combined, y, feature_cols, user_ids = prepare_dataset(
            features_df, survey_df, daily_features
        )

    if not X_modalities:
        logger.error("No modality features found!")
        return

    logger.info(f"Available modalities: {list(X_modalities.keys())}")
    logger.info(f"Feature mode: {'temporal' if args.use_temporal else 'user-level'}")

    # Build user graph using combined features
    logger.info(f"\n--- Building User Graph (k={args.k_neighbors}) ---")
    edge_index, edge_weight = build_user_graph(X_combined, user_ids, args.k_neighbors)
    logger.info(f"Graph edges: {edge_index.shape[1]}")

    # Model config
    # Note: n_modalities is fixed at 4 (gps, phone, activity, social) in MentalHealthGNN
    # Missing modalities are filled with zeros
    model_kwargs = {
        "hidden_dim": args.hidden_dim,
        "embed_dim": args.embed_dim,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
        "n_modalities": 4  # Fixed: gps, phone, activity, social
    }

    # Cross-validation
    logger.info(f"\n--- Running {args.n_folds}-Fold Cross-Validation ---")
    cv_results, best_model_state = cross_validate(
        X_modalities, X_combined, y, edge_index,
        n_folds=args.n_folds,
        device=device,
        **model_kwargs
    )

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Modalities used: {cv_results['modalities']}")

    for metric, values in cv_results["metrics"].items():
        logger.info(f"  {metric}: {values['mean']:.4f} +/- {values['std']:.4f}")

    # Save best model
    if best_model_state is not None:
        model_path = output_dir / "gnn_best.pt"
        torch.save({
            "model_state_dict": best_model_state,
            "model_config": model_kwargs,
            "feature_cols": feature_cols,
            "modalities": list(X_modalities.keys()),
            "n_features_per_modality": {k: v.shape[1] for k, v in X_modalities.items()},
            "device_trained_on": str(device)
        }, model_path)
        logger.info(f"\nSaved best model to {model_path}")

    # Save CV results
    results_path = output_dir / "cv_results.json"
    # Convert numpy types for JSON serialization
    cv_results_json = json.loads(
        json.dumps(cv_results, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    )
    with open(results_path, "w") as f:
        json.dump(cv_results_json, f, indent=2)
    logger.info(f"Saved CV results to {results_path}")

    logger.info("\nGNN training complete!")


if __name__ == "__main__":
    main()
