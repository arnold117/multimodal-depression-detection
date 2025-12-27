"""
PyTorch Utilities

Helper functions for PyTorch training:
- Device selection (MPS, CUDA, CPU)
- Dataset wrapper for tabular data
- DataLoader creation
- Reproducibility (seed setting)
- Model parameter counting
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import Tuple, Optional


def get_device(force_device: Optional[str] = None) -> torch.device:
    """
    Get the best available device (MPS > CUDA > CPU).

    Args:
        force_device: Force specific device ('mps', 'cuda', 'cpu')

    Returns:
        torch.device object
    """
    if force_device:
        device = torch.device(force_device)
        print(f"Using forced device: {device}")
        return device

    # Auto-detect best device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✓ Using Apple Silicon MPS acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"Using CPU")

    return device


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"✓ Random seed set to {seed}")


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular data (features + labels)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    batch_size: int = 16,
    shuffle_train: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create PyTorch DataLoaders for training and validation.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        batch_size: Batch size
        shuffle_train: Whether to shuffle training data
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = TabularDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers
    )

    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TabularDataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    print(f"✓ DataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    if val_loader:
        print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")

    return train_loader, val_loader


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a PyTorch model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total:,} trainable parameters")
    return total


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 20, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy/AUC
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        # Check if improved
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n⚠️  Early stopping triggered (patience={self.patience})")
                return True

        return False


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str
):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> int:
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint
        model: PyTorch model
        optimizer: Optimizer (optional)

    Returns:
        Epoch number from checkpoint
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"✓ Loaded checkpoint from epoch {epoch} (loss={loss:.4f})")
    return epoch


# Example usage
if __name__ == '__main__':
    import sys
    sys.path.append('/Users/arnold/Documents/Dev/multimodal-depression-detection')
    from src.utils.data_loader import load_features_labels

    print("=== Testing PyTorch Utilities ===\n")

    # Set seed
    set_seed(42)

    # Get device
    device = get_device()

    # Load data
    X, y, _ = load_features_labels()

    # Create dataloaders
    train_loader, _ = get_dataloaders(X, y, batch_size=16)

    # Test iteration
    print("\nTesting DataLoader iteration:")
    for batch_idx, (features, labels) in enumerate(train_loader):
        print(f"  Batch {batch_idx}: features={features.shape}, labels={labels.shape}")
        if batch_idx >= 2:
            break

    # Test simple model
    class SimpleModel(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.fc = nn.Linear(input_dim, 2)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel(X.shape[1])
    count_parameters(model)

    # Test early stopping
    print("\nTesting Early Stopping:")
    early_stop = EarlyStopping(patience=3, mode='min')

    for epoch in range(10):
        val_loss = 1.0 - epoch * 0.05  # Simulated improving loss
        if epoch > 5:
            val_loss = 0.7  # Plateau

        print(f"  Epoch {epoch}: val_loss={val_loss:.3f}")
        if early_stop(val_loss):
            break

    print("\n✓ PyTorch utilities test complete!")
