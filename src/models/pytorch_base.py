"""
Base PyTorch Model

Abstract base class for all deep learning models with:
- MPS/CUDA/CPU device management
- Training/validation loops
- Early stopping
- Model checkpointing
- Reproducibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
import json


class BaseDeepModel(nn.Module):
    """Base class for all PyTorch models."""

    def __init__(self, device: Optional[str] = None):
        """
        Initialize base model.

        Args:
            device: Device to use ('mps', 'cuda', 'cpu', or None for auto)
        """
        super().__init__()

        # Device setup
        if device:
            self.device = torch.device(device)
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        self.history = {'train_loss': [], 'val_loss': []}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement forward()")

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        criterion: nn.Module
    ) -> float:
        """
        Single training step.

        Args:
            batch: Tuple of (features, labels)
            criterion: Loss function

        Returns:
            Loss value
        """
        features, labels = batch
        features = features.to(self.device)
        labels = labels.to(self.device)

        # Forward pass
        outputs = self(features)

        # Compute loss
        loss = criterion(outputs, labels)

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        criterion: nn.Module
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Single validation step.

        Args:
            batch: Tuple of (features, labels)
            criterion: Loss function

        Returns:
            Tuple of (loss, predictions, labels)
        """
        features, labels = batch
        features = features.to(self.device)
        labels = labels.to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self(features)
            loss = criterion(outputs, labels)

            # Get predictions
            if outputs.shape[-1] > 1:  # Multi-class
                preds = torch.argmax(outputs, dim=-1)
            else:  # Binary
                preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()

        return loss.item(), preds, labels

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        epochs: int = 100,
        patience: int = 20,
        verbose: bool = True,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            optimizer: Optimizer (defaults to Adam)
            criterion: Loss function (defaults to CrossEntropyLoss)
            epochs: Maximum number of epochs
            patience: Early stopping patience
            verbose: Print progress
            save_path: Path to save best model

        Returns:
            Training history
        """
        # Default optimizer and criterion
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.01)

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Move model to device
        self.to(self.device)

        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            # Training
            self.train()
            train_losses = []

            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}',
                              disable=not verbose, leave=False)

            for batch in progress_bar:
                optimizer.zero_grad()

                # Training step
                loss = self.training_step(batch, criterion)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                progress_bar.set_postfix({'train_loss': np.mean(train_losses)})

            avg_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(avg_train_loss)

            # Validation
            if val_loader:
                self.eval()
                val_losses = []
                all_preds = []
                all_labels = []

                for batch in val_loader:
                    loss, preds, labels = self.validation_step(batch, criterion)
                    val_losses.append(loss)
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

                avg_val_loss = np.mean(val_losses)
                self.history['val_loss'].append(avg_val_loss)

                # Calculate accuracy
                all_preds = torch.cat(all_preds)
                all_labels = torch.cat(all_labels)
                accuracy = (all_preds == all_labels).float().mean().item()

                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}: "
                          f"train_loss={avg_train_loss:.4f}, "
                          f"val_loss={avg_val_loss:.4f}, "
                          f"val_acc={accuracy:.4f}")

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0

                    # Save best model
                    if save_path:
                        self.save(save_path)
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}")

        return self.history

    def predict(
        self,
        data_loader: DataLoader,
        return_proba: bool = False
    ) -> np.ndarray:
        """
        Make predictions.

        Args:
            data_loader: DataLoader with input data
            return_proba: Return probabilities instead of class labels

        Returns:
            Predictions (class labels or probabilities)
        """
        self.eval()
        self.to(self.device)

        all_outputs = []

        with torch.no_grad():
            for batch in data_loader:
                features, _ = batch
                features = features.to(self.device)

                outputs = self(features)
                all_outputs.append(outputs.cpu())

        all_outputs = torch.cat(all_outputs)

        if return_proba:
            # Return probabilities
            if all_outputs.shape[-1] > 1:
                probs = F.softmax(all_outputs, dim=-1)
            else:
                probs = torch.sigmoid(all_outputs).squeeze()
            return probs.numpy()
        else:
            # Return class labels
            if all_outputs.shape[-1] > 1:
                preds = torch.argmax(all_outputs, dim=-1)
            else:
                preds = (torch.sigmoid(all_outputs) > 0.5).long().squeeze()
            return preds.numpy()

    def save(self, filepath: str):
        """Save model state."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'history': self.history
        }, filepath)
        print(f"✓ Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})
        print(f"✓ Model loaded from {filepath}")


# Example usage
if __name__ == '__main__':
    import sys
    sys.path.append('/Users/arnold/Documents/Dev/multimodal-depression-detection')
    from src.utils.data_loader import load_features_labels, train_test_split_stratified
    from src.utils.pytorch_utils import get_dataloaders, set_seed

    print("=== Testing Base Deep Model ===\n")

    # Set seed
    set_seed(42)

    # Load data
    X, y, _ = load_features_labels()
    X_train, X_val, y_train, y_val = train_test_split_stratified(X, y, test_size=0.2)

    # Create dataloaders
    train_loader, val_loader = get_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=16
    )

    # Define simple test model
    class SimpleClassifier(BaseDeepModel):
        def __init__(self, input_dim: int, device: Optional[str] = None):
            super().__init__(device)
            self.fc1 = nn.Linear(input_dim, 32)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 2)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    # Create and train model
    model = SimpleClassifier(input_dim=X.shape[1])
    print(f"Model device: {model.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\nTraining model...")
    history = model.fit(
        train_loader,
        val_loader,
        epochs=10,
        patience=5,
        verbose=True
    )

    # Make predictions
    print("\nMaking predictions...")
    preds = model.predict(val_loader)
    probs = model.predict(val_loader, return_proba=True)

    print(f"Predictions shape: {preds.shape}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Sample predictions: {preds[:5]}")

    print("\n✓ Base deep model test complete!")
