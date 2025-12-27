"""
Data Augmentation for Tabular Features

Implements augmentation strategies for small-sample tabular data:
1. Mixup - Linear interpolation between samples
2. Gaussian noise - Add random noise
3. Feature cutout - Random feature masking
4. Feature swap - Swap random features between samples
"""

import torch
import numpy as np
from typing import Tuple, Optional


class TabularAugmentation:
    """Data augmentation for tabular data."""

    def __init__(self, augmentation_strength: float = 0.2):
        """
        Initialize augmentation.

        Args:
            augmentation_strength: Overall strength of augmentation (0-1)
        """
        self.strength = augmentation_strength

    def mixup(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        alpha: float = 0.2
    ) -> torch.Tensor:
        """
        Mixup: Linear interpolation between two samples.

        Args:
            x1: First sample [batch, features]
            x2: Second sample [batch, features]
            alpha: Beta distribution parameter

        Returns:
            Mixed sample
        """
        lam = np.random.beta(alpha, alpha)
        return lam * x1 + (1 - lam) * x2

    def gaussian_noise(
        self,
        x: torch.Tensor,
        std: Optional[float] = None
    ) -> torch.Tensor:
        """
        Add Gaussian noise to features.

        Args:
            x: Input features [batch, features]
            std: Standard deviation (defaults to strength)

        Returns:
            Noisy features
        """
        if std is None:
            std = self.strength

        noise = torch.randn_like(x) * std
        return x + noise

    def feature_cutout(
        self,
        x: torch.Tensor,
        p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Randomly mask out features (set to 0).

        Args:
            x: Input features [batch, features]
            p: Probability of masking each feature

        Returns:
            Masked features
        """
        if p is None:
            p = self.strength

        mask = torch.rand_like(x) > p
        return x * mask.float()

    def feature_swap(
        self,
        x: torch.Tensor,
        p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Randomly swap features between samples in batch.

        Args:
            x: Input features [batch, features]
            p: Probability of swapping each feature

        Returns:
            Swapped features
        """
        if p is None:
            p = self.strength

        batch_size = x.shape[0]
        if batch_size < 2:
            return x

        # Create random permutation for each feature
        x_aug = x.clone()
        for i in range(x.shape[1]):
            if torch.rand(1).item() < p:
                # Randomly permute this feature across batch
                perm = torch.randperm(batch_size)
                x_aug[:, i] = x[perm, i]

        return x_aug

    def random_augment(
        self,
        x: torch.Tensor,
        augmentation_type: Optional[str] = None
    ) -> torch.Tensor:
        """
        Apply random augmentation.

        Args:
            x: Input features [batch, features]
            augmentation_type: Type ('mixup', 'noise', 'cutout', 'swap', or None for random)

        Returns:
            Augmented features
        """
        if augmentation_type is None:
            # Randomly choose augmentation
            augmentation_type = np.random.choice(['noise', 'cutout', 'swap'])

        if augmentation_type == 'mixup':
            # For mixup, we need two samples
            if x.shape[0] >= 2:
                # Randomly pair samples
                indices = torch.randperm(x.shape[0])
                return self.mixup(x, x[indices])
            else:
                return self.gaussian_noise(x)

        elif augmentation_type == 'noise':
            return self.gaussian_noise(x)

        elif augmentation_type == 'cutout':
            return self.feature_cutout(x)

        elif augmentation_type == 'swap':
            return self.feature_swap(x)

        else:
            raise ValueError(f"Unknown augmentation type: {augmentation_type}")

    def create_positive_pairs(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create positive pairs for contrastive learning.
        Each sample is augmented twice to create a pair.

        Args:
            x: Input features [batch, features]

        Returns:
            Tuple of (view1, view2) - two augmented views of same samples
        """
        view1 = self.random_augment(x)
        view2 = self.random_augment(x)

        return view1, view2

    def augment_class_samples(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        class_label: int,
        n_augmented: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate augmented samples for a specific class.

        Args:
            X: All features [n_samples, n_features]
            y: All labels [n_samples]
            class_label: Class to augment (0 or 1)
            n_augmented: Number of augmented samples per original sample

        Returns:
            Tuple of (augmented_X, augmented_y)
        """
        # Get samples from target class
        class_indices = (y == class_label).nonzero(as_tuple=True)[0]
        class_samples = X[class_indices]

        if len(class_samples) == 0:
            raise ValueError(f"No samples found for class {class_label}")

        augmented_samples = []

        for i in range(len(class_samples)):
            original = class_samples[i:i+1]  # Keep batch dimension

            for _ in range(n_augmented):
                # Apply random augmentation
                aug_type = np.random.choice(['noise', 'cutout', 'swap'])
                if aug_type == 'noise':
                    aug_sample = self.gaussian_noise(original)
                elif aug_type == 'cutout':
                    aug_sample = self.feature_cutout(original)
                else:  # swap - use mixup with random class sample instead
                    if len(class_samples) > 1:
                        other_idx = np.random.choice(
                            [j for j in range(len(class_samples)) if j != i]
                        )
                        other = class_samples[other_idx:other_idx+1]
                        aug_sample = self.mixup(original, other)
                    else:
                        aug_sample = self.gaussian_noise(original)

                augmented_samples.append(aug_sample)

        # Concatenate all augmented samples
        augmented_X = torch.cat(augmented_samples, dim=0)
        augmented_y = torch.full((len(augmented_X),), class_label, dtype=y.dtype)

        return augmented_X, augmented_y


# Example usage and testing
if __name__ == '__main__':
    import sys
    sys.path.append('/Users/arnold/Documents/Dev/multimodal-depression-detection')
    from src.utils.data_loader import load_features_labels

    print("=== Testing Tabular Augmentation ===\n")

    # Load data
    X, y, _ = load_features_labels()
    print(f"Original data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # Create augmenter
    augmenter = TabularAugmentation(augmentation_strength=0.2)

    # Test individual augmentations
    print("\nTesting individual augmentations:")

    sample = X_tensor[:4]  # Take 4 samples
    print(f"Sample shape: {sample.shape}")

    # Gaussian noise
    noisy = augmenter.gaussian_noise(sample)
    print(f"\nGaussian noise:")
    print(f"  Original mean: {sample.mean():.4f}, std: {sample.std():.4f}")
    print(f"  Noisy mean: {noisy.mean():.4f}, std: {noisy.std():.4f}")
    print(f"  Difference: {(sample - noisy).abs().mean():.4f}")

    # Feature cutout
    cutout = augmenter.feature_cutout(sample)
    print(f"\nFeature cutout:")
    print(f"  Original non-zero: {(sample != 0).sum().item()}")
    print(f"  Cutout non-zero: {(cutout != 0).sum().item()}")
    print(f"  Features masked: {((sample != 0) & (cutout == 0)).sum().item()}")

    # Mixup
    if sample.shape[0] >= 2:
        mixed = augmenter.mixup(sample[0:1], sample[1:2])
        print(f"\nMixup:")
        print(f"  Sample 1 mean: {sample[0].mean():.4f}")
        print(f"  Sample 2 mean: {sample[1].mean():.4f}")
        print(f"  Mixed mean: {mixed.mean():.4f}")

    # Create positive pairs
    view1, view2 = augmenter.create_positive_pairs(sample)
    print(f"\nPositive pairs:")
    print(f"  View 1 shape: {view1.shape}")
    print(f"  View 2 shape: {view2.shape}")
    print(f"  Correlation: {torch.corrcoef(torch.stack([view1.flatten(), view2.flatten()]))[0, 1]:.4f}")

    # Augment positive class
    print(f"\nAugmenting positive class:")
    print(f"  Original positive samples: {y.sum()}")

    aug_X, aug_y = augmenter.augment_class_samples(
        X_tensor, y_tensor, class_label=1, n_augmented=10
    )

    print(f"  Augmented samples: {len(aug_X)}")
    print(f"  Augmented labels: {aug_y.unique()} (all should be 1)")

    # Combined dataset
    combined_X = torch.cat([X_tensor, aug_X], dim=0)
    combined_y = torch.cat([y_tensor, aug_y], dim=0)

    print(f"\nCombined dataset:")
    print(f"  Total samples: {len(combined_X)}")
    print(f"  Class distribution: {np.bincount(combined_y.numpy())}")
    print(f"  New balance ratio: {(combined_y == 0).sum().item() / (combined_y == 1).sum().item():.2f}:1")

    print("\n✓ Augmentation test complete!")
