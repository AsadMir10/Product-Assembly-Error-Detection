"""PatchCore feature extraction and memory bank creation.

No neural network training — just extracts features from good images
using a pretrained ResNet-18 and stores them as a memory bank.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import create_train_val_split
from src.model import PatchCoreModel, get_device


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    data_root: str,
    category: str,
    batch_size: int = 32,
    coreset_ratio: float = 0.1,
    output_dir: str = "models",
    seed: int = 42,
) -> None:
    """Build PatchCore memory bank from good images.

    Args:
        data_root: Path to MVTec AD dataset root.
        category: Product category (bottle, cable, transistor).
        batch_size: Batch size for feature extraction.
        coreset_ratio: Fraction of patches to keep in memory bank.
        output_dir: Directory to save the model.
        seed: Random seed.
    """
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    # Load good images
    print(f"Loading '{category}' dataset from {data_root}...")
    train_dataset, val_dataset = create_train_val_split(
        data_root=data_root,
        category=category,
        val_ratio=0.2,
        seed=seed,
    )
    print(f"Train: {len(train_dataset)} good images | Val: {len(val_dataset)} good images")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Build PatchCore model
    model = PatchCoreModel(device=device)

    print("\nExtracting features from good images...")
    model.fit(train_loader, coreset_ratio=coreset_ratio)

    print("\nComputing anomaly threshold from validation set...")
    model.compute_threshold(val_loader)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"best_{category}.pth")
    model.save(save_path)

    print(f"\nDone! Model saved to {save_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build PatchCore memory bank")
    parser.add_argument("--data_root", type=str, default="data/raw",
                        help="Path to MVTec AD dataset root")
    parser.add_argument("--category", type=str, default="bottle",
                        help="MVTec AD category (bottle, cable, transistor)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for feature extraction")
    parser.add_argument("--coreset_ratio", type=float, default=0.1,
                        help="Fraction of patches to keep in memory bank")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        category=args.category,
        batch_size=args.batch_size,
        coreset_ratio=args.coreset_ratio,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
