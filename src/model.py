"""PatchCore anomaly detection model.

Uses a pretrained ResNet-18 to extract patch-level features from good images,
stores them in a memory bank, and detects anomalies by computing nearest-neighbor
distances at test time. No training required — just feature extraction.

Reference: Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# ImageNet normalization (required for pretrained ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device() -> torch.device:
    """Get the best available device (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_transform(image_size: int = 224) -> transforms.Compose:
    """Get preprocessing transform for PatchCore."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class FeatureExtractor(nn.Module):
    """Extract intermediate features from a pretrained ResNet-18.

    Hooks into layer2 and layer3 to get multi-scale patch features.
    These mid-level features capture both texture and structure.
    """

    def __init__(self) -> None:
        super().__init__()
        backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
        backbone.eval()

        self.layer1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1,
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and concatenate multi-scale features.

        Args:
            x: Input images (B, 3, 224, 224), ImageNet-normalized.

        Returns:
            Patch features (B, C, H, W) where H=W=28.
        """
        f1 = self.layer1(x)   # (B, 256, 56, 56)
        f2 = self.layer2(f1)  # (B, 512, 28, 28)
        f3 = self.layer3(f2)  # (B, 1024, 14, 14)

        # Upsample f3 to match f2 spatial size (28x28)
        f3_up = F.interpolate(f3, size=f2.shape[2:], mode="bilinear", align_corners=False)

        return torch.cat([f2, f3_up], dim=1)  # (B, 1536, 28, 28)


class PatchCoreModel:
    """PatchCore anomaly detection model.

    Stores a memory bank of patch features from good images.
    Detects anomalies by finding nearest-neighbor distances.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or get_device()
        self.feature_extractor = FeatureExtractor().to(self.device)
        self.feature_extractor.eval()
        self.memory_bank: Optional[torch.Tensor] = None  # (N, C)
        self.threshold: float = 0.0

    def fit(self, dataloader: torch.utils.data.DataLoader, coreset_ratio: float = 0.1) -> None:
        """Build memory bank from good images.

        Args:
            dataloader: DataLoader of good training images.
            coreset_ratio: Fraction of patches to keep (random subsampling).
        """
        all_features = []

        for images, _ in dataloader:
            images = images.to(self.device)
            features = self.feature_extractor(images)  # (B, C, H, W)

            # Reshape to patch features: (B*H*W, C)
            B, C, H, W = features.shape
            patches = features.permute(0, 2, 3, 1).reshape(-1, C)
            all_features.append(patches.cpu())

        all_features = torch.cat(all_features, dim=0)  # (total_patches, C)

        # Random coreset subsampling to reduce memory
        n_patches = all_features.shape[0]
        n_keep = max(int(n_patches * coreset_ratio), 1000)
        n_keep = min(n_keep, n_patches)

        indices = torch.randperm(n_patches)[:n_keep]
        self.memory_bank = all_features[indices]

        print(f"Memory bank: {self.memory_bank.shape[0]} patches "
              f"(from {n_patches} total, {coreset_ratio:.0%} kept)")

    def compute_anomaly_map(self, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Compute anomaly scores and maps for a batch of images.

        Args:
            images: Input images (B, 3, 224, 224), ImageNet-normalized.

        Returns:
            Tuple of (anomaly_maps, anomaly_scores).
            - anomaly_maps: (B, 28, 28) per-patch distance maps.
            - anomaly_scores: (B,) per-image anomaly scores (max distance).
        """
        images = images.to(self.device)
        features = self.feature_extractor(images)  # (B, C, H, W)
        B, C, H, W = features.shape

        # Reshape to (B*H*W, C)
        patches = features.permute(0, 2, 3, 1).reshape(-1, C)

        # Compute nearest-neighbor distances to memory bank
        # Process in chunks to avoid OOM
        distances = self._compute_distances(patches)  # (B*H*W,)

        # Reshape to (B, H, W)
        anomaly_maps = distances.reshape(B, H, W)
        anomaly_scores = anomaly_maps.reshape(B, -1).max(dim=1)[0].numpy()
        anomaly_maps = anomaly_maps.numpy()

        return anomaly_maps, anomaly_scores

    def _compute_distances(self, patches: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
        """Compute nearest-neighbor distances in chunks to save memory."""
        memory = self.memory_bank
        all_distances = []

        for i in range(0, patches.shape[0], chunk_size):
            chunk = patches[i:i + chunk_size].cpu()
            # L2 distance to all memory bank entries
            dists = torch.cdist(chunk, memory)  # (chunk, N_memory)
            min_dists = dists.min(dim=1)[0]  # (chunk,)
            all_distances.append(min_dists)

        return torch.cat(all_distances, dim=0)

    def compute_threshold(self, dataloader: torch.utils.data.DataLoader, sigma: float = 1.0) -> float:
        """Compute anomaly threshold from validation good images.

        Args:
            dataloader: DataLoader of good validation images.
            sigma: Number of standard deviations above mean.

        Returns:
            Anomaly detection threshold.
        """
        all_scores = []

        for images, _ in dataloader:
            _, scores = self.compute_anomaly_map(images)
            all_scores.extend(scores.tolist())

        scores_arr = np.array(all_scores)
        self.threshold = float(scores_arr.mean() + sigma * scores_arr.std())
        print(f"Threshold: {self.threshold:.4f} (mean={scores_arr.mean():.4f}, std={scores_arr.std():.4f})")
        return self.threshold

    def save(self, path: str) -> None:
        """Save the memory bank and threshold."""
        data = {
            "memory_bank": self.memory_bank,
            "threshold": self.threshold,
        }
        torch.save(data, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "PatchCoreModel":
        """Load a saved PatchCore model."""
        model = cls(device=device)
        data = torch.load(path, map_location="cpu", weights_only=True)
        model.memory_bank = data["memory_bank"]
        model.threshold = float(data["threshold"])
        return model
