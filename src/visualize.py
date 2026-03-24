"""Visualization module for PatchCore anomaly detection.

Generates anomaly heatmaps that highlight defective regions by upscaling
the patch-level distance maps from PatchCore to full image resolution.
"""

import os
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import PatchCoreModel, get_device, get_transform


def compute_anomaly_heatmap(
    model: PatchCoreModel,
    image: Image.Image,
    image_size: int = 224,
) -> tuple[np.ndarray, float]:
    """Compute anomaly heatmap for a single image.

    Args:
        model: Loaded PatchCore model.
        image: PIL Image to analyze.
        image_size: Input image size.

    Returns:
        Tuple of (heatmap, anomaly_score).
        - heatmap: (image_size, image_size) array normalized to [0, 1].
        - anomaly_score: Scalar anomaly score (max patch distance).
    """
    transform = get_transform(image_size)
    input_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)

    anomaly_maps, anomaly_scores = model.compute_anomaly_map(input_tensor)

    # anomaly_maps is (1, 28, 28), upscale to image_size
    heatmap = anomaly_maps[0]  # (28, 28)
    heatmap = cv2.resize(heatmap, (image_size, image_size), interpolation=cv2.INTER_CUBIC)

    # Normalize to [0, 1]
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap = np.zeros_like(heatmap)

    anomaly_score = float(anomaly_scores[0])

    return heatmap, anomaly_score


def create_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    image_size: int = 224,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay anomaly heatmap on the original image.

    Args:
        image: Original PIL image.
        heatmap: Heatmap (H, W) in [0, 1].
        image_size: Target size.
        alpha: Overlay opacity.

    Returns:
        Overlay image (H, W, 3) as uint8 array.
    """
    img_resized = np.array(image.resize((image_size, image_size))) / 255.0

    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

    overlay = (1 - alpha) * img_resized + alpha * heatmap_colored
    overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    return overlay


def save_visualization(
    image: Image.Image,
    heatmap: np.ndarray,
    anomaly_score: float,
    threshold: float,
    save_path: str,
    image_size: int = 224,
) -> None:
    """Save a 2-panel visualization: original + anomaly heatmap overlay."""
    is_anomaly = anomaly_score > threshold
    overlay = create_overlay(image, heatmap, image_size)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original
    img_resized = image.resize((image_size, image_size))
    axes[0].imshow(img_resized)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Heatmap overlay
    axes[1].imshow(overlay)
    status = "DEFECT" if is_anomaly else "GOOD"
    color = "red" if is_anomaly else "green"
    axes[1].set_title(f"Anomaly Heatmap — {status} (score: {anomaly_score:.4f})", color=color)
    axes[1].axis("off")

    plt.suptitle(
        f"Anomaly Score: {anomaly_score:.4f} | Threshold: {threshold:.4f}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {save_path}")


def visualize_from_path(
    model_path: str,
    image_path: str,
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, float, float, bool]:
    """Generate visualization from file paths.

    Returns:
        Tuple of (overlay, anomaly_score, threshold, is_anomaly).
    """
    model = PatchCoreModel.load(model_path, device=device)
    image = Image.open(image_path).convert("RGB")

    heatmap, anomaly_score = compute_anomaly_heatmap(model, image)
    is_anomaly = anomaly_score > model.threshold

    overlay = create_overlay(image, heatmap)

    if save_path:
        save_visualization(image, heatmap, anomaly_score, model.threshold, save_path)

    return overlay, anomaly_score, model.threshold, is_anomaly
