"""Evaluation module for PatchCore anomaly detection.

Computes anomaly scores for test images, applies threshold, and generates
metrics (accuracy, precision, recall, F1), confusion matrix, ROC curve,
and example visualizations.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import MVTecDataset
from src.model import PatchCoreModel, get_device
from src.visualize import compute_anomaly_heatmap, create_overlay


def compute_all_scores(
    model: PatchCoreModel,
    dataloader: DataLoader,
) -> tuple[list[int], list[float]]:
    """Compute per-image anomaly scores for the entire dataset.

    Returns:
        Tuple of (true_labels, anomaly_scores).
    """
    all_labels = []
    all_scores = []

    for images, labels in tqdm(dataloader, desc="Computing anomaly scores"):
        _, scores = model.compute_anomaly_map(images)
        all_scores.extend(scores.tolist())
        all_labels.extend(labels.numpy().tolist())

    return all_labels, all_scores


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    save_path: str,
    class_names: tuple[str, str] = ("Good", "Defective"),
) -> None:
    """Generate and save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_score_distribution(
    y_true: list[int],
    scores: list[float],
    threshold: float,
    save_path: str,
) -> None:
    """Plot distribution of anomaly scores for good vs defective images."""
    scores_arr = np.array(scores)
    labels_arr = np.array(y_true)

    good_scores = scores_arr[labels_arr == 0]
    defective_scores = scores_arr[labels_arr == 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(good_scores, bins=30, alpha=0.6, label="Good", color="green")
    ax.hist(defective_scores, bins=30, alpha=0.6, label="Defective", color="red")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2,
               label=f"Threshold ({threshold:.4f})")
    ax.set_xlabel("Anomaly Score (Patch Distance)")
    ax.set_ylabel("Count")
    ax.set_title("Anomaly Score Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Score distribution saved to {save_path}")


def plot_roc_curve(
    y_true: list[int],
    scores: list[float],
    save_path: str,
) -> float:
    """Plot ROC curve and return AUC."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to {save_path}")

    return auc


def plot_example_grid(
    dataset: MVTecDataset,
    model: PatchCoreModel,
    y_true: list[int],
    y_pred: list[int],
    scores: list[float],
    correct: bool,
    save_path: str,
    max_images: int = 8,
) -> None:
    """Plot a grid of correctly or incorrectly classified examples with heatmaps."""
    indices = [
        i for i in range(len(y_true))
        if (y_true[i] == y_pred[i]) == correct
    ]

    if not indices:
        print(f"No {'correct' if correct else 'incorrect'} predictions to display.")
        return

    indices = indices[:max_images]
    n_images = len(indices)

    fig, axes = plt.subplots(n_images, 2, figsize=(8, 4 * n_images))
    if n_images == 1:
        axes = axes[np.newaxis, :]

    for i, idx in enumerate(indices):
        img_path = dataset.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        heatmap, score = compute_anomaly_heatmap(model, image)
        overlay = create_overlay(image, heatmap)

        # Original
        axes[i, 0].imshow(image.resize((224, 224)))
        true_label = "Good" if y_true[idx] == 0 else "Defective"
        axes[i, 0].set_title(f"True: {true_label}", fontsize=10)
        axes[i, 0].axis("off")

        # Heatmap overlay
        pred_label = "Good" if y_pred[idx] == 0 else "Defective"
        color = "green" if y_true[idx] == y_pred[idx] else "red"
        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f"Pred: {pred_label} ({score:.4f})", fontsize=10, color=color)
        axes[i, 1].axis("off")

    status = "Correct" if correct else "Incorrect"
    fig.suptitle(f"{status} Predictions", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"{status} predictions grid saved to {save_path}")


def evaluate(
    model_path: str,
    data_root: str,
    category: str,
    batch_size: int = 32,
    output_dir: str = "results",
) -> dict[str, float]:
    """Full evaluation pipeline.

    Args:
        model_path: Path to the saved PatchCore model.
        data_root: Path to the MVTec AD dataset root.
        category: Product category or 'all'.
        batch_size: Batch size.
        output_dir: Directory to save outputs.

    Returns:
        Dictionary with evaluation metrics.
    """
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    model = PatchCoreModel.load(model_path, device=device)
    threshold = model.threshold
    print(f"Loaded model from {model_path}")
    print(f"Anomaly threshold: {threshold:.4f}")

    # Load test dataset
    test_dataset = MVTecDataset(data_root=data_root, category=category, split="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    counts = test_dataset.get_class_counts()
    print(f"Test set — Good: {counts['good']} | Defective: {counts['defective']}")

    # Compute anomaly scores
    y_true, scores = compute_all_scores(model, test_loader)

    # Apply threshold
    y_pred = [1 if s > threshold else 0 for s in scores]

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    for name, value in metrics.items():
        print(f"  {name.capitalize():>10}: {value:.4f}")
    print("=" * 40)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Good", "Defective"],
                                zero_division=0))

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    plot_confusion_matrix(y_true, y_pred,
                          save_path=os.path.join(output_dir, f"confusion_matrix_{category}.png"))
    plot_score_distribution(y_true, scores, threshold,
                            save_path=os.path.join(output_dir, f"score_distribution_{category}.png"))

    try:
        auc = plot_roc_curve(y_true, scores,
                             save_path=os.path.join(output_dir, f"roc_curve_{category}.png"))
        metrics["auc"] = auc
        print(f"  AUC: {auc:.4f}")
    except ValueError:
        print("  Could not compute ROC AUC (single class in test set).")

    plot_example_grid(test_dataset, model, y_true, y_pred, scores, correct=True,
                      save_path=os.path.join(output_dir, f"correct_{category}.png"))
    plot_example_grid(test_dataset, model, y_true, y_pred, scores, correct=False,
                      save_path=os.path.join(output_dir, f"incorrect_{category}.png"))

    return metrics


def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate PatchCore anomaly detection model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to saved PatchCore model")
    parser.add_argument("--data_root", type=str, default="data/raw",
                        help="Path to MVTec AD dataset root")
    parser.add_argument("--category", type=str, default="bottle",
                        help="MVTec AD category or 'all'")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save outputs")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        data_root=args.data_root,
        category=args.category,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
