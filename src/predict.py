"""Single image inference for PatchCore anomaly detection.

Loads a saved PatchCore model, computes anomaly score, and generates
a heatmap showing where the anomaly is.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.visualize import visualize_from_path


def predict(
    image_path: str,
    model_path: str,
    save_dir: str = "results",
) -> tuple[str, float]:
    """Run anomaly detection on a single image.

    Args:
        image_path: Path to the input image.
        model_path: Path to the saved PatchCore model.
        save_dir: Directory to save the visualization.

    Returns:
        Tuple of (prediction_label, anomaly_score).
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    image_name = Path(image_path).stem
    save_path = str(Path(save_dir) / f"anomaly_{image_name}.png")

    overlay, anomaly_score, threshold, is_anomaly = visualize_from_path(
        model_path=model_path,
        image_path=image_path,
        save_path=save_path,
    )

    label = "DEFECT" if is_anomaly else "GOOD"
    icon = "\u274c" if is_anomaly else "\u2705"

    print(f"\n{'=' * 40}")
    print(f"  Prediction:     {icon} {label}")
    print(f"  Anomaly Score:  {anomaly_score:.4f}")
    print(f"  Threshold:      {threshold:.4f}")
    print(f"  Visualization:  {save_path}")
    print(f"{'=' * 40}\n")

    return label, anomaly_score


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Detect anomalies in a single image")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--model", type=str, default="models/best_bottle.pth",
                        help="Path to the saved PatchCore model")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory to save visualization")
    args = parser.parse_args()

    predict(
        image_path=args.image,
        model_path=args.model,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
