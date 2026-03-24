"""Streamlit prototype app for Assembly Error Detection.

Provides a web interface for uploading product images and getting
PatchCore anomaly detection results with heatmap overlays.

Run with: streamlit run src/app.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import PatchCoreModel, get_device
from src.visualize import compute_anomaly_heatmap, create_overlay


CATEGORIES = ["bottle", "cable", "transistor"]


@st.cache_resource
def cached_load_model(model_path: str) -> PatchCoreModel:
    """Load and cache the PatchCore model."""
    device = get_device()
    return PatchCoreModel.load(model_path, device=device)


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Assembly Error Detection",
        page_icon="\U0001f50d",
        layout="wide",
    )

    st.title("\U0001f50d Assembly Error Detection System")
    st.markdown(
        "Upload a product image to detect assembly defects. "
        "The system uses **PatchCore** (pretrained WideResNet-50 features + nearest-neighbor) "
        "to detect anomalies — the heatmap shows **where** the defect is."
    )

    # Sidebar
    st.sidebar.header("Settings")
    category = st.sidebar.selectbox("Category", CATEGORIES)
    sensitivity = st.sidebar.slider(
        "Sensitivity", 0.5, 2.0, 1.0, 0.1,
        help="Multiplier for the anomaly threshold. Lower = more sensitive (more detections)."
    )

    model_path = f"models/best_{category}.pth"

    if not Path(model_path).exists():
        st.warning(
            f"No trained model found at `{model_path}`. "
            f"Train first: `python -m src.train --category {category}`"
        )
        st.sidebar.error("Model: Not Found")
    else:
        st.sidebar.success("Model: Loaded")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### How it works")
    st.sidebar.markdown(
        "1. **PatchCore** extracts features from a pretrained WideResNet-50\n"
        "2. Compares test image patches to a **memory bank** of good patches\n"
        "3. High distance = anomaly — heatmap shows **where**"
    )

    # Main content
    tab1, tab2 = st.tabs(["\U0001f4f7 Upload Image", "\U0001f4ca Model Metrics"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Upload a product image",
            type=["png", "jpg", "jpeg", "bmp"],
        )

        # Sample images
        sample_dir = Path("data/raw")
        cat_test = sample_dir / category / "test"
        sample_images = []
        if cat_test.exists():
            for defect_dir in sorted(cat_test.iterdir()):
                if defect_dir.is_dir():
                    imgs = sorted(defect_dir.glob("*.png"))[:1]
                    for img in imgs:
                        sample_images.append((img, f"{category}/{defect_dir.name}"))

        if sample_images:
            st.markdown("**Or try a sample image:**")
            cols = st.columns(min(len(sample_images), 6))
            for i, (img_path, label) in enumerate(sample_images[:6]):
                with cols[i]:
                    if st.button(label, key=f"sample_{i}"):
                        uploaded_file = img_path

        if uploaded_file is not None and Path(model_path).exists():
            # Load image
            if isinstance(uploaded_file, Path):
                image = Image.open(uploaded_file).convert("RGB")
            else:
                image = Image.open(uploaded_file).convert("RGB")

            model = cached_load_model(model_path)
            adjusted_threshold = model.threshold * sensitivity

            # Compute anomaly
            heatmap, anomaly_score = compute_anomaly_heatmap(model, image)
            overlay = create_overlay(image, heatmap)
            is_anomaly = anomaly_score > adjusted_threshold

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.image(image.resize((224, 224)), caption="Original", width=300)
            with col2:
                st.image(overlay, caption="Anomaly Heatmap", width=300)

            # Prediction result
            st.markdown("---")
            if is_anomaly:
                st.error(f"\u274c **DEFECT DETECTED** — Anomaly Score: {anomaly_score:.4f} (threshold: {adjusted_threshold:.4f})")
            else:
                st.success(f"\u2705 **PASS** — Anomaly Score: {anomaly_score:.4f} (threshold: {adjusted_threshold:.4f})")

            # Score bar
            max_score = max(anomaly_score, adjusted_threshold * 2)
            progress = min(anomaly_score / max_score, 1.0)
            st.markdown(f"**Anomaly Score:** {anomaly_score:.4f}")
            st.progress(progress)

    with tab2:
        st.subheader("Model Performance")

        if not Path(model_path).exists():
            st.info("Train a model to see metrics here.")
        else:
            st.markdown(
                f"Run evaluation:\n"
                f"```bash\npython -m src.evaluate --model {model_path} --category {category}\n```"
            )

            results_dir = Path("results")
            for name, caption in [
                (f"confusion_matrix_{category}.png", "Confusion Matrix"),
                (f"score_distribution_{category}.png", "Anomaly Score Distribution"),
                (f"roc_curve_{category}.png", "ROC Curve"),
            ]:
                path = results_dir / name
                if path.exists():
                    st.image(str(path), caption=caption, width=300)


if __name__ == "__main__":
    main()
