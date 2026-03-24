# Visual Detection of Product Assembly Errors

**Course:** Computer Vision for Industry 5.0
**Instructor:** Dr. Yosra Hajjaji
**Team:** [Team Name]
**Date:** [Date]

---

## 1. Introduction

### 1.1 Problem Statement

In modern manufacturing, assembly errors such as missing components, misaligned parts, and surface defects pose significant challenges to product quality and safety. Manual visual inspection is time-consuming, error-prone, and inconsistent — particularly under fatigue or high throughput conditions.

### 1.2 Industry 5.0 Context

Industry 5.0 emphasizes a human-centric approach to manufacturing, where intelligent systems augment human capabilities rather than replacing them. Our system supports this vision by providing:
- **Automated anomaly detection** that learns from defect-free examples only
- **Visual explanations** (anomaly heatmaps) that show operators exactly WHERE a defect was detected
- **Real-time feedback** through a prototype interface for quality control workflows

### 1.3 Objective

Design and implement a deep learning-based anomaly detection system that:
1. Learns the appearance of **normal (good)** products from pretrained features
2. Detects **anomalies** via nearest-neighbor distance in feature space
3. Provides **anomaly heatmaps** that localize defective regions without pixel-level supervision

---

## 2. Dataset

### 2.1 MVTec Anomaly Detection Dataset

- **Source:** MVTec Research (Bergmann et al., CVPR 2019)
- **Size:** ~5,000 high-resolution images across 15 product categories
- **Categories used:** Bottle, Cable, Transistor
- **Structure:** Training set contains only "good" images; test set contains both good and defective
- **Ground truth:** Pixel-level segmentation masks for defective regions

### 2.2 Category Details

| Category | Train (good) | Test (good) | Test (defective) | Defect Types |
|-----------|-------------|-------------|-------------------|--------------|
| Bottle | 209 | 20 | 63 | broken_large, broken_small, contamination |
| Cable | 224 | 58 | 92 | bent_wire, cable_swap, combined, cut_inner, cut_outer, missing_cable, missing_wire, poke_insulation |
| Transistor | 213 | 60 | 40 | bent_lead, cut_lead, damaged_case, misplaced |

### 2.3 Preprocessing

- Resized all images to 224×224 pixels
- Normalized with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Training set split: 80% train / 20% validation (from "good" images only)

---

## 3. Methodology

### 3.1 Approach Evolution

We explored three approaches before arriving at PatchCore:

| Approach | Accuracy | F1 | AUC | Outcome |
|----------|----------|------|------|---------|
| Conv Autoencoder (MSE) | 41% | 0.49 | 0.40 | Failed — poor reconstruction quality |
| U-Net Autoencoder (SSIM+MSE) | ~40% | ~0.48 | ~0.39 | Failed — skip connections passed defects through |
| **PatchCore (WideResNet-50)** | **93%** | **0.95** | **0.99** | Success — pretrained features + nearest-neighbor |

1. **Convolutional Autoencoder (Attempt 1):** Encoder-decoder architecture (3→32→64→128→256 channels) trained with MSE reconstruction loss on good images only. Achieved only 41% accuracy and 0.40 AUC on the bottle test set — the model reconstructed defective images almost as well as good ones, producing overlapping anomaly score distributions.

2. **U-Net Autoencoder with SSIM Loss (Attempt 2):** Added skip connections to preserve spatial detail and combined SSIM + MSE loss for perceptually better reconstructions. Counterintuitively, skip connections made things worse — they passed defect details through to the decoder, so defective images were reconstructed just as well as good ones. Accuracy remained around 40%.

3. **PatchCore (Final Approach):** Switched from learned reconstructions to pretrained feature comparison. Uses a frozen WideResNet-50 backbone to extract patch-level features from good images, then detects anomalies via nearest-neighbor distance. Achieved 93% accuracy and 0.99 AUC on bottle — a dramatic improvement with zero training.

### 3.2 PatchCore Architecture

- **Feature Extractor:** WideResNet-50 pretrained on ImageNet (frozen, no training)
- **Feature Layers:** Layer 2 (512 channels, 28×28) + Layer 3 (1024 channels, 14×14 upsampled to 28×28)
- **Concatenated features:** 1536 channels per patch at 28×28 spatial resolution
- **Memory Bank:** Coreset subsampled patch features from training good images (10-50% kept)
- **Anomaly Score:** Maximum nearest-neighbor L2 distance between test patches and memory bank
- **Threshold:** mean + σ from validation good image anomaly scores

### 3.3 Why PatchCore?

- **No training required** — just feature extraction and nearest-neighbor search
- **Pretrained features** capture rich semantic and textural information from ImageNet
- **Anomaly localization** via per-patch distance maps (28×28 upscaled to 224×224)
- **State-of-the-art** on MVTec AD benchmark (Roth et al., CVPR 2022)

### 3.4 Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | WideResNet-50-2 (ImageNet pretrained) |
| Image size | 224×224 |
| Feature layers | layer2 + layer3 |
| Coreset ratio | 25% (bottle, cable), 50% (transistor) |
| Threshold sigma | 1.0 |
| Device | Apple M4 (MPS) |

---

## 4. Results

### 4.1 Detection Performance

| Category | Accuracy | Precision | Recall | F1 | AUC |
|-----------|----------|-----------|--------|------|------|
| Bottle | **93%** | 0.93 | 0.98 | 0.95 | 0.99 |
| Cable | **78%** | 0.82 | 0.82 | 0.82 | 0.86 |
| Transistor | **71%** | 0.60 | 0.80 | 0.69 | 0.80 |

### 4.2 Anomaly Score Distribution

*[Insert score distribution plots from results/ showing separation between good and defective scores for each category]*

### 4.3 Confusion Matrices

*[Insert confusion matrix figures from results/ for each category]*

### 4.4 ROC Curves

*[Insert ROC curve plots showing AUC per category]*

### 4.5 Anomaly Heatmap Visualizations

*[Insert examples showing:]*
- Good products with low, uniform patch distances (cool heatmap)
- Defective products with high distance localized at the defect region (hot heatmap)
- Examples from each category

### 4.6 Analysis

- **Bottle** performs best (AUC 0.99) — defects are large and visually distinct
- **Cable** is harder (AUC 0.86) — 8 defect types, some very subtle (poke_insulation)
- **Transistor** is hardest (AUC 0.80) — misplaced and bent_lead defects are small spatial changes
- High recall is critical in manufacturing — missing a defect is costlier than a false alarm

---

## 5. Prototype Application

### 5.1 Streamlit Interface

The prototype application provides:
- **Category selection** (bottle, cable, transistor) with per-category models
- **Image upload** for on-demand inspection
- **Real-time anomaly detection** with anomaly score
- **2-panel display:** Original Image + Anomaly Heatmap Overlay
- **Sensitivity slider** to adjust the detection threshold
- **Sample images** to quickly test with known examples
- **Model metrics tab** showing evaluation plots

*[Insert screenshot(s) of the Streamlit app]*

### 5.2 Usage

```bash
streamlit run src/app.py
```

---

## 6. Conclusion

### 6.1 Summary

We implemented an unsupervised anomaly detection system for product assembly error detection using PatchCore. The system achieves up to 93% accuracy and 0.99 AUC on the MVTec AD bottle category, with strong anomaly localization through patch-distance heatmaps. The approach requires no training — only feature extraction from a pretrained WideResNet-50 — making it practical and fast to deploy.

### 6.2 Limitations

- Performance varies by category: subtle defects (transistor) are harder to detect
- Threshold selection impacts precision/recall tradeoff
- Memory bank size grows with training data — may need pruning for large-scale deployment
- Controlled lab conditions in MVTec AD may not reflect real factory environments

### 6.3 Future Work

- **EfficientAD** or **PaDiM** for faster inference
- Multi-class defect classification (not just good/bad)
- Real-time video processing for production line integration
- Domain adaptation to handle varying lighting and camera conditions
- Active learning: incorporate operator feedback to refine detection

---

## 7. References

1. Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., & Gehler, P. (2022). Towards Total Recall in Industrial Anomaly Detection. *CVPR*.
2. Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. *CVPR*.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
4. European Commission. (2021). Industry 5.0: Towards a sustainable, human-centric and resilient European industry.
