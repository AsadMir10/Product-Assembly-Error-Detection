# Presentation Outline — Visual Detection of Product Assembly Errors

**Duration:** 10 minutes + Q&A

---

## Slide 1: Title (30 sec)

- Project title: Visual Detection of Product Assembly Errors
- Team name and members
- Course: Computer Vision for Industry 5.0
- Date

---

## Slide 2: Problem & Motivation (1 min)

- Assembly errors cost manufacturers billions annually in recalls and rework
- Manual inspection is slow, inconsistent, and prone to fatigue-related errors
- Industry 5.0 vision: AI-augmented quality control that empowers human operators
- Key challenge: defective samples are rare — we need a method that learns from NORMAL data only

---

## Slide 3: Dataset Overview (1 min)

- MVTec Anomaly Detection Dataset (Bergmann et al., CVPR 2019)
- 3 categories: Bottle (3 defect types), Cable (8 defect types), Transistor (4 defect types)
- Training: only "good" images (~200 per category)
- Test: good + multiple defect types per category
- Show sample images: good vs defective side by side

---

## Slide 4: Approach — PatchCore Anomaly Detection (1.5 min)

- **Key idea:** Use a pretrained WideResNet-50 to extract patch-level features from normal images
- Build a **memory bank** of normal patch features (coreset subsampling for efficiency)
- At test time: compute **nearest-neighbor distance** between test patches and memory bank
- High distance = anomaly → no training required, just feature extraction
- Architecture diagram: Image → WideResNet-50 (layer2 + layer3) → Patch Features → Memory Bank → Distance → Anomaly Score
- Threshold = mean + σ from validation good image scores

---

## Slide 5: Why PatchCore over Autoencoders? (1 min)

- We first tried a convolutional autoencoder (41% accuracy) and U-Net autoencoder (also poor)
- Problem: autoencoders reconstructed defects too well → couldn't distinguish good from bad
- PatchCore leverages **pretrained ImageNet features** — much richer representations
- No gradient-based training needed — memory bank built in seconds
- State-of-the-art method on MVTec AD (Roth et al., CVPR 2022)

---

## Slide 6: Results — Metrics (1.5 min)

| Category | Accuracy | F1 | AUC | Recall |
|-----------|----------|------|------|--------|
| Bottle | **93%** | 0.95 | 0.99 | 98% |
| Cable | **78%** | 0.82 | 0.86 | 82% |
| Transistor | **71%** | 0.69 | 0.80 | 80% |

- Anomaly score distribution plots (good vs defective — clear separation)
- ROC curves showing strong AUC
- Confusion matrices per category
- Key insight: recall is critical in manufacturing — missing a defect is worse than a false alarm

---

## Slide 7: Results — Anomaly Heatmaps (1 min)

- Side-by-side: Original Image → Anomaly Heatmap Overlay
- Good products: low, uniform distance (cool colors)
- Defective products: high distance concentrated at defect location (hot colors)
- The heatmap shows WHERE the defect is — without any pixel-level supervision
- Examples from each category (bottle broken, cable missing wire, transistor bent lead)

---

## Slide 8: Live Demo / Streamlit App (1.5 min)

- Streamlit app walkthrough:
  - Select category (bottle, cable, transistor)
  - Upload a product image or use sample images
  - See 2-panel view: Original + Anomaly Heatmap
  - Prediction: PASS or DEFECT with anomaly score
  - Sensitivity slider to adjust threshold
- Test with good and defective examples live

---

## Slide 9: Challenges & Lessons Learned (1 min)

- **Challenge:** Autoencoder approach failed — skip connections passed defects through
- **Challenge:** Cable and transistor have very subtle defects (AUC 0.80-0.86 vs bottle 0.99)
- **Lesson:** Pretrained features (WideResNet-50) vastly outperform learned features for anomaly detection
- **Lesson:** PatchCore is practical — no training, runs in seconds, strong results
- **Lesson:** Anomaly heatmaps build operator trust — they can SEE why something was flagged
- **Future work:** PaDiM, EfficientAD, real-time video processing, domain adaptation

---

## Slide 10: Q&A

- Summary of key contributions:
  1. Unsupervised anomaly detection — no labeled defects needed
  2. PatchCore with WideResNet-50 achieves up to 93% accuracy and 0.99 AUC
  3. Anomaly heatmaps that localize defects without pixel-level supervision
  4. Interactive Streamlit prototype for quality control workflows
- Thank the audience
- Open for questions
