# Visual Detection of Product Assembly Errors

**Course:** Computer Vision for Industry 5.0 — ECE Engineering School
**Instructor:** Dr. Yosra Hajjaji
**Team:** Asad Ahsan, Abdullah Hameed, Francis Gallo

## Problem Statement

In modern manufacturing, assembly errors — missing components, misaligned parts, or damaged surfaces — lead to costly recalls, safety hazards, and production delays. Industry 5.0 emphasizes human-centric, resilient, and sustainable manufacturing, where intelligent visual inspection systems augment human workers rather than replacing them.

This project implements an **unsupervised anomaly detection** system using **PatchCore** — a feature-based method that extracts patch-level features from a pretrained WideResNet-50 and detects anomalies via nearest-neighbor distance. Trained on defect-free images only, it achieves up to **93% accuracy** and **0.99 AUC** on the MVTec AD dataset, with heatmaps that localize **where** the defect is.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Follow instructions in `data/README.md` to download the MVTec Anomaly Detection dataset (bottle, cable, transistor), then verify:

```bash
python data/download_dataset.py --verify --stats
```

### 3. Build PatchCore Memory Bank

No GPU training needed — just feature extraction (takes seconds):

```bash
python -m src.train --category bottle
python -m src.train --category cable
python -m src.train --category transistor
```

### 4. Evaluate

```bash
python -m src.evaluate --model models/best_bottle.pth --category bottle
python -m src.evaluate --model models/best_cable.pth --category cable
python -m src.evaluate --model models/best_transistor.pth --category transistor
```

### 5. Single Image Prediction

```bash
python -m src.predict --image path/to/image.png --model models/best_bottle.pth
```

### 6. Run the Streamlit App

```bash
streamlit run src/app.py
```

## How It Works

1. **Extract** patch-level features from good images using a pretrained WideResNet-50 (layers 2 & 3)
2. **Store** a memory bank of these normal patch features (coreset subsampling)
3. **Detect** anomalies by computing nearest-neighbor distance between test patches and the memory bank
4. **Localize** defects via a distance heatmap — high distance = anomaly
5. **Threshold** using mean + σ from validation scores

## Results

| Category   | Accuracy      | F1   | AUC  | Recall |
| ---------- | ------------- | ---- | ---- | ------ |
| Bottle     | **93%** | 0.95 | 0.99 | 98%    |
| Cable      | **78%** | 0.82 | 0.86 | 82%    |
| Transistor | **71%** | 0.69 | 0.80 | 80%    |

## Tech Stack

- **Deep Learning:** PyTorch (WideResNet-50 feature extraction)
- **Anomaly Detection:** PatchCore (nearest-neighbor on patch features)
- **Visualization:** matplotlib, seaborn, OpenCV (anomaly heatmaps)
- **App Prototype:** Streamlit
- **Dataset:** MVTec Anomaly Detection Dataset

## Project Structure

```
Product-Assembly-Error-Detection/
├── .gitignore
├── README.md
├── requirements.txt
├── computer_vision_industry5_project_guidelines.pdf
│
├── data/
│   ├── README.md              # Dataset documentation & citation
│   ├── download_dataset.py    # Dataset verification & stats script
│   └── raw/                   # MVTec AD images (not tracked by git)
│       ├── bottle/
│       ├── cable/
│       └── transistor/
│
├── src/
│   ├── __init__.py
│   ├── dataset.py             # PyTorch Dataset class for MVTec AD
│   ├── model.py               # PatchCore model (WideResNet-50 + memory bank)
│   ├── train.py               # Feature extraction & memory bank creation
│   ├── evaluate.py            # Evaluation metrics, ROC, confusion matrix
│   ├── visualize.py           # Anomaly heatmap generation
│   ├── predict.py             # Single image anomaly detection
│   └── app.py                 # Streamlit demo application
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── models/                    # Saved PatchCore memory banks (.pth)
├── results/                   # Evaluation outputs (plots, heatmaps)
│
├── report/
│   └── report_template.md
└── presentation/
    └── outline.md
```

## License

MIT License
