# Dataset — MVTec Anomaly Detection

## Source

- **Name:** MVTec Anomaly Detection Dataset (MVTec AD)
- **Website:** https://www.mvtec.com/company/research/datasets/mvtec-ad
- **Paper:** Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection," CVPR 2019
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

## Overview

MVTec AD contains ~5000 high-resolution images across 15 product categories. Each category has:
- **Training set:** Only "good" (defect-free) images
- **Test set:** Both "good" and various defect types
- **Ground truth:** Pixel-level segmentation masks for defective regions

## Focus Categories

This project focuses on **3 categories** selected for their variety of defect types:

| Category | Train (good) | Test (good) | Test (defective) | Defect Types |
|-----------|-------------|-------------|-------------------|--------------|
| **Bottle** | 209 | 20 | 63 | broken_large, broken_small, contamination |
| **Cable** | 224 | 58 | 92 | bent_wire, cable_swap, combined, cut_inner, cut_outer, missing_cable, missing_wire, poke_insulation |
| **Transistor** | 213 | 60 | 40 | bent_lead, cut_lead, damaged_case, misplaced |

## Directory Structure

After download and extraction, the data should be organized as:

```
data/raw/
├── bottle/
│   ├── train/
│   │   └── good/          # 209 defect-free images
│   ├── test/
│   │   ├── good/          # 20 defect-free test images
│   │   ├── broken_large/  # Defective test images
│   │   ├── broken_small/
│   │   └── contamination/
│   └── ground_truth/      # Pixel-level masks
├── cable/
│   ├── train/good/
│   ├── test/...
│   └── ground_truth/
└── transistor/
    ├── train/good/
    ├── test/...
    └── ground_truth/
```

## Download Instructions

1. Visit https://www.mvtec.com/company/research/datasets/mvtec-ad
2. Accept the license agreement
3. Download the dataset (~4.9 GB total, or individual categories)
4. Extract to `data/raw/`

Or use the helper script:
```bash
python data/download_dataset.py --verify --stats
```

## Citation

```bibtex
@inproceedings{bergmann2019mvtec,
  title={MVTec AD--A comprehensive real-world dataset for unsupervised anomaly detection},
  author={Bergmann, Paul and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9592--9600},
  year={2019}
}
```
