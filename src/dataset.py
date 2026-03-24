"""PyTorch Dataset class for MVTec Anomaly Detection dataset.

For anomaly detection: training uses only 'good' images (no labels needed).
Test set has both good and defective images with binary labels.
Images are normalized with ImageNet stats for pretrained ResNet feature extraction.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Training transforms with augmentation + ImageNet normalization."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Evaluation transforms (no augmentation) + ImageNet normalization."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class MVTecDataset(Dataset):
    """PyTorch Dataset for MVTec Anomaly Detection data.

    Expects the standard MVTec AD directory structure:
        data_root/
            <category>/
                train/
                    good/
                test/
                    good/
                    <defect_type_1>/
                    ...

    Labels:
        0 = good (no defect)
        1 = defective (any defect type)
    """

    def __init__(
        self,
        data_root: str,
        category: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        image_size: int = 224,
    ) -> None:
        """Initialize the MVTec dataset.

        Args:
            data_root: Path to the root MVTec AD directory.
            category: Product category (e.g., 'bottle', 'cable', 'transistor'),
                      or 'all' to load all available categories.
            split: 'train' or 'test'.
            transform: Optional custom transform. If None, uses default transforms.
            image_size: Target image size for default transforms.
        """
        self.data_root = Path(data_root)
        self.category = category
        self.split = split
        self.image_size = image_size

        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_eval_transforms(image_size)

        self.image_paths: list[str] = []
        self.labels: list[int] = []
        self.defect_types: list[str] = []

        if category == "all":
            for cat in sorted(self.data_root.iterdir()):
                if cat.is_dir() and (cat / split).exists():
                    self._load_samples(cat.name)
        else:
            self._load_samples(category)

    def _load_samples(self, category: Optional[str] = None) -> None:
        """Scan the directory structure and collect image paths and labels."""
        cat = category or self.category
        split_dir = self.data_root / cat / self.split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Directory not found: {split_dir}. "
                f"Make sure the MVTec AD dataset is downloaded and extracted."
            )

        for defect_dir in sorted(split_dir.iterdir()):
            if not defect_dir.is_dir():
                continue

            defect_type = defect_dir.name
            label = 0 if defect_type == "good" else 1

            for img_path in sorted(defect_dir.iterdir()):
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                    self.image_paths.append(str(img_path))
                    self.labels.append(label)
                    self.defect_types.append(defect_type)

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found in {split_dir}. "
                f"Check that the dataset is properly organized."
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label).
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label

    def get_class_counts(self) -> dict[str, int]:
        """Return count of good vs defective samples."""
        return {
            "good": self.labels.count(0),
            "defective": self.labels.count(1),
        }


def create_train_val_split(
    data_root: str,
    category: str,
    val_ratio: float = 0.2,
    image_size: int = 224,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """Split the 'good' training images into train and validation sets.

    Args:
        data_root: Path to the root MVTec AD directory.
        category: Product category or 'all'.
        val_ratio: Fraction for validation.
        image_size: Target image size.
        seed: Random seed.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    # Load all good images with eval transforms first
    full_dataset = MVTecDataset(
        data_root=data_root,
        category=category,
        split="train",
        transform=get_eval_transforms(image_size),
        image_size=image_size,
    )

    total = len(full_dataset)
    val_size = int(total * val_ratio)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # Apply training augmentations to training subset
    train_dataset = _TransformSubset(train_subset, get_train_transforms(image_size))
    val_dataset = val_subset  # Keep eval transforms

    return train_dataset, val_dataset


class _TransformSubset(Dataset):
    """Wrapper that applies a different transform to a Subset."""

    def __init__(self, subset: torch.utils.data.Subset, transform: transforms.Compose) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.subset.indices[idx]
        dataset = self.subset.dataset
        img_path = dataset.image_paths[real_idx]
        label = dataset.labels[real_idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label
