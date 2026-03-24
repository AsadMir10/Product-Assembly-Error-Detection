"""MVTec Anomaly Detection Dataset download and organization script.

The MVTec AD dataset requires manual download due to license agreement.
This script helps organize the downloaded data into the expected structure.

Dataset: https://www.mvtec.com/company/research/datasets/mvtec-ad
"""

import argparse
import os
import shutil
import tarfile
from pathlib import Path


MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

FOCUS_CATEGORIES = ["bottle", "cable", "transistor"]

DOWNLOAD_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════╗
║           MVTec AD Dataset — Download Instructions          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Visit: https://www.mvtec.com/company/research/          ║
║           datasets/mvtec-ad                                  ║
║                                                              ║
║  2. Accept the license agreement                             ║
║                                                              ║
║  3. Download the full dataset (~4.9 GB) or individual        ║
║     category archives                                        ║
║                                                              ║
║  4. Extract to: data/raw/                                    ║
║                                                              ║
║  Expected structure after extraction:                        ║
║     data/raw/                                                ║
║       ├── bottle/                                            ║
║       │   ├── train/                                         ║
║       │   │   └── good/                                      ║
║       │   ├── test/                                          ║
║       │   │   ├── good/                                      ║
║       │   │   ├── broken_large/                              ║
║       │   │   ├── broken_small/                              ║
║       │   │   └── contamination/                             ║
║       │   └── ground_truth/                                  ║
║       ├── cable/                                             ║
║       └── transistor/                                        ║
║                                                              ║
║  For this project, we focus on: bottle, cable, transistor    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


def verify_dataset(data_dir: str, categories: list[str] | None = None) -> bool:
    """Verify that the dataset is properly organized.

    Args:
        data_dir: Path to the dataset root directory.
        categories: List of categories to verify. Defaults to focus categories.

    Returns:
        True if all expected directories exist, False otherwise.
    """
    if categories is None:
        categories = FOCUS_CATEGORIES

    data_path = Path(data_dir)
    all_valid = True

    for category in categories:
        cat_dir = data_path / category
        train_good = cat_dir / "train" / "good"
        test_good = cat_dir / "test" / "good"

        if not cat_dir.exists():
            print(f"  [MISSING] {cat_dir}")
            all_valid = False
            continue

        if not train_good.exists():
            print(f"  [MISSING] {train_good}")
            all_valid = False
        else:
            n_train = len(list(train_good.glob("*.png")))
            print(f"  [OK] {category}/train/good: {n_train} images")

        if not test_good.exists():
            print(f"  [MISSING] {test_good}")
            all_valid = False
        else:
            # Count test images across all subdirs
            test_dir = cat_dir / "test"
            n_test = 0
            for subdir in sorted(test_dir.iterdir()):
                if subdir.is_dir():
                    n_imgs = len(list(subdir.glob("*.png")))
                    n_test += n_imgs
                    defect_label = "good" if subdir.name == "good" else "defect"
                    print(f"  [OK] {category}/test/{subdir.name}: {n_imgs} images ({defect_label})")

    return all_valid


def extract_archive(archive_path: str, extract_to: str) -> None:
    """Extract a tar.xz archive.

    Args:
        archive_path: Path to the archive file.
        extract_to: Directory to extract to.
    """
    print(f"Extracting {archive_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)

    with tarfile.open(archive_path, "r:xz") as tar:
        tar.extractall(path=extract_to)

    print("Extraction complete.")


def print_dataset_stats(data_dir: str, categories: list[str] | None = None) -> None:
    """Print statistics about the dataset.

    Args:
        data_dir: Path to the dataset root directory.
        categories: Categories to report on.
    """
    if categories is None:
        categories = FOCUS_CATEGORIES

    data_path = Path(data_dir)

    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)

    for category in categories:
        cat_dir = data_path / category
        if not cat_dir.exists():
            continue

        print(f"\n{category.upper()}")
        print("-" * 30)

        # Training images
        train_dir = cat_dir / "train" / "good"
        if train_dir.exists():
            n_train = len(list(train_dir.glob("*.png")))
            print(f"  Train (good only): {n_train}")

        # Test images
        test_dir = cat_dir / "test"
        if test_dir.exists():
            total_test = 0
            for subdir in sorted(test_dir.iterdir()):
                if subdir.is_dir():
                    n = len(list(subdir.glob("*.png")))
                    total_test += n
                    label = "good" if subdir.name == "good" else subdir.name
                    print(f"  Test ({label}): {n}")
            print(f"  Test total: {total_test}")

    print()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MVTec AD Dataset setup and verification"
    )
    parser.add_argument("--data_dir", type=str, default="data/raw",
                        help="Target directory for the dataset")
    parser.add_argument("--extract", type=str, default=None,
                        help="Path to a downloaded tar.xz archive to extract")
    parser.add_argument("--verify", action="store_true",
                        help="Verify the dataset structure")
    parser.add_argument("--stats", action="store_true",
                        help="Print dataset statistics")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Categories to process (default: bottle, cable, transistor)")
    args = parser.parse_args()

    categories = args.categories or FOCUS_CATEGORIES

    if args.extract:
        extract_archive(args.extract, args.data_dir)

    if args.verify or args.stats:
        if args.verify:
            print("\nVerifying dataset structure...")
            valid = verify_dataset(args.data_dir, categories)
            if valid:
                print("\nAll categories verified successfully!")
            else:
                print("\nSome categories are missing. See above for details.")

        if args.stats:
            print_dataset_stats(args.data_dir, categories)
    else:
        # Default: print download instructions and verify
        print(DOWNLOAD_INSTRUCTIONS)
        data_path = Path(args.data_dir)
        if data_path.exists() and any(data_path.iterdir()):
            print("Existing data found. Verifying...")
            verify_dataset(args.data_dir, categories)
            print_dataset_stats(args.data_dir, categories)
        else:
            print(f"No data found in {args.data_dir}/")
            print("Follow the instructions above to download the dataset.")


if __name__ == "__main__":
    main()
