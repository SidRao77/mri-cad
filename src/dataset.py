"""
dataset.py

Handles everything related to loading the brain MRI dataset:
  1. Finding the image files on disk (auto-detecting how the Kaggle
     download landed under data/raw/ — flat or nested a level deeper).
  2. Splitting them into stratified train/val/test sets (70/15/15).
  3. Wrapping them in a PyTorch Dataset + DataLoader with the right
     transforms: resize to 224x224, ImageNet normalization, and light
     augmentation for the training set only.

Run this file directly to sanity-check the pipeline:
    python src/dataset.py
It prints the folder structure it found under data/raw/, the per-class
image counts, and the resulting split sizes.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed every source of randomness we use, so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Locating images on disk
# ---------------------------------------------------------------------------
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"

# The two class folder names we expect to find, mapped to integer labels.
# "yes" = tumor present, "no" = no tumor.
CLASS_TO_LABEL = {"no": 0, "yes": 1}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_class_folders(data_root: Path) -> dict:
    """
    Walk data_root looking for folders literally named "yes" and "no".

    The Kaggle dataset sometimes lands flat (data/raw/yes, data/raw/no)
    and sometimes nested a level deeper (data/raw/brain_tumor_dataset/yes),
    depending on how kagglehub packages it. Rather than hardcode a path,
    we search for the folders wherever they are so this keeps working
    either way.
    """
    found = {}
    for dirpath, dirnames, _ in os.walk(data_root):
        for dirname in dirnames:
            if dirname.lower() in CLASS_TO_LABEL and dirname.lower() not in found:
                found[dirname.lower()] = Path(dirpath) / dirname
    return found


def print_folder_structure(data_root: Path, max_depth: int = 3) -> None:
    """Print the directory tree under data_root, for eyeballing structure."""
    print(f"Folder structure under {data_root}:")
    if not data_root.exists():
        print("  (does not exist — run download_data.py first)")
        return
    for dirpath, dirnames, filenames in os.walk(data_root):
        depth = len(Path(dirpath).relative_to(data_root).parts)
        if depth > max_depth:
            dirnames[:] = []  # don't descend further
            continue
        indent = "  " * depth
        print(f"{indent}{Path(dirpath).name}/  ({len(filenames)} files)")


def collect_image_paths(data_root: Path = DATA_ROOT):
    """
    Find the yes/no class folders under data_root and return a list of
    (path, label) pairs for every image file inside them, plus a dict
    of per-class counts.
    """
    class_folders = find_class_folders(data_root)
    missing = set(CLASS_TO_LABEL) - set(class_folders)
    if missing:
        raise FileNotFoundError(
            f"Could not find folder(s) named {sorted(missing)} anywhere under "
            f"{data_root}. Run download_data.py first, or check that the "
            f"dataset was extracted correctly."
        )

    samples = []
    counts = {}
    for class_name, label in CLASS_TO_LABEL.items():
        folder = class_folders[class_name]
        paths = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        counts[class_name] = len(paths)
        samples.extend((p, label) for p in paths)

    return samples, counts


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

# Training transform: resize + light augmentation + normalize.
# Augmentation is kept mild on purpose — MRI scans have a fixed, meaningful
# orientation, so we avoid aggressive flips/rotations that could distort
# anatomy in ways that don't occur in real scans.
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Validation/test transform: resize + normalize only, no augmentation.
# We evaluate on the image as-is so metrics reflect real-world performance.
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BrainMRIDataset(Dataset):
    """A simple Dataset over a list of (image_path, label) pairs."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # MRI images may be grayscale; convert to RGB since ResNet18 expects
        # 3 input channels (it was pretrained on RGB ImageNet images).
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------
def stratified_split(samples, train_frac=0.7, val_frac=0.15, seed=SEED):
    """
    Split (path, label) samples into train/val/test lists, preserving the
    class balance in each split (stratified sampling), using a fixed seed
    for reproducibility.
    """
    labels = [label for _, label in samples]

    # First split off the training set...
    train_samples, remainder = train_test_split(
        samples,
        train_size=train_frac,
        stratify=labels,
        random_state=seed,
    )

    # ...then split what's left evenly into val/test. val_frac was a
    # fraction of the *original* total, so we convert it to a fraction
    # of the remainder before splitting again.
    remainder_labels = [label for _, label in remainder]
    relative_val_frac = val_frac / (1 - train_frac)
    val_samples, test_samples = train_test_split(
        remainder,
        train_size=relative_val_frac,
        stratify=remainder_labels,
        random_state=seed,
    )

    return train_samples, val_samples, test_samples


# ---------------------------------------------------------------------------
# Public entry point used by train.py / evaluate.py
# ---------------------------------------------------------------------------
def get_dataloaders(batch_size=16, num_workers=0):
    """
    Build train/val/test DataLoaders from data/raw/.

    Returns a dict of DataLoaders keyed by "train", "val", "test".
    """
    set_seed()
    samples, _ = collect_image_paths()
    train_samples, val_samples, test_samples = stratified_split(samples)

    train_ds = BrainMRIDataset(train_samples, transform=train_transform)
    val_ds = BrainMRIDataset(val_samples, transform=eval_transform)
    test_ds = BrainMRIDataset(test_samples, transform=eval_transform)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return loaders


# ---------------------------------------------------------------------------
# Sanity check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    set_seed()

    print_folder_structure(DATA_ROOT)
    print()

    samples, counts = collect_image_paths()
    print("Class counts:")
    for class_name, count in counts.items():
        print(f"  {class_name}: {count} images")
    print(f"  total: {len(samples)} images")
    print()

    train_samples, val_samples, test_samples = stratified_split(samples)
    print("Stratified split sizes:")
    for name, split in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        split_labels = [label for _, label in split]
        n_yes = sum(split_labels)
        n_no = len(split_labels) - n_yes
        print(f"  {name}: {len(split)} images  (no={n_no}, yes={n_yes})")
    print()

    # Load a single batch to confirm the DataLoader/transform pipeline works.
    loaders = get_dataloaders(batch_size=8)
    images, labels = next(iter(loaders["train"]))
    print(f"Sample train batch: images.shape={tuple(images.shape)}, labels={labels.tolist()}")
