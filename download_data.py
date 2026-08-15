"""
download_data.py

Downloads the "Brain MRI Images for Brain Tumor Detection" dataset from
Kaggle via kagglehub, and copies it into data/raw/ inside this project.

Run once, before anything else:
    python download_data.py
"""

import kagglehub
import shutil
import os

path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")
print("Downloaded to:", path)

os.makedirs("data/raw", exist_ok=True)
shutil.copytree(path, "data/raw", dirs_exist_ok=True)
print("Copied into data/raw")
