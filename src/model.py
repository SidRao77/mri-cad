"""
model.py

Builds the ResNet18 model used for binary tumor classification.

We start from ImageNet-pretrained weights (the network already knows how
to detect general-purpose visual features like edges and textures) and
replace the final fully-connected layer so it outputs 2 class scores
(no-tumor, tumor) instead of ImageNet's 1000 classes.

Given how small our dataset is (253 images), fine-tuning the whole network
rather than freezing early layers still works well here because we're
starting from strong pretrained weights and using a low learning rate —
freezing is more valuable when the dataset is even smaller or the target
domain is very different from natural images.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

NUM_CLASSES = 2


def get_device() -> torch.device:
    """
    Pick the best available device on this machine.

    Apple Silicon Macs use PyTorch's "mps" backend instead of CUDA. We fall
    back to CPU if MPS isn't available (e.g. running this on a non-Mac).
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Load a pretrained ResNet18 and swap its final layer for our 2-class head.

    ResNet18's original final layer (`fc`) maps its 512-dimensional feature
    vector to 1000 ImageNet classes. We replace it with a fresh linear layer
    mapping to `num_classes` instead. This new layer starts with random
    weights and gets trained along with the rest of the (already pretrained)
    network.
    """
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


if __name__ == "__main__":
    # Quick sanity check: build the model, move it to the right device, and
    # run one dummy batch through it to confirm the output shape is right.
    device = get_device()
    print(f"Using device: {device}")

    model = build_model().to(device)
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"Output shape: {tuple(output.shape)}  (expected: (4, {NUM_CLASSES}))")
