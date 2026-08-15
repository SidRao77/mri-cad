"""
gradcam.py

Grad-CAM (Gradient-weighted Class Activation Mapping) visualization.

Given an input image and a trained model, Grad-CAM highlights which
regions of the image most influenced the model's prediction, by looking
at how strongly the gradients flowing back from the predicted class
activate each spatial location in the last convolutional layer.

This is purely a visualization tool — it doesn't change how the model
makes its prediction. It's built as a class (GradCAM) so app.py can reuse
it: build once, then call generate() on whatever image the user uploads.

Run this file directly to generate an example heatmap for one test image:
    python src/gradcam.py
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.cm as cm
from PIL import Image

from dataset import collect_image_paths, stratified_split, eval_transform, set_seed
from model import build_model, get_device

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model.pt")
EXAMPLE_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "gradcam_example.png")


class GradCAM:
    """
    Computes Grad-CAM heatmaps for a model, hooked onto one target layer.

    For ResNet18, the natural target layer is `model.layer4[-1]` — the
    last residual block, right before global average pooling. Its output
    is the last feature map that still has spatial structure (7x7 for a
    224x224 input), which is what lets us produce a heatmap instead of
    just a single number.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Run a forward + backward pass and produce a Grad-CAM heatmap.

        input_tensor: a single preprocessed image, shape (1, 3, 224, 224).
        target_class: which class to explain (0=no tumor, 1=tumor). If
            None, explains the model's own predicted class.

        Returns: (heatmap, predicted_class, confidence)
            heatmap is a (224, 224) numpy array normalized to [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)  # (1, num_classes)
        probs = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probs[0, predicted_class].item()

        if target_class is None:
            target_class = predicted_class

        # Backprop only the score for the class we want to explain, so the
        # resulting gradients tell us what the model looked at *for that
        # class specifically*.
        score = output[0, target_class]
        score.backward()

        # Global-average-pool the gradients over spatial dims to get one
        # importance weight per channel, then take a weighted sum of the
        # activation maps using those weights.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        weighted_activations = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        heatmap = F.relu(weighted_activations)  # keep only features that positively support the class

        # Upsample from the feature map's small spatial size (7x7) up to
        # the original 224x224 image size.
        heatmap = F.interpolate(heatmap, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        heatmap = heatmap.squeeze().cpu().numpy()

        # Normalize to [0, 1] for visualization.
        heatmap -= heatmap.min()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap, predicted_class, confidence


def overlay_heatmap(original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """
    Blend a Grad-CAM heatmap on top of the original image for display.

    original_image: PIL image (any size — resized to match the heatmap).
    heatmap: (H, W) numpy array normalized to [0, 1].
    alpha: how much the heatmap contributes vs. the original image (0-1).
    """
    original_resized = original_image.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))
    original_array = np.array(original_resized).astype(np.float32) / 255.0

    # Map the single-channel heatmap through a colormap (red = high
    # importance, blue = low) to turn it into an RGB image.
    colored_heatmap = cm.jet(heatmap)[:, :, :3]  # drop the colormap's alpha channel

    blended = (1 - alpha) * original_array + alpha * colored_heatmap
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def build_gradcam(model) -> GradCAM:
    """Convenience constructor: wire up a GradCAM on ResNet18's last conv block."""
    return GradCAM(model, target_layer=model.layer4[-1])


if __name__ == "__main__":
    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    model = build_model().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    # Grab one image from the test split to demo on (same split train.py
    # never saw, so this is a fair example of real model behavior).
    samples, _ = collect_image_paths()
    _, _, test_samples = stratified_split(samples)
    image_path, true_label = test_samples[0]
    print(f"Demo image: {image_path} (true label: {'yes' if true_label == 1 else 'no'})")

    original_image = Image.open(image_path).convert("RGB")
    input_tensor = eval_transform(original_image).unsqueeze(0).to(device)

    gradcam = build_gradcam(model)
    heatmap, predicted_class, confidence = gradcam.generate(input_tensor)

    label_name = "yes" if predicted_class == 1 else "no"
    print(f"Predicted: {label_name} (confidence: {confidence:.4f})")

    overlay = overlay_heatmap(original_image, heatmap)
    os.makedirs(os.path.dirname(EXAMPLE_OUTPUT_PATH), exist_ok=True)
    overlay.save(EXAMPLE_OUTPUT_PATH)
    print(f"Saved Grad-CAM overlay to {EXAMPLE_OUTPUT_PATH}")
