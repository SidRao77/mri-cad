"""
calibration.py

Makes the model's confidence scores mean what they say. A network trained
to ~100% train accuracy with cross-entropy keeps pushing its logits apart,
so its softmax outputs end up near 0 or 1 even on images it gets wrong.

Temperature scaling fixes this with a single learned number T: logits are
divided by T before the softmax (T > 1 softens overconfident outputs). T is
fit on the validation set by minimizing negative log-likelihood, and it
never changes which class is predicted — only how confident the model is.

Also provides expected calibration error (ECE) and a reliability diagram,
used by evaluate.py to check whether confidence matches accuracy.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

TEMPERATURE_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "temperature.json")


def collect_logits(model, loader, device):
    """Run the model over `loader` and return (logits, labels) as CPU tensors."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            all_logits.append(model(images.to(device)).cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits, labels) -> float:
    """Find the temperature T minimizing NLL of softmax(logits / T)."""
    # Optimize log(T) so T stays positive.
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_t.exp().item()


def save_temperature(temperature: float, path: str = TEMPERATURE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"temperature": temperature}, f)


def load_temperature(path: str = TEMPERATURE_PATH) -> float:
    """Load the fitted temperature, or 1.0 (no scaling) if none was saved."""
    if not os.path.exists(path):
        return 1.0
    with open(path) as f:
        return json.load(f)["temperature"]


def expected_calibration_error(confidences, correct, n_bins=5) -> float:
    """
    Weighted average gap between confidence and accuracy across confidence
    bins. 0 means perfectly calibrated. For a binary classifier the top-class
    confidence is always >= 0.5, so bins span [0.5, 1.0].
    """
    confidences, correct = np.asarray(confidences), np.asarray(correct, dtype=float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(confidences[in_bin].mean() - correct[in_bin].mean())
    return ece


def plot_reliability_diagram(confidences, correct, save_path, n_bins=5):
    """Save a reliability diagram: per-bin accuracy vs. mean confidence."""
    import matplotlib.pyplot as plt

    confidences, correct = np.asarray(confidences), np.asarray(correct, dtype=float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    centers, accs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.any():
            centers.append(confidences[in_bin].mean())
            accs.append(correct[in_bin].mean())
            counts.append(in_bin.sum())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0.5, 1], [0.5, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(centers, accs, marker="o", label="Model")
    for x, y, n in zip(centers, accs, counts):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram (test set, calibrated)")
    ax.legend(loc="lower right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    print(f"Saved reliability diagram to {save_path}")
