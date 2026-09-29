"""
evaluate.py

Loads the best saved checkpoint and evaluates it on the held-out test set:
  - Reports accuracy (with a bootstrap 95% CI — the test set is small),
    precision, recall, F1, and ROC-AUC.
  - Reports calibration — whether the model's confidence matches how often
    it's actually right — before and after temperature scaling: expected
    calibration error (ECE), negative log-likelihood, and Brier score.
  - Saves a confusion matrix, ROC curve, and reliability diagram to results/.
  - Prints a short summary emphasizing recall, since in this context a
    missed tumor (false negative) is more costly than a false alarm
    (false positive).

Run:
    python src/evaluate.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

from calibration import (
    collect_logits,
    expected_calibration_error,
    load_temperature,
    plot_reliability_diagram,
)
from dataset import get_dataloaders, set_seed
from model import build_model, get_device

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model.pt")
CONFUSION_MATRIX_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "confusion_matrix.png")
ROC_CURVE_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "roc_curve.png")
RELIABILITY_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "reliability_diagram.png")

CLASS_NAMES = ["no", "yes"]  # index 0 = no tumor, index 1 = tumor


def calibration_report(logits, labels, temperature):
    """Print ECE / NLL / Brier and the share of near-certain predictions."""
    probs = F.softmax(logits / temperature, dim=1)
    confidences, preds = probs.max(dim=1)
    correct = (preds == labels).numpy()
    confidences = confidences.numpy()

    ece = expected_calibration_error(confidences, correct)
    nll = F.cross_entropy(logits / temperature, labels).item()
    brier = ((probs[:, 1] - labels.float()) ** 2).mean().item()
    print(
        f"  T={temperature:.3f}: mean confidence {confidences.mean():.3f} vs accuracy "
        f"{correct.mean():.3f} | ECE {ece:.3f} | NLL {nll:.3f} | Brier {brier:.3f} | "
        f"{(confidences >= 0.99).mean():.0%} of predictions at >=99% confidence"
    )
    return confidences, correct


def bootstrap_accuracy_ci(correct, n_resamples=5000, seed=0):
    """95% bootstrap confidence interval for accuracy."""
    rng = np.random.default_rng(seed)
    correct = np.asarray(correct, dtype=float)
    resampled = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n_resamples)]
    return np.percentile(resampled, 2.5), np.percentile(resampled, 97.5)


def plot_confusion_matrix(labels, preds, save_path):
    """Save a simple annotated confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    # Write the raw counts inside each cell.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    print(f"Saved confusion matrix to {save_path}")


def plot_roc_curve(labels, probs, save_path):
    """Save an ROC curve plot with the diagonal no-skill baseline."""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    print(f"Saved ROC curve to {save_path}")


def main():
    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    test_loader = get_dataloaders()["test"]

    model = build_model().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    temperature = load_temperature()

    logits, label_tensor = collect_logits(model, test_loader, device)
    labels = label_tensor.tolist()
    preds = logits.argmax(dim=1).tolist()
    probs = F.softmax(logits / temperature, dim=1)[:, 1].tolist()  # P(tumor), calibrated

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    ci_low, ci_high = bootstrap_accuracy_ci(np.array(labels) == np.array(preds))

    print(f"\nTest set metrics (n={len(labels)}):")
    print(f"  Accuracy:  {accuracy:.4f}  (95% CI {ci_low:.3f}-{ci_high:.3f})")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")

    print("\nCalibration (does confidence match accuracy? lower ECE/NLL/Brier is better):")
    calibration_report(logits, label_tensor, 1.0)
    confidences, correct = calibration_report(logits, label_tensor, temperature)

    plot_confusion_matrix(labels, preds, CONFUSION_MATRIX_PATH)
    plot_roc_curve(labels, probs, ROC_CURVE_PATH)
    plot_reliability_diagram(confidences, correct, RELIABILITY_PATH)

    cm = confusion_matrix(labels, preds)
    false_negatives = cm[1, 0]  # actual tumor, predicted no tumor
    print(
        "\nSummary: In this setting, recall matters most — a false negative "
        "(missing a real tumor) is far more costly than a false positive "
        "(an unnecessary follow-up scan). "
        f"This model's recall on the test set is {recall:.1%}, meaning it "
        f"missed {false_negatives} of {sum(cm[1])} actual tumor cases. "
        "Precision and F1 are reported for completeness, but recall (and "
        "ROC-AUC as a threshold-independent view) are the metrics to watch "
        "when judging whether this model is good enough to build on."
    )


if __name__ == "__main__":
    main()
