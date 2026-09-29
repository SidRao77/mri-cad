"""
cross_validate.py

Estimates how well the training recipe generalizes using 5-fold stratified
group cross-validation. With only ~250 images, a single ~37-image test set
gives a noisy estimate (one extra mistake moves accuracy by ~3 points), so
here every image is held out exactly once and we report mean ± std across
folds instead.

Each fold follows the same recipe as train.py: near-duplicate images stay
in the same fold, a validation set is carved out of the training portion
for early stopping and temperature fitting, and the fold's held-out images
are only used for the final metrics.

This does not save a checkpoint — use train.py for the model the app uses.

Run:
    python src/cross_validate.py
"""

import copy

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from calibration import collect_logits, expected_calibration_error, fit_temperature
from dataset import (
    SEED,
    BrainMRIDataset,
    group_fold,
    collect_image_paths,
    duplicate_groups,
    eval_transform,
    set_seed,
    train_transform,
)
from model import build_model, get_device
from train import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    compute_class_weights,
    run_epoch,
)

N_FOLDS = 5


def train_fold(train_samples, val_samples, device):
    """Train one model with the train.py recipe; return it and its fitted temperature."""
    train_loader = DataLoader(BrainMRIDataset(train_samples, train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(BrainMRIDataset(val_samples, eval_transform), batch_size=BATCH_SIZE)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(train_loader, device=device), label_smoothing=LABEL_SMOOTHING
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_loss, best_state, stale_epochs = float("inf"), None, 0
    for _ in range(NUM_EPOCHS):
        run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, _ = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        if val_loss < best_val_loss:
            best_val_loss, best_state, stale_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale_epochs += 1
            if stale_epochs >= EARLY_STOPPING_PATIENCE:
                break

    model.load_state_dict(best_state)
    temperature = fit_temperature(*collect_logits(model, val_loader, device))
    return model, temperature


def main():
    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    samples, _ = collect_image_paths()
    groups = duplicate_groups(samples)
    labels = np.array([label for _, label in samples])

    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []
    for fold, (dev_idx, test_idx) in enumerate(splitter.split(np.zeros(len(samples)), labels, groups), 1):
        train_sub, val_sub = group_fold(labels[dev_idx], groups[dev_idx], round(1 / 0.15), SEED)
        pick = lambda idx: [samples[i] for i in idx]

        model, temperature = train_fold(pick(dev_idx[train_sub]), pick(dev_idx[val_sub]), device)

        test_loader = DataLoader(BrainMRIDataset(pick(test_idx), eval_transform), batch_size=BATCH_SIZE)
        logits, y = collect_logits(model, test_loader, device)
        probs = F.softmax(logits / temperature, dim=1)
        confidences, preds = probs.max(dim=1)
        correct = (preds == y).numpy()
        raw_confidences = F.softmax(logits, dim=1).max(dim=1).values

        metrics = {
            "accuracy": correct.mean(),
            "recall": recall_score(y, preds),
            "roc_auc": roc_auc_score(y, probs[:, 1]),
            "ece": expected_calibration_error(confidences.numpy(), correct),
            "mean_confidence": confidences.mean().item(),
            "ece_uncalibrated": expected_calibration_error(raw_confidences.numpy(), correct),
            "mean_conf_uncal": raw_confidences.mean().item(),
        }
        fold_metrics.append(metrics)
        print(
            f"Fold {fold}/{N_FOLDS} (n={len(test_idx)}, T={temperature:.2f}): "
            + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        )

    print(f"\n{N_FOLDS}-fold cross-validation (mean ± std):")
    for key in fold_metrics[0]:
        values = np.array([m[key] for m in fold_metrics])
        print(f"  {key:16s} {values.mean():.3f} ± {values.std():.3f}")


if __name__ == "__main__":
    main()
