"""
train.py

Trains the ResNet18 model on the brain MRI dataset:
  - Loads train/val data via dataset.py.
  - Uses class-weighted cross-entropy loss to compensate for the ~1.6:1
    class imbalance (155 "yes" vs 98 "no" images) found in the dataset —
    this keeps the model from getting an easy win by leaning toward the
    majority class.
  - Uses label smoothing and weight decay to discourage the extreme,
    near-100% softmax outputs a small dataset otherwise produces once the
    model has memorized the training set.
  - Trains with Adam, tracking train/val loss and accuracy per epoch.
  - Saves the checkpoint with the lowest validation loss to checkpoints/,
    stopping early once val loss hasn't improved for a few epochs. Val loss
    (unlike accuracy) penalizes overconfident mistakes, so it's the better
    signal for when the model starts overfitting.
  - Fits a temperature-scaling calibration on the validation set and saves
    it alongside the checkpoint (see calibration.py).
  - Saves a loss/accuracy curve plot to results/.

Run:
    python src/train.py
"""

import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # write plots straight to file, no GUI window needed
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from calibration import collect_logits, fit_temperature, save_temperature
from dataset import get_dataloaders, set_seed
from model import build_model, get_device

NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 5  # epochs without val loss improvement before stopping

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model.pt")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "training_curves.png")


def compute_class_weights(loader, num_classes=2, device="cpu"):
    """
    Compute inverse-frequency class weights from the training set, for use
    in a weighted cross-entropy loss. Rarer classes get a higher weight so
    misclassifying them costs more, which pushes the model to pay attention
    to the minority class instead of just predicting the majority class.
    """
    label_counts = Counter()
    for _, labels in loader:
        label_counts.update(labels.tolist())

    total = sum(label_counts.values())
    weights = [total / (num_classes * label_counts[c]) for c in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32).to(device)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """
    Run one pass over `loader` — either training (updates model weights)
    or evaluating (no gradient updates, depending on `train`). Returns the
    average loss and accuracy for the epoch.
    """
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # Only track gradients when training — saves memory/compute during eval.
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def plot_curves(history, save_path):
    """Save a two-panel plot of loss and accuracy curves over training."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    print(f"Saved training curves to {save_path}")


def main():
    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    loaders = get_dataloaders(batch_size=BATCH_SIZE)
    train_loader, val_loader = loaders["train"], loaders["val"]

    class_weights = compute_class_weights(train_loader, device=device)
    print(f"Class weights (no, yes): {class_weights.tolist()}")

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> New best val_loss={val_loss:.4f}, saved checkpoint to {CHECKPOINT_PATH}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"  -> No val loss improvement for {EARLY_STOPPING_PATIENCE} epochs, stopping early.")
                break

    plot_curves(history, RESULTS_PATH)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    # Calibrate the best checkpoint's confidence on the validation set.
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    val_logits, val_labels = collect_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    save_temperature(temperature)
    print(f"Fitted calibration temperature T={temperature:.3f} (saved to checkpoints/temperature.json)")


if __name__ == "__main__":
    main()
