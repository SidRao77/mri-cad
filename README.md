# MRI-CAD — Cranial Tumor Screening

A small project that fine-tunes a pretrained ResNet18 to classify brain MRI
scans as "tumor" or "no tumor," evaluates it with clinically meaningful
metrics, and serves predictions through a local Streamlit demo styled as a
computer-aided detection (CAD) report.

> **⚠️ Disclaimer: This is a learning/portfolio project, not a diagnostic
> tool.** It is trained on a small public dataset, has not been validated
> for clinical use, and must not be used to inform any real medical decision.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python download_data.py       # download the dataset into data/raw/
python src/dataset.py         # sanity-check the data pipeline
python src/train.py           # train the model, save best checkpoint + calibration to checkpoints/
python src/evaluate.py        # evaluate on the held-out test set, save plots to results/
python src/cross_validate.py  # 5-fold cross-validated accuracy/calibration estimate
streamlit run app.py          # run the local demo
```

## Dataset

[Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
(Kaggle, `navoneel/brain-mri-images-for-brain-tumor-detection`).

## Results

5-fold stratified group cross-validation (each image held out exactly once):

| Metric   | Mean ± std    |
|----------|---------------|
| Accuracy | 0.898 ± 0.065 |
| Recall   | 0.913 ± 0.076 |
| ROC-AUC  | 0.954 ± 0.033 |
| ECE      | 0.086 ± 0.034 |

On the single held-out test split (n=37): accuracy 0.919 (95% bootstrap CI
0.81–1.00), recall 1.00, ROC-AUC 0.96. The wide interval reflects how small
the test set is — the cross-validated figures are the better estimate.

## Notes on evaluation

- **Duplicate-aware splitting.** The dataset contains many duplicate scans
  saved under different filenames (22 byte-identical groups plus re-encoded
  copies). Near-duplicates are grouped with a perceptual hash and kept
  within a single split. Without this, ~25% of test images had a copy in
  the training set, inflating both accuracy and confidence.
- **Confidence calibration.** An earlier version reported ~100% confidence
  on most scans, including a missed tumor scored at 99.99%. Training now
  uses label smoothing, weight decay, and early stopping on validation
  loss, and outputs are calibrated with temperature scaling. Calibration
  is measured with expected calibration error (ECE) and a reliability
  diagram.
- **Honest display.** The demo caps its confidence readout at >99% and
  flags readings below 75% as low-confidence, since a model trained on
  ~250 images can't support more precise claims.

## Project structure

```
brain-tumor-classifier/
├── data/raw/          gitignored — dataset images
├── src/
│   ├── dataset.py      loading, stratified split, transforms
│   ├── model.py         ResNet18 setup
│   ├── train.py          training loop
│   ├── evaluate.py       test set metrics, calibration + plots
│   ├── calibration.py    temperature scaling, ECE, reliability diagram
│   ├── cross_validate.py 5-fold cross-validation
│   └── gradcam.py        Grad-CAM visualization
├── checkpoints/        gitignored — saved model weights
├── results/             gitignored — saved plots
└── app.py               Streamlit demo
```
