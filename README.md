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
python src/train.py           # train the model, save best checkpoint to checkpoints/
python src/evaluate.py        # evaluate on the held-out test set, save plots to results/
streamlit run app.py          # run the local demo
```

## Dataset

[Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
(Kaggle, `navoneel/brain-mri-images-for-brain-tumor-detection`).

## Project structure

```
brain-tumor-classifier/
├── data/raw/          gitignored — dataset images
├── src/
│   ├── dataset.py      loading, stratified split, transforms
│   ├── model.py         ResNet18 setup
│   ├── train.py          training loop
│   ├── evaluate.py       test set metrics + plots
│   └── gradcam.py        Grad-CAM visualization
├── checkpoints/        gitignored — saved model weights
├── results/             gitignored — saved plots
└── app.py               Streamlit demo
```
