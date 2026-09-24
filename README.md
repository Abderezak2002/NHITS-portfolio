# N-HiTS Time Series Forecasting

A PyTorch Lightning implementation of the **N-HiTS** architecture for 
multi-horizon time series forecasting, featuring a custom model design, 
a full training/evaluation pipeline, and configurable windowed data loaders.

> **Project context:** This project was developed collaboratively as part of 
> Advanced Machine Learning Course at Sorbonne University, together with 
> [@aminekhelif](https://github.com/aminekhelif). You can find the original 
> shared repository [here](https://github.com/aminekhelif/N-HITS). 
> This repository has been restructured and documented to highlight my 
> specific contributions.

---

## My Contributions

- **Model architecture** — Designed and implemented the core N-HiTS 
  block structure (`models/nhits/nhits.py`), including stack configuration, 
  pooling strategy, and residual hierarchical forecasting logic.
- **Training & evaluation pipeline** — Built the PyTorch Lightning training 
  loop, checkpointing, early stopping, and evaluation/metrics scripts.
- **Data loaders** — Hleped design and implement the custom windowed 
  `(X, Y)` dataset classes and preprocessing pipeline for arbitrary 
  time series CSVs (multi-horizon, multi-frequency support).
- **Project coordination** — Managed task planning and integration 
  between components during development.

Project logistics and initial exploratory setup were shared with my 
teammate [@aminekhelif](https://github.com/aminekhelif).

---

## Overview

N-HiTS (Neural Hierarchical Interpolation for Time Series) tackles 
long-horizon forecasting by stacking multi-block modules that each 
model a portion of the residual signal at a different resolution, 
using linear interpolation/upsampling to combine them into a final 
forecast. This implementation supports configurable stack types, 
pooling modes, dropout, and multi-frequency inputs.

## Features

- Modular N-HiTS blocks with configurable stacks, pooling, and dropout
- PyTorch Lightning training loop with checkpointing and early stopping
- Windowed data preprocessing for arbitrary-frequency time series
- CSV merging/normalization utilities for custom datasets

## Requirements

- Python 3.10+
- PyTorch 2.5.1 · PyTorch Lightning 2.5.0 · NumPy 2.2.1 · Pandas 2.2.3 
  · scikit-learn 1.4.2 · matplotlib 3.8.4

## Installation

\`\`\`bash
git clone https://github.com/<your-username>/N-HITS.git
cd N-HITS
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
\`\`\`

## Project Structure

\`\`\`
N-HITS/
├── data/               # Preprocessing, dataloaders, merging utilities
├── models/nhits/        # Core N-HiTS architecture (LightningModule)
├── main.py
├── requirements.txt
└── README.md
\`\`\`

## Usage

**1. Preprocess data**
\`\`\`bash
python data/preprocess_data.py \
    --file_path data/daily.csv \
    --output_folder data/processed_daily \
    --h_values 96 192 336 720 \
    --multiplier 5 --train_split 0.7 --validation_split 0.1
\`\`\`

**2. Train**
\`\`\`bash
python main.py --train_csv data/processed_daily/H=96/train.csv \
    --val_csv data/processed_daily/H=96/validation.csv \
    --epochs 30 --batch_size 256
\`\`\`

**3. Evaluate / Inference**
\`\`\`bash
python main.py --mode inference \
    --checkpoint ./checkpoints/nhits-best.ckpt \
    --test_csv data/processed_daily/H=96/test.csv
\`\`\`

## Results
*(Add: forecast plots, error metrics table — MAE/MSE/MAPE by horizon — 
this section matters most to recruiters, add it if you have the data.)*

## Acknowledgments

Developed with [@aminekhelif](https://github.com/aminekhelif) as part of 
Advanced Machine Learning Course, Sorbonne University, 2024-2025.