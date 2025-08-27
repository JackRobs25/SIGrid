# SIGrid: SUPERPIXEL INTEGRATED GRIDS FOR FAST IMAGE SEGMENTATION

This repository contains the official code for experiments on **SIGrid**, a hybrid pixel–superpixel input structure designed to enhance deep learning models for image segmentation.  
The project builds upon U-Net and FCN backbones, integrating superpixel-derived geometric features (area, width, height, compactness, eccentricity, solidity, Hu moments, etc.) into fixed-resolution grids.

---

## 📂 Repository Structure

```
SIGrid/
├── scripts/
│   ├── generate_sigrids.py   # Calls sigrid_compute functionality
│   ├── train_and_test.py     # Train + evaluate UNet/FCN on SIGrids
│   ├── save_data.py          # Save experiment metadata + model checkpoints
├── sigrid/
│   ├── models/               # UNet and FCN implementations
│   │   ├── unet.py
│   │   ├── fcn.py
│   ├── data/                 
│   │   ├── dataset.py        # Dataset loaders
│   │   ├── transforms.py     # Transforms
│   ├── train/
│   │   ├── trainer.py        # Trainer + evaluation utilities
│   ├── pipeline/
│   │   ├── sigrid_compute.py # Compute SIGrids
├── artifacts/                # (created at runtime) cache, results, predictions
├── data/                     # (ignored) raw datasets (e.g. CUB, DUTS, ECSSD, DUTS-OMRON)
└── requirements.txt
```

---

## 🚀 Installation

Clone the repo:

```bash
git clone https://github.com/<your-username>/SIGrid.git
cd SIGrid
```

Set up the environment (recommended with conda/venv):

```bash
pip install -r requirements.txt
```

## 📊 Datasets

The project supports several segmentation datasets:
- **CUB** (Caltech-UCSD Birds)
- **DUTS**
- **DUTS-OMRON**
- **ECSSD**

Place datasets inside a `data/` folder (not tracked by git):

```
data/
  CUB/
    train_images/   train_masks/   test_images/   test_masks/
  DUTS/
    train_images/   train_masks/   test_images/   test_masks/
  ...
```

---

## ⚡ Usage

### 1. Precompute SIGrids
Run the preprocessing script to compute fixed-grid representations:

```bash
python3 -m scripts.generate_sigrids   --input data/CUB/train_images   --masks data/CUB/train_masks   --output artifacts/CUB/cache   --dataset CUB   --n_segments 1500 --compactness 20 --grid 80   --features avg,hu
```

### 2. Train + Test

```bash
python3 -m scripts.train_and_test   --dataset CUB   --train_images data/CUB/train_images   --train_masks  data/CUB/train_masks   --test_images  data/CUB/test_images   --test_masks   data/CUB/test_masks   --n_segments 1500 --compactness 20 --grid 80   --features avg,hu   --arch unet --batch_size 32 --epochs 100 --eval_every 10   --lr 1e-3   --save_preds_dir artifacts/CUB/preds   --save_first_n 5   --workdir artifacts/CUB/results
```

At the end of training, the script will:
- Save model parameters (`.pth`)
- Save JSON with experiment metrics
- Dump a few prediction visualizations

