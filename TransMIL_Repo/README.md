# TransMIL: Overcoming Temporal Smearing and Data-Capacity Mismatch in Weakly Supervised Video Anomaly Detection

This repository contains the official PyTorch implementation for **TransMIL**, a lightweight Transformer-based architecture designed as a diagnostic tool for Weakly Supervised Video Anomaly Detection (WS-VAD).

We provide our core `TransMIL` architecture alongside full PyTorch implementations of several major baseline models (Sultani-MIL, RTFM, UMIL, PE-MIL), allowing for fair, unified comparisons using standard I3D features.

## 🚀 Features
- **TransMIL**: A shallow, highly optimized 1-layer, 4-head Transformer.
- **Baseline Implementations**:
  - Sultani-MIL (CVPR 2018)
  - RTFM (ICCV 2021)
  - UMIL (CVPR 2023)
  - PE-MIL (CVPR 2024)
- **Dual-Stream Support**: Easily combine RGB and Optical Flow features to eliminate Temporal Smearing.
- **Dataset Analysis Tools**: Scripts to analyze dataset variance, PCA feature space, and temporal activation profiles.

## 📁 Repository Structure

```
├── notebooks/                  # Interactive tutorials and experiments
├── scripts/                    # Scripts for dataset analysis and plotting
├── src/                        # Core codebase
│   ├── datasets/               # PyTorch Datasets for SHT and UCF-Crime
│   ├── models/                 # Model architectures (TransMIL, RTFM, etc.)
│   ├── utils/                  # Losses, evaluation metrics, and visualization
│   └── train.py                # Main training script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/TransMIL.git
cd TransMIL
pip install -r requirements.txt
```

## 📊 Dataset Preparation

You need pre-extracted I3D features for **ShanghaiTech** and **UCF-Crime**. 
Place the features in standard formats:
- `all_rgbs/Normal_Videos_event/*.npy`
- `all_rgbs/abnormal/*.npy`
- `splits/train.txt` and `splits/test.txt`

## 💻 Usage

To train the models using the main training script:

```bash
python src/train.py --model transmil --dataset sht --epochs 120 --batch_size 32
python src/train.py --model rtfm --dataset ucf
python src/train.py --model umil --dataset sht
```

For interactive training, evaluation, and visualizations, check the provided Jupyter Notebooks in the `notebooks/` directory.

## 📜 Citation

If you find this code useful in your research, please consider citing our paper.
