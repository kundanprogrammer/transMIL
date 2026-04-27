# TransMIL: Temporal Transformer for Video Anomaly Detection

Official implementation of the TransMIL architecture for weakly-supervised Video Anomaly Detection (VAD). This repository modularizes the original Kaggle research experiments into a professional, reproducible library.

## Overview
TransMIL re-evaluates the Transformer paradigm in Video Anomaly Detection. By utilizing multi-head self-attention to model temporal dependencies between video segments, TransMIL achieves state-of-the-art results on the ShanghaiTech and UCF-Crime datasets.

### Key Contributions
- **Temporal transformer encoder** for global context modeling.
- **Dual-stream integration** (RGB + Optical Flow).
- **Addressing the "Temporal Smearing Effect"** using sparse attention.

## Repository Structure
```
transmil_code/
├── src/
│   ├── data/             # Dataset classes (ShanghaiTech, UCF-Crime)
│   ├── models/           # Architectures (MIL, RTFM, TransMIL)
│   ├── training/         # Training engine and loss functions
│   ├── utils/            # Visualization utilities
│   └── main.py           # Experiment entry point
├── requirements.txt      # Dependency list
└── README.md            # You are here
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/transmil-vad.git
   cd transmil-vad
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation
This code expects the I3D features for ShanghaiTech and UCF-Crime datasets.
- **ShanghaiTech**: Download from [Source] and place in `data/ShanghaiTech`.
- **UCF-Crime**: Download from [Source] and place in `data/UCF-Crime`.

## Usage
To run the default TransMIL experiment:
```bash
python src/main.py
```

## Citation
If you use this code in your research, please cite our paper:
```bibtex
@article{transmil2026,
  title={Re-evaluating Weakly-Supervised Video Anomaly Detection},
  author={Kunda, et al.},
  journal={Journal of Visual Communication and Image Representation},
  year={2026}
}
```

## License
MIT License. See `LICENSE` for details.
