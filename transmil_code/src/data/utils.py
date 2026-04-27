import numpy as np
import pickle
import os

def load_features(npy_path):
    """Load I3D features: (32, 1024) float32."""
    feat = np.load(npy_path)
    assert feat.shape == (32, 1024), f"Unexpected shape {feat.shape} in {npy_path}"
    return feat

def load_pickle(pkl_path):
    """Load pickle file."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def detect_dataset_root(primary, fallbacks):
    """Try primary path, then fallbacks."""
    for p in [primary] + fallbacks:
        if os.path.isdir(p):
            return p
    return primary
