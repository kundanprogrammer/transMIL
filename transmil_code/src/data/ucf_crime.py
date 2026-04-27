import os
import torch
from torch.utils.data import Dataset, DataLoader
from .utils import load_features, load_pickle

class UCFCrimeMIL(Dataset):
    """
    UCF-Crime I3D features for MIL training.
    """
    def __init__(self, root, split="train", fold=1, stream="rgb"):
        self.root = root
        self.split = split
        
        feat_base = "all_rgbs" if stream == "rgb" else "all_flows"
        
        # Load split file
        split_file = os.path.join(root, "splits", f"{split}_{fold:03d}.txt")
        with open(split_file) as f:
            split_entries = [line.strip() for line in f if line.strip()]
        
        # Load exclusion list
        excl_path = os.path.join(root, "exclusion.pkl")
        exclusions = set()
        if os.path.exists(excl_path):
            excl_data = load_pickle(excl_path)
            if isinstance(excl_data, (list, set)):
                exclusions = set(excl_data)
            elif isinstance(excl_data, dict):
                for v in excl_data.values():
                    if isinstance(v, (list, set)):
                        exclusions.update(v)
        
        # Build samples
        self.samples = []
        for entry in split_entries:
            parts = entry.strip().split("/")
            if len(parts) != 2:
                continue
            category, video_file = parts[0], parts[1]
            feat_path = os.path.join(root, feat_base, category, f"{video_file}.npy")
            
            if not os.path.exists(feat_path):
                continue
            
            vid_key = f"{category}/{video_file}"
            if vid_key in exclusions or video_file in exclusions:
                continue
            
            label = 0 if category == "Normal_Videos_event" else 1
            self.samples.append((feat_path, label, vid_key))
        
        n_normal = sum(1 for _, l, _ in self.samples if l == 0)
        n_abnormal = sum(1 for _, l, _ in self.samples if l == 1)
        print(f"[UCF-{split}-fold{fold}] Normal: {n_normal} | Abnormal: {n_abnormal} | Total: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label, vid_id = self.samples[idx]
        feat = torch.from_numpy(load_features(path)).float()
        return feat, label, vid_id

class MILPairDataset(Dataset):
    """Creates paired batches: one anomaly video + one normal video."""
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.normal_indices = [i for i, s in enumerate(base_dataset.samples) if s[-2] == 0]
        self.anomaly_indices = [i for i, s in enumerate(base_dataset.samples) if s[-2] == 1]
        
        if len(self.anomaly_indices) == 0:
            self.anomaly_indices = self.normal_indices  # fallback
        
        self.length = max(len(self.normal_indices), len(self.anomaly_indices))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        norm_idx = self.normal_indices[idx % len(self.normal_indices)]
        anom_idx = self.anomaly_indices[idx % len(self.anomaly_indices)]
        
        norm_data = self.base[norm_idx]
        anom_data = self.base[anom_idx]
        
        return anom_data[0], norm_data[0]
