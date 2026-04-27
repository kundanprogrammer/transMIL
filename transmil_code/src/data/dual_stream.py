import os
import torch
import numpy as np
from torch.utils.data import Dataset
from .utils import load_pickle

class ShanghaiTechDualStream(Dataset):
    """Load both RGB + Flow features, concatenated → (32, 2048)."""
    def __init__(self, root, split="train"):
        self.root = root
        self.split = split
        
        rgb_normal = os.path.join(root, "all_rgbs", "Normal_Videos_event")
        rgb_abnormal = os.path.join(root, "all_rgbs", "abnormal")
        flow_normal = os.path.join(root, "all_flows", "Normal_Videos_event")
        flow_abnormal = os.path.join(root, "all_flows", "abnormal")
        
        split_file = os.path.join(root, "splits", f"{split}.txt")
        if not os.path.exists(split_file):
            self.samples = []
            return
            
        with open(split_file) as f:
            split_ids = set(line.strip() for line in f if line.strip())
        
        self.samples = []
        # Normal
        if os.path.isdir(rgb_normal):
            for fname in sorted(os.listdir(rgb_normal)):
                if fname.endswith(".npy"):
                    vid_id = fname.replace(".npy", "")
                    if vid_id in split_ids:
                        rgb_path = os.path.join(rgb_normal, fname)
                        flow_path = os.path.join(flow_normal, fname)
                        if os.path.exists(flow_path):
                            self.samples.append((rgb_path, flow_path, 0, vid_id))
        # Abnormal
        if os.path.isdir(rgb_abnormal):
            for fname in sorted(os.listdir(rgb_abnormal)):
                if fname.endswith(".npy"):
                    vid_id = fname.replace(".npy", "")
                    if vid_id in split_ids:
                        rgb_path = os.path.join(rgb_abnormal, fname)
                        flow_path = os.path.join(flow_abnormal, fname)
                        if os.path.exists(flow_path):
                            self.samples.append((rgb_path, flow_path, 1, vid_id))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        rgb_path, flow_path, label, vid_id = self.samples[idx]
        rgb = np.load(rgb_path)   # (32, 1024)
        flow = np.load(flow_path) # (32, 1024)
        feat = np.concatenate([rgb, flow], axis=-1)  # (32, 2048)
        return torch.from_numpy(feat).float(), label, vid_id

class UCFCrimeDualStream(Dataset):
    """Load both RGB + Flow features for UCF-Crime → (32, 2048)."""
    def __init__(self, root, split="train", fold=1):
        self.root = root
        
        split_file = os.path.join(root, "splits", f"{split}_{fold:03d}.txt")
        if not os.path.exists(split_file):
            self.samples = []
            return
            
        with open(split_file) as f:
            split_entries = [line.strip() for line in f if line.strip()]
        
        excl_path = os.path.join(root, "exclusion.pkl")
        exclusions = set()
        if os.path.exists(excl_path):
            excl_data = load_pickle(excl_path)
            if isinstance(excl_data, (list, set)):
                exclusions = set(excl_data)
        
        self.samples = []
        for entry in split_entries:
            parts = entry.strip().split("/")
            if len(parts) != 2:
                continue
            category, video_file = parts
            rgb_path = os.path.join(root, "all_rgbs", category, f"{video_file}.npy")
            flow_path = os.path.join(root, "all_flows", category, f"{video_file}.npy")
            
            if not os.path.exists(rgb_path) or not os.path.exists(flow_path):
                continue
            if f"{category}/{video_file}" in exclusions:
                continue
            
            label = 0 if category == "Normal_Videos_event" else 1
            self.samples.append((rgb_path, flow_path, label, f"{category}/{video_file}"))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        rgb_path, flow_path, label, vid_id = self.samples[idx]
        rgb = np.load(rgb_path)
        flow = np.load(flow_path)
        feat = np.concatenate([rgb, flow], axis=-1)  # (32, 2048)
        return torch.from_numpy(feat).float(), label, vid_id
