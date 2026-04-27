import os
import torch
from torch.utils.data import Dataset
from .utils import load_features

class ShanghaiTechMIL(Dataset):
    """
    ShanghaiTech I3D features for MIL training.
    """
    def __init__(self, root, split="train", stream="rgb"):
        self.root = root
        self.split = split
        
        feat_dir = "all_rgbs" if stream == "rgb" else "all_flows"
        normal_dir = os.path.join(root, feat_dir, "Normal_Videos_event")
        abnormal_dir = os.path.join(root, feat_dir, "abnormal")
        
        # Load split
        split_file = os.path.join(root, "splits", f"{split}.txt")
        with open(split_file) as f:
            split_ids = set(line.strip() for line in f if line.strip())
        
        # Build file list with labels
        self.samples = []  # (path, label, video_id)
        
        # Normal videos
        if os.path.isdir(normal_dir):
            for fname in sorted(os.listdir(normal_dir)):
                if fname.endswith(".npy"):
                    vid_id = fname.replace(".npy", "")
                    if vid_id in split_ids:
                        self.samples.append((
                            os.path.join(normal_dir, fname), 0, vid_id
                        ))
        
        # Abnormal videos
        if os.path.isdir(abnormal_dir):
            for fname in sorted(os.listdir(abnormal_dir)):
                if fname.endswith(".npy"):
                    vid_id = fname.replace(".npy", "")
                    if vid_id in split_ids:
                        self.samples.append((
                            os.path.join(abnormal_dir, fname), 1, vid_id
                        ))
        
        n_normal = sum(1 for _, l, _ in self.samples if l == 0)
        n_abnormal = sum(1 for _, l, _ in self.samples if l == 1)
        print(f"[SHT-{split}] Normal: {n_normal} | Abnormal: {n_abnormal} | Total: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label, vid_id = self.samples[idx]
        feat = torch.from_numpy(load_features(path)).float()  # (32, 1024)
        return feat, label, vid_id
