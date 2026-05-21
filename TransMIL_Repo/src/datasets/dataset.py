import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ShanghaiTechMIL(Dataset):
    """
    ShanghaiTech I3D features for MIL training.
    
    Structure used:
      all_rgbs/Normal_Videos_event/{vid}.npy  — normal videos (32×1024)
      all_rgbs/abnormal/{vid}.npy             — abnormal videos (32×1024)
      splits/train.txt, test.txt              — video ID lists
      GT_anomaly.pkl                          — ground truth labels
      testing/test_frame_mask/{vid}.npy       — per-frame binary masks
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


# ── 4. UCF-Crime Dataset ──────────────────────────────────────────────────

class UCFCrimeMIL(Dataset):
    """
    UCF-Crime I3D features for MIL training.
    
    Structure:
      all_rgbs/{Category}/{VideoName}.npy     — (32×1024) features
      splits/train_001.txt, test_001.txt      — video path lists
      GT_anomaly.pkl                          — ground truth
      exclusion.pkl                           — videos to exclude
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
            # entry format: "Category/VideoName.mp4"
            parts = entry.strip().split("/")
            if len(parts) != 2:
                continue
            category, video_file = parts[0], parts[1]
            
            # Feature file: Category/VideoName.mp4.npy
            feat_path = os.path.join(root, feat_base, category, f"{video_file}.npy")
            
            if not os.path.exists(feat_path):
                continue
            
            # Check exclusion
            vid_key = f"{category}/{video_file}"
            if vid_key in exclusions or video_file in exclusions:
                continue
            
            # Label: Normal_Videos_event → 0, everything else → 1
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


# ── 5. MIL Ranking Loss (Sultani et al., CVPR 2018) ───────────────────────

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
        
        n_n = sum(1 for s in self.samples if s[2] == 0)
        n_a = sum(1 for s in self.samples if s[2] == 1)
        print(f"[SHT-DualStream-{split}] Normal: {n_n} | Abnormal: {n_a}")
    
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
        
        n_n = sum(1 for s in self.samples if s[2] == 0)
        n_a = sum(1 for s in self.samples if s[2] == 1)
        print(f"[UCF-DualStream-{split}-fold{fold}] Normal: {n_n} | Abnormal: {n_a}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        rgb_path, flow_path, label, vid_id = self.samples[idx]
        rgb = np.load(rgb_path)
        flow = np.load(flow_path)
        feat = np.concatenate([rgb, flow], axis=-1)  # (32, 2048)
        return torch.from_numpy(feat).float(), label, vid_id


# ── 18. MILPairDataset for dual-stream (generic) ─────────────────────────

class MILPairDataset(Dataset):
    """Creates paired batches: one anomaly video + one normal video.
    Works with both 3-tuple (path, label, vid_id) and 4-tuple (rgb, flow, label, vid_id)."""
    def __init__(self, base_dataset):
        self.base = base_dataset
        # Use index -2 for label: works for both (path, label, id) and (rgb, flow, label, id)
        self.normal_indices = [i for i, s in enumerate(base_dataset.samples) if s[-2] == 0]
        self.anomaly_indices = [i for i, s in enumerate(base_dataset.samples) if s[-2] == 1]
        
        if len(self.anomaly_indices) == 0:
            print("⚠️  No anomaly samples in training set — using one-class mode")
            self.anomaly_indices = self.normal_indices  # fallback
        
        self.length = max(len(self.normal_indices), len(self.anomaly_indices))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        norm_idx = self.normal_indices[idx % len(self.normal_indices)]
        anom_idx = self.anomaly_indices[idx % len(self.anomaly_indices)]
        
        norm_data = self.base[norm_idx]
        anom_data = self.base[anom_idx]
        
        return anom_data[0], norm_data[0]  # features only


# ── 9. Training Loop ──────────────────────────────────────────────────────

class MILPairDatasetGeneric(Dataset):
    """Paired MIL loader that works with any dataset having .samples attribute."""
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.normal_indices = [i for i, s in enumerate(base_dataset.samples) if s[2] == 0]
        self.anomaly_indices = [i for i, s in enumerate(base_dataset.samples) if s[2] == 1]
        if not self.anomaly_indices:
            self.anomaly_indices = self.normal_indices
        self.length = max(len(self.normal_indices), len(self.anomaly_indices))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        n_idx = self.normal_indices[idx % len(self.normal_indices)]
        a_idx = self.anomaly_indices[idx % len(self.anomaly_indices)]
        norm_feat, _, _ = self.base[n_idx]
        anom_feat, _, _ = self.base[a_idx]
        return anom_feat, norm_feat


# ── 19. Per-Category Analysis (UCF-Crime) ─────────────────────────────────

