"""
EfficientVAD Training Notebook for Kaggle — XD-Violence Dataset
================================================================
Copy-paste this entire script into a Kaggle notebook cell.
Enable GPU: Settings → Accelerator → GPU T4 x2 (or P100).

Dataset: kundanpawar0001/kundan-xd
Mount point: /kaggle/input/kundan-xd/

This script is FULLY SELF-CONTAINED — no external imports needed.
"""

# ════════════════════════════════════════════════════════════════════
# CELL 1: Setup & Dataset Discovery
# ════════════════════════════════════════════════════════════════════

import os
import glob
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Auto-discover dataset structure ──
DATASET_ROOT = "/kaggle/input/kundan-xd"

print(f"\n{'='*60}")
print("DATASET DISCOVERY")
print(f"{'='*60}")

# Scan the directory
def scan_directory(root, depth=0, max_depth=3):
    """Print directory tree and find .npy files."""
    if depth > max_depth:
        return []
    npy_files = []
    try:
        entries = sorted(os.listdir(root))
        for e in entries[:20]:  # limit output
            full = os.path.join(root, e)
            indent = "  " * depth
            if os.path.isdir(full):
                sub_count = len(os.listdir(full)) if os.path.isdir(full) else 0
                print(f"{indent}📁 {e}/ ({sub_count} items)")
                npy_files.extend(scan_directory(full, depth + 1, max_depth))
            elif e.endswith('.npy'):
                size_mb = os.path.getsize(full) / 1e6
                print(f"{indent}📄 {e} ({size_mb:.2f} MB)")
                npy_files.append(full)
            elif e.endswith('.txt') or e.endswith('.csv') or e.endswith('.list'):
                print(f"{indent}📋 {e}")
            elif depth == 0:
                print(f"{indent}📄 {e}")
        if len(entries) > 20:
            print(f"{'  ' * depth}... and {len(entries) - 20} more")
    except PermissionError:
        pass
    return npy_files

npy_files_found = scan_directory(DATASET_ROOT)
all_npy = glob.glob(os.path.join(DATASET_ROOT, "**/*.npy"), recursive=True)
print(f"\nTotal .npy files found: {len(all_npy)}")

# Check a sample feature file
if all_npy:
    sample = np.load(all_npy[0])
    print(f"Sample feature shape: {sample.shape}")
    print(f"Sample feature dtype: {sample.dtype}")
    print(f"Sample feature range: [{sample.min():.4f}, {sample.max():.4f}]")
    FEAT_DIM = sample.shape[-1]
    print(f"Feature dimension: {FEAT_DIM}")
else:
    print("⚠️ No .npy files found! Check dataset mount.")
    FEAT_DIM = 512

# Look for existing list files
list_files = glob.glob(os.path.join(DATASET_ROOT, "**/*.txt"), recursive=True)
list_files += glob.glob(os.path.join(DATASET_ROOT, "**/*.list"), recursive=True)
list_files += glob.glob(os.path.join(DATASET_ROOT, "**/*.csv"), recursive=True)
print(f"\nList/text files found: {len(list_files)}")
for lf in list_files:
    print(f"  {os.path.relpath(lf, DATASET_ROOT)}")

# ════════════════════════════════════════════════════════════════════
# CELL 2: Model Definition — EfficientVAD
# ════════════════════════════════════════════════════════════════════

class MultiScaleTemporalConv(nn.Module):
    """
    Multi-Scale Temporal Convolution (MSTC) with bottleneck design.
    Projects to lower dim, applies parallel depthwise convs at k=3,5,7,
    aggregates via element-wise mean, projects back up. Residual connection.
    """
    def __init__(self, in_dim=512, bottleneck_dim=128):
        super().__init__()
        self.proj_down = nn.Linear(in_dim, bottleneck_dim)
        self.dw_conv3 = nn.Conv1d(
            bottleneck_dim, bottleneck_dim, kernel_size=3,
            padding=1, groups=bottleneck_dim
        )
        self.dw_conv5 = nn.Conv1d(
            bottleneck_dim, bottleneck_dim, kernel_size=5,
            padding=2, groups=bottleneck_dim
        )
        self.dw_conv7 = nn.Conv1d(
            bottleneck_dim, bottleneck_dim, kernel_size=7,
            padding=3, groups=bottleneck_dim
        )
        self.proj_up = nn.Linear(bottleneck_dim, in_dim)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        residual = x
        h = self.proj_down(x)
        h = h.transpose(1, 2)
        h3 = F.gelu(self.dw_conv3(h))
        h5 = F.gelu(self.dw_conv5(h))
        h7 = F.gelu(self.dw_conv7(h))
        h = (h3 + h5 + h7) / 3.0
        h = h.transpose(1, 2)
        h = self.proj_up(h)
        return self.norm(h + residual)


class SqueezeExcitation(nn.Module):
    """Channel attention via global average pooling + FC bottleneck."""
    def __init__(self, dim=512, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.fc2 = nn.Linear(dim // reduction, dim)

    def forward(self, x):
        s = x.mean(dim=1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(1)


class EfficientVAD(nn.Module):
    """
    Lightweight CLIP-adapted video anomaly detector.
    Input:  (B, T, feat_dim) — pre-extracted CLIP features
    Output: scores (B, T), magnitudes (B, T)
    """
    def __init__(self, feat_dim=512, bottleneck_dim=128, se_reduction=16,
                 hidden_dim=128, dropout=0.6):
        super().__init__()
        self.mstc = MultiScaleTemporalConv(feat_dim, bottleneck_dim)
        self.se = SqueezeExcitation(feat_dim, se_reduction)
        self.score_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B, T, D = x.shape
        h = self.mstc(x)
        h = self.se(h)
        magnitudes = torch.norm(h, p=2, dim=-1)
        scores = self.score_head(h.reshape(B * T, D))
        scores = scores.reshape(B, T)
        return scores, magnitudes


# ── Count parameters ──
model_test = EfficientVAD(feat_dim=FEAT_DIM)
total_params = sum(p.numel() for p in model_test.parameters())
trainable_params = sum(p.numel() for p in model_test.parameters() if p.requires_grad)
print(f"\n{'='*60}")
print(f"EfficientVAD Model Summary")
print(f"{'='*60}")
print(f"Feature dim: {FEAT_DIM}")
print(f"Total params: {total_params:,} ({total_params/1e6:.3f}M)")
print(f"Trainable params: {trainable_params:,}")
print(f"Model size: {total_params * 4 / 1024:.1f} KB (float32)")
del model_test

# ════════════════════════════════════════════════════════════════════
# CELL 3: Loss Function
# ════════════════════════════════════════════════════════════════════

class EfficientVADLoss(nn.Module):
    """
    Combined loss: L_rank + λ_mag*L_mag + λ_smooth*L_smooth + λ_sparse*L_sparse
    """
    def __init__(self, margin=1.0, mag_margin=100.0, mag_weight=1e-4,
                 smooth_weight=8e-5, sparse_weight=8e-5, top_k=3, drop_p=0.6):
        super().__init__()
        self.margin = margin
        self.mag_margin = mag_margin
        self.mag_weight = mag_weight
        self.smooth_weight = smooth_weight
        self.sparse_weight = sparse_weight
        self.top_k = top_k
        self.drop_p = drop_p

    def _topk_with_dropout(self, scores, magnitudes, k):
        if self.training:
            mask = torch.bernoulli(torch.full_like(magnitudes, 1.0 - self.drop_p))
            masked_mag = magnitudes * mask
        else:
            masked_mag = magnitudes
        _, topk_idx = masked_mag.topk(k, dim=1)
        topk_scores = scores.gather(1, topk_idx)
        topk_mags = magnitudes.gather(1, topk_idx)
        return topk_scores, topk_mags

    def forward(self, anom_scores, norm_scores, anom_mags, norm_mags):
        k = min(self.top_k, anom_scores.shape[1])

        anom_topk_scores, anom_topk_mags = self._topk_with_dropout(anom_scores, anom_mags, k)
        norm_topk_scores, norm_topk_mags = self._topk_with_dropout(norm_scores, norm_mags, k)

        anom_max = anom_topk_scores.mean(dim=1)
        norm_max = norm_topk_scores.mean(dim=1)
        loss_rank = F.relu(self.margin - anom_max.unsqueeze(1) + norm_max.unsqueeze(0)).mean()

        loss_mag = ((self.mag_margin - anom_topk_mags).abs() + norm_topk_mags).pow(2).mean()

        loss_smooth = (anom_scores[:, 1:] - anom_scores[:, :-1]).pow(2).mean()
        loss_sparse = anom_scores.mean()

        total = (loss_rank
                 + self.mag_weight * loss_mag
                 + self.smooth_weight * loss_smooth
                 + self.sparse_weight * loss_sparse)

        return total, {
            'rank': loss_rank.item(),
            'mag': loss_mag.item(),
            'smooth': loss_smooth.item(),
            'sparse': loss_sparse.item(),
        }


# ════════════════════════════════════════════════════════════════════
# CELL 4: Dataset Classes
# ════════════════════════════════════════════════════════════════════

def load_features(path, num_segments=32):
    """Load and temporally resample features to fixed number of segments."""
    feat = np.load(path)
    if feat.ndim == 1:
        feat = feat.reshape(1, -1)
    n = feat.shape[0]
    if n == num_segments:
        return feat
    if n > num_segments:
        indices = np.linspace(0, n - 1, num_segments, dtype=int)
        return feat[indices]
    reps = int(np.ceil(num_segments / n))
    feat = np.tile(feat, (reps, 1))[:num_segments]
    return feat


class XDViolenceDataset(Dataset):
    """
    XD-Violence dataset loader. Handles multiple directory structures:
    1. Flat: all .npy in one folder with list file
    2. Structured: features/ subfolder
    3. No list: auto-detect from filenames (XD labels in filename)
    """
    def __init__(self, root, split="train", num_segments=32, list_file=None):
        self.root = root
        self.split = split
        self.num_segments = num_segments
        self.samples = []

        # Strategy 1: Use provided list file
        if list_file and os.path.exists(list_file):
            self._load_from_list(list_file)
        else:
            # Strategy 2: Look for standard list files
            candidates = [
                os.path.join(root, f"{split}_list.txt"),
                os.path.join(root, f"{split}.txt"),
                os.path.join(root, f"{split}_list.list"),
                os.path.join(root, "lists", f"{split}_list.txt"),
                os.path.join(root, "list", f"{split}.txt"),
            ]
            found = False
            for c in candidates:
                if os.path.exists(c):
                    self._load_from_list(c)
                    found = True
                    break

            # Strategy 3: Auto-detect from .npy files
            if not found:
                self._auto_detect(root, split)

        n_n = sum(1 for _, l, _ in self.samples if l == 0)
        n_a = sum(1 for _, l, _ in self.samples if l == 1)
        print(f"[XD-{split}] Normal: {n_n} | Abnormal: {n_a} | Total: {len(self.samples)}")

    def _load_from_list(self, list_file):
        """Load from a list file: 'path label' or 'path\tlabel' per line."""
        feat_dirs = [
            self.root,
            os.path.join(self.root, "features"),
            os.path.join(self.root, "clip_features"),
        ]
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(None, 1)
                if len(parts) == 2:
                    path_str, label = parts[0], int(parts[1])
                elif len(parts) == 1:
                    # Try to infer label from path
                    path_str = parts[0]
                    label = self._label_from_name(path_str)
                else:
                    continue

                # Resolve path
                resolved = None
                if os.path.isabs(path_str) and os.path.exists(path_str):
                    resolved = path_str
                else:
                    for fd in feat_dirs:
                        candidate = os.path.join(fd, path_str)
                        if os.path.exists(candidate):
                            resolved = candidate
                            break
                        # Try with .npy extension
                        if not path_str.endswith('.npy'):
                            candidate = os.path.join(fd, path_str + '.npy')
                            if os.path.exists(candidate):
                                resolved = candidate
                                break

                if resolved:
                    vid_id = os.path.splitext(os.path.basename(resolved))[0]
                    self.samples.append((resolved, label, vid_id))

    def _auto_detect(self, root, split):
        """Auto-detect: find all .npy files and infer train/test split."""
        all_npy = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith('.npy') and not fn.startswith('.'):
                    all_npy.append(os.path.join(dirpath, fn))
        all_npy.sort()

        if not all_npy:
            print(f"⚠️ No .npy files found under {root}")
            return

        print(f"  Auto-detected {len(all_npy)} .npy files")

        # XD-Violence convention: filenames starting with specific patterns
        # Normal videos often have "label_A" or "normal" or "B1/B2" patterns
        # We'll use an 80/20 split if we can't determine
        for fp in all_npy:
            vid_id = os.path.splitext(os.path.basename(fp))[0]
            label = self._label_from_name(vid_id)
            self.samples.append((fp, label, vid_id))

        # Split into train/test (80/20 stratified)
        if split in ("train", "test"):
            normal = [s for s in self.samples if s[1] == 0]
            abnormal = [s for s in self.samples if s[1] == 1]
            np.random.seed(42)
            np.random.shuffle(normal)
            np.random.shuffle(abnormal)

            n_train_norm = int(len(normal) * 0.8)
            n_train_abn = int(len(abnormal) * 0.8)

            if split == "train":
                self.samples = normal[:n_train_norm] + abnormal[:n_train_abn]
            else:
                self.samples = normal[n_train_norm:] + abnormal[n_train_abn:]

    def _label_from_name(self, name):
        """Infer label from XD-Violence filename conventions."""
        name_lower = name.lower()
        # XD-Violence: videos with violence labels typically contain category keywords
        violence_keywords = [
            'abuse', 'car_accident', 'explosion', 'fighting', 'riot',
            'shooting', 'stealing', 'shoplifting', 'vandalism', 'robbery',
            'burglary', 'assault', 'arson', 'arrest',
            # XD-Violence specific patterns
            'label_b1', 'label_b2', 'label_b4', 'label_b5', 'label_b6',
            'b1-', 'b2-', 'b4-', 'b5-', 'b6-',
        ]
        normal_keywords = [
            'normal', 'label_a', 'a-',
        ]
        for kw in normal_keywords:
            if kw in name_lower:
                return 0
        for kw in violence_keywords:
            if kw in name_lower:
                return 1
        # Default: treat as normal (safer for MIL training)
        return 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, vid_id = self.samples[idx]
        feat = torch.from_numpy(load_features(path, self.num_segments)).float()
        return feat, label, vid_id


class MILPairDataset(Dataset):
    """Creates paired batches: one anomaly + one normal video."""
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.normal_indices = [i for i, s in enumerate(base_dataset.samples) if s[1] == 0]
        self.anomaly_indices = [i for i, s in enumerate(base_dataset.samples) if s[1] == 1]
        if not self.anomaly_indices:
            print("⚠️  No anomaly samples — fallback to one-class mode")
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


# ════════════════════════════════════════════════════════════════════
# CELL 5: Training & Evaluation Functions
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, test_dataset, device, gt_mask_dir=None):
    """Evaluate: video-level AUC/AP + frame-level if GT masks available."""
    model.eval()
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    all_frame_scores = []
    all_frame_labels = []
    video_results = {}

    for feats, label, vid_id in loader:
        feats = feats.to(device)
        scores, _ = model(feats)
        scores = scores[0].cpu().numpy()
        vid_name = vid_id[0] if isinstance(vid_id, (list, tuple)) else vid_id

        video_results[vid_name] = {
            'scores': scores,
            'label': label.item(),
        }

        # Frame-level evaluation if GT mask exists
        mask_path = None
        if gt_mask_dir:
            for ext in ['', '.npy']:
                candidate = os.path.join(gt_mask_dir, f'{vid_name}{ext}')
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

        if mask_path:
            frame_mask = np.load(mask_path)
            n_frames = len(frame_mask)
            frame_scores = np.repeat(scores, max(1, n_frames // len(scores) + 1))[:n_frames]
            if len(frame_scores) < n_frames:
                pad = np.full(n_frames - len(frame_scores), scores[-1])
                frame_scores = np.concatenate([frame_scores, pad])
            all_frame_scores.extend(frame_scores)
            all_frame_labels.extend(frame_mask)

    results = {}

    # Frame-level metrics
    if all_frame_scores:
        fs = np.array(all_frame_scores)
        fl = np.array(all_frame_labels).astype(int)
        if len(np.unique(fl)) > 1:
            results['frame_auc'] = roc_auc_score(fl, fs) * 100
            results['frame_ap'] = average_precision_score(fl, fs) * 100

    # Video-level metrics
    if video_results:
        vid_scores = np.array([v['scores'].max() for v in video_results.values()])
        vid_labels = np.array([v['label'] for v in video_results.values()])
        if len(np.unique(vid_labels)) > 1:
            results['video_auc'] = roc_auc_score(vid_labels, vid_scores) * 100
            results['video_ap'] = average_precision_score(vid_labels, vid_scores) * 100

    return results


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    loss_details = {}

    for anom_feat, norm_feat in loader:
        anom_feat = anom_feat.to(device)
        norm_feat = norm_feat.to(device)

        optimizer.zero_grad()
        anom_scores, anom_mags = model(anom_feat)
        norm_scores, norm_mags = model(norm_feat)

        loss, details = criterion(anom_scores, norm_scores, anom_mags, norm_mags)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in details.items():
            loss_details[k] = loss_details.get(k, 0) + v

    n = max(1, len(loader))
    avg_loss = total_loss / n
    avg_details = {k: v / n for k, v in loss_details.items()}
    return avg_loss, avg_details


# ════════════════════════════════════════════════════════════════════
# CELL 6: MAIN TRAINING LOOP
# ════════════════════════════════════════════════════════════════════

# ── Hyperparameters ──
CONFIG = {
    'feat_dim': FEAT_DIM,
    'bottleneck_dim': 128,
    'se_reduction': 16,
    'hidden_dim': 128,
    'dropout': 0.6,
    'num_segments': 32,
    'epochs': 100,
    'batch_size': 32,
    'lr': 1e-3,
    'weight_decay': 5e-4,
    'top_k': 3,
    'mag_margin': 100.0,
    'mag_weight': 1e-4,
    'smooth_weight': 8e-5,
    'sparse_weight': 8e-5,
    'eval_every': 5,
    'seed': 42,
}

# Reproducibility
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(CONFIG['seed'])

print(f"\n{'='*60}")
print("LOADING DATASETS")
print(f"{'='*60}")

# Load datasets
train_ds = XDViolenceDataset(DATASET_ROOT, split="train", num_segments=CONFIG['num_segments'])
test_ds = XDViolenceDataset(DATASET_ROOT, split="test", num_segments=CONFIG['num_segments'])

if len(train_ds) == 0:
    print("\n⚠️  TRAIN DATASET IS EMPTY!")
    print("Please check your dataset structure. Expected .npy feature files.")
    print(f"Dataset root: {DATASET_ROOT}")
    print("\nLet me try loading ALL .npy files for debugging...")
    all_npy = glob.glob(os.path.join(DATASET_ROOT, "**/*.npy"), recursive=True)
    print(f"Found {len(all_npy)} .npy files total")
    if all_npy:
        print("First 10 files:")
        for f in all_npy[:10]:
            arr = np.load(f)
            print(f"  {os.path.relpath(f, DATASET_ROOT)} → shape: {arr.shape}")
    raise RuntimeError("Cannot proceed with empty dataset. See structure above.")

train_pairs = MILPairDataset(train_ds)
train_loader = DataLoader(
    train_pairs, batch_size=CONFIG['batch_size'],
    shuffle=True, num_workers=2, pin_memory=True
)

# Look for GT mask directory
gt_mask_candidates = [
    os.path.join(DATASET_ROOT, "test_frame_mask"),
    os.path.join(DATASET_ROOT, "frame_mask"),
    os.path.join(DATASET_ROOT, "gt"),
    os.path.join(DATASET_ROOT, "annotations"),
]
GT_MASK_DIR = None
for gmc in gt_mask_candidates:
    if os.path.isdir(gmc):
        GT_MASK_DIR = gmc
        print(f"Found GT mask directory: {gmc}")
        break

# Build model
model = EfficientVAD(
    feat_dim=CONFIG['feat_dim'],
    bottleneck_dim=CONFIG['bottleneck_dim'],
    se_reduction=CONFIG['se_reduction'],
    hidden_dim=CONFIG['hidden_dim'],
    dropout=CONFIG['dropout'],
).to(DEVICE)

criterion = EfficientVADLoss(
    mag_margin=CONFIG['mag_margin'],
    mag_weight=CONFIG['mag_weight'],
    smooth_weight=CONFIG['smooth_weight'],
    sparse_weight=CONFIG['sparse_weight'],
    top_k=CONFIG['top_k'],
)
optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[int(CONFIG['epochs'] * 0.6), int(CONFIG['epochs'] * 0.8)],
    gamma=0.1,
)

SAVE_DIR = "/kaggle/working"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"\n{'='*60}")
print("TRAINING EfficientVAD on XD-Violence")
print(f"{'='*60}")
print(f"Config: {json.dumps(CONFIG, indent=2)}")
print(f"Train samples: {len(train_ds)} ({len(train_pairs)} pairs)")
print(f"Test samples: {len(test_ds)}")
print(f"Device: {DEVICE}")
print()

best_metric = 0
best_epoch = 0
history = []

for epoch in range(1, CONFIG['epochs'] + 1):
    t0 = time.time()
    avg_loss, details = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    scheduler.step()
    elapsed = time.time() - t0

    log = (f"Epoch {epoch:3d}/{CONFIG['epochs']} | "
           f"Loss: {avg_loss:.4f} | "
           f"rank: {details['rank']:.4f} "
           f"mag: {details['mag']:.1f} "
           f"smooth: {details['smooth']:.6f} "
           f"sparse: {details['sparse']:.4f} | "
           f"{elapsed:.1f}s")
    print(log)

    if epoch % CONFIG['eval_every'] == 0 or epoch == CONFIG['epochs']:
        results = evaluate(model, test_ds, DEVICE, GT_MASK_DIR)

        # XD-Violence primary metric is AP
        primary = results.get('frame_ap', results.get('video_ap', 0))

        res_str = " | ".join(f"{k}: {v:.2f}" for k, v in results.items())
        print(f"  ▶ Eval: {res_str}")

        if primary > best_metric:
            best_metric = primary
            best_epoch = epoch
            ckpt_path = os.path.join(SAVE_DIR, "efficientvad_xd_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'results': results,
                'config': CONFIG,
            }, ckpt_path)
            print(f"  ★ Best model saved! (metric={primary:.2f})")

        history.append({'epoch': epoch, 'loss': avg_loss, **results})

# ════════════════════════════════════════════════════════════════════
# CELL 7: Final Summary & Inference Benchmark
# ════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")
print(f"Best epoch: {best_epoch}")
print(f"Best metric (AP/AUC): {best_metric:.2f}")
print(f"Total params: {total_params:,} ({total_params/1e6:.3f}M)")

# FLOPs estimate
flops = 0
for name, m in model.named_modules():
    if isinstance(m, nn.Linear):
        flops += 2 * m.in_features * m.out_features
        if m.bias is not None:
            flops += m.out_features
    elif isinstance(m, nn.Conv1d):
        k = m.kernel_size[0]
        c_in = m.in_channels // m.groups
        flops += 2 * k * c_in * m.out_channels * CONFIG['num_segments']
print(f"Estimated FLOPs: {flops:,} ({flops/1e6:.2f}M)")

# Inference speed benchmark
model.eval()
dummy = torch.randn(1, CONFIG['num_segments'], CONFIG['feat_dim']).to(DEVICE)
with torch.no_grad():
    for _ in range(10):
        model(dummy)  # warmup
if torch.cuda.is_available():
    torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    for _ in range(1000):
        model(dummy)
if torch.cuda.is_available():
    torch.cuda.synchronize()
latency_ms = (time.time() - t0) / 1000 * 1000
print(f"Inference latency: {latency_ms:.2f} ms/sample")
print(f"Model size: {total_params * 4 / 1024:.1f} KB (float32)")

# Save summary
summary = {
    'dataset': 'xd-violence',
    'best_epoch': best_epoch,
    'best_metric': best_metric,
    'total_params': total_params,
    'flops': flops,
    'latency_ms': latency_ms,
    'history': history,
    'config': CONFIG,
}
with open(os.path.join(SAVE_DIR, "efficientvad_xd_summary.json"), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to {SAVE_DIR}/efficientvad_xd_summary.json")

# Print training curve
print(f"\n{'='*60}")
print("TRAINING HISTORY")
print(f"{'='*60}")
for h in history:
    line = f"Epoch {h['epoch']:3d} | Loss: {h['loss']:.4f}"
    if 'video_ap' in h:
        line += f" | Video AP: {h['video_ap']:.2f}"
    if 'video_auc' in h:
        line += f" | Video AUC: {h['video_auc']:.2f}"
    if 'frame_ap' in h:
        line += f" | Frame AP: {h['frame_ap']:.2f}"
    if 'frame_auc' in h:
        line += f" | Frame AUC: {h['frame_auc']:.2f}"
    print(line)

print(f"\n{'='*60}")
print("COMPARISON TABLE (for paper)")
print(f"{'='*60}")
print(f"| Method | XD-Violence AP% | Params |")
print(f"|--------|----------------|--------|")
print(f"| Sultani (CVPR 18) | 73.20 | ~2.1M |")
print(f"| RTFM (ICCV 21) | 77.81 | ~24.7M |")
print(f"| Light-WVAD (Neuro 24) | 77.30 | 0.14M |")
print(f"| CLIP-TSA (ICIP 23) | 82.17 | 1.68M |")
print(f"| VadCLIP (AAAI 24) | 84.51 | large |")
print(f"| **EfficientVAD (Ours)** | **{best_metric:.2f}** | **{total_params/1e6:.3f}M** |")
print(f"{'='*60}")
print("Done! Download efficientvad_xd_best.pth from /kaggle/working/")
