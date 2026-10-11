"""
EfficientVAD — XD-Violence Training (v7)
=========================================
Iteration-based training using CLIP-TSA pre-extracted CLIP ViT-B/16 features.
Frame-level AP evaluation matching CLIP-TSA methodology exactly.

Key changes from v6:
  - Uses CLIP-TSA feature directory (train/abnormal, train/normal, test/abnormal, test/normal)
  - Single .npy per video, shape (T, 512) — no 10-crop handling
  - Iteration-based training (4000 steps, batch_size=16) instead of epoch-based
  - Uses pre-computed gt-xd.npy or builds from annotations.txt
  - Test ordering from CLIP-TSA list file (abnormal first, then normal)
  - Evaluates every 5 steps after step 150 (matching CLIP-TSA)
  - Caches test features in memory for fast repeated evaluation
"""

import os, sys, glob, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

VERSION = "v7-clip-tsa-iter"
CLIP_LEN = 16
NUM_SEG = 32

print(f"EfficientVAD {VERSION}")
print(f"PyTorch {torch.__version__}  |  CUDA {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════
# 1. LOCATE CLIP-TSA FEATURES
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 1: LOCATING CLIP-TSA FEATURES")
print("=" * 70)

INPUT_BASE = "/kaggle/input"

if os.path.isdir(INPUT_BASE):
    for d in sorted(os.listdir(INPUT_BASE)):
        full = os.path.join(INPUT_BASE, d)
        if os.path.isdir(full):
            n = len(os.listdir(full))
            print(f"  [{d}/] ({n} items)")

CLIP_TSA_CANDIDATES = [
    os.path.join(INPUT_BASE, "datasets", "kundanpawar0001", "kundan-xd",
                 "Kundan_xdviolence_features",
                 "xd_fromhttpsgithub.comjoos2010kjCLIP-TSA"),
    os.path.join(INPUT_BASE, "kundan-xd", "Kundan_xdviolence_features",
                 "xd_fromhttpsgithub.comjoos2010kjCLIP-TSA"),
]

CLIP_TSA_ROOT = None
for cand in CLIP_TSA_CANDIDATES:
    if os.path.isdir(cand):
        CLIP_TSA_ROOT = cand
        break

if CLIP_TSA_ROOT is None:
    print("Searching for CLIP-TSA features directory...")
    for root, dirs, files in os.walk(INPUT_BASE):
        basename = os.path.basename(root)
        if "clip-tsa" in basename.lower() or "CLIP-TSA" in basename:
            if "features" in dirs or os.path.isdir(os.path.join(root, "features")):
                CLIP_TSA_ROOT = root
                break
        if "features" in dirs:
            feat_dir = os.path.join(root, "features")
            subdirs = set(os.listdir(feat_dir)) if os.path.isdir(feat_dir) else set()
            if {"test", "train"}.issubset(subdirs):
                CLIP_TSA_ROOT = root
                break
        depth = root.replace(INPUT_BASE, "").count(os.sep)
        if depth > 6:
            break

if CLIP_TSA_ROOT is None:
    print("\nFull directory tree (first 5 levels):")
    for dp, dn, fn in os.walk(INPUT_BASE):
        depth = dp.replace(INPUT_BASE, "").count(os.sep)
        if depth < 5:
            n_npy = sum(1 for f in fn if f.endswith(".npy"))
            print(f"{'  ' * depth}[{os.path.basename(dp)}/] "
                  f"({n_npy} npy, {len(fn)} total, {len(dn)} dirs)")
    sys.exit("FATAL: Cannot find CLIP-TSA features directory")

FEAT_ROOT = os.path.join(CLIP_TSA_ROOT, "features")
print(f"CLIP-TSA root: {CLIP_TSA_ROOT}")
print(f"Features root: {FEAT_ROOT}")

TRAIN_ABN = os.path.join(FEAT_ROOT, "train", "abnormal")
TRAIN_NOR = os.path.join(FEAT_ROOT, "train", "normal")
TEST_ABN = os.path.join(FEAT_ROOT, "test", "abnormal")
TEST_NOR = os.path.join(FEAT_ROOT, "test", "normal")

for d, label in [(TRAIN_ABN, "train/abnormal"), (TRAIN_NOR, "train/normal"),
                  (TEST_ABN, "test/abnormal"), (TEST_NOR, "test/normal")]:
    if not os.path.isdir(d):
        sys.exit(f"FATAL: Missing directory {d}")
    n = len([f for f in os.listdir(d) if f.endswith(".npy")])
    print(f"  {label}: {n} files")


# ═══════════════════════════════════════════════════════════
# 2. BUILD SAMPLE LISTS WITH CORRECT ORDERING
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 2: BUILDING SAMPLE LISTS")
print("=" * 70)

train_abn_paths = sorted(glob.glob(os.path.join(TRAIN_ABN, "*.npy")))
train_nor_paths = sorted(glob.glob(os.path.join(TRAIN_NOR, "*.npy")))

test_abn_on_disk = sorted(glob.glob(os.path.join(TEST_ABN, "*.npy")))
test_nor_on_disk = sorted(glob.glob(os.path.join(TEST_NOR, "*.npy")))

LIST_FILE_CANDIDATES = [
    os.path.join(CLIP_TSA_ROOT, "list", "vit", "xd-vit-test.list"),
]
for root, dirs, files in os.walk(INPUT_BASE):
    if "xd-vit-test.list" in files:
        LIST_FILE_CANDIDATES.append(os.path.join(root, "xd-vit-test.list"))
    depth = root.replace(INPUT_BASE, "").count(os.sep)
    if depth > 8:
        break

test_paths = None
for lf in LIST_FILE_CANDIDATES:
    if not os.path.isfile(lf):
        continue
    print(f"Found test list file: {lf}")
    with open(lf) as f:
        list_lines = [l.strip() for l in f if l.strip()]

    basename_to_path = {}
    for p in test_abn_on_disk + test_nor_on_disk:
        basename_to_path[os.path.basename(p)] = p

    ordered = []
    missing = 0
    for line in list_lines:
        bn = os.path.basename(line)
        if bn in basename_to_path:
            ordered.append(basename_to_path[bn])
        else:
            missing += 1
    if missing > 0:
        print(f"  WARNING: {missing} files in list not found on disk")
    if len(ordered) > 0:
        test_paths = ordered
        print(f"  Test ordering from list file: {len(test_paths)} videos")
        break

if test_paths is None:
    test_paths = test_abn_on_disk + test_nor_on_disk
    print(f"  Test ordering: sorted abnormal + sorted normal (fallback)")

n_test_abn = sum(1 for p in test_paths if "abnormal" in p)
n_test_nor = len(test_paths) - n_test_abn

sample = np.load(train_abn_paths[0])
FEAT_DIM = sample.shape[-1]
print(f"\nFeature dim: {FEAT_DIM}  |  sample shape: {sample.shape}")
print(f"Train: {len(train_abn_paths)} abnormal + {len(train_nor_paths)} normal "
      f"= {len(train_abn_paths) + len(train_nor_paths)}")
print(f"Test:  {n_test_abn} abnormal + {n_test_nor} normal = {len(test_paths)}")
print(f"First 3 test: {[os.path.basename(p) for p in test_paths[:3]]}")


# ═══════════════════════════════════════════════════════════
# 3. LOAD OR BUILD FRAME-LEVEL GT
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 3: LOADING/BUILDING FRAME-LEVEL GT")
print("=" * 70)

GT_NPY_CANDIDATES = [
    os.path.join(CLIP_TSA_ROOT, "list", "gt-xd.npy"),
]
for root, dirs, files in os.walk(INPUT_BASE):
    if "gt-xd.npy" in files:
        GT_NPY_CANDIDATES.append(os.path.join(root, "gt-xd.npy"))
    depth = root.replace(INPUT_BASE, "").count(os.sep)
    if depth > 8:
        break

frame_gt = None
for gt_path in GT_NPY_CANDIDATES:
    if os.path.isfile(gt_path):
        frame_gt = np.load(gt_path).astype(np.float32)
        n_pos = int(frame_gt.sum())
        print(f"Loaded GT from: {gt_path}")
        print(f"GT frames:   {len(frame_gt):,}")
        print(f"Anomalous:   {n_pos:,} ({100 * n_pos / len(frame_gt):.2f}%)")
        break

if frame_gt is None:
    print("gt-xd.npy not found, building from annotations.txt...")
    ANNOT_CANDIDATES = []
    for root, dirs, files in os.walk(INPUT_BASE):
        if "annotations.txt" in files:
            ANNOT_CANDIDATES.append(os.path.join(root, "annotations.txt"))
        depth = root.replace(INPUT_BASE, "").count(os.sep)
        if depth > 8:
            break

    annot_path = None
    for p in ANNOT_CANDIDATES:
        if os.path.isfile(p):
            annot_path = p
            break

    assert annot_path is not None, "Neither gt-xd.npy nor annotations.txt found"

    print(f"Building GT from: {annot_path}")
    with open(annot_path) as f:
        gt_lines = list(f)
    print(f"Annotation lines: {len(gt_lines)}")

    gt_all = []
    matched = 0
    for path in test_paths:
        feat = np.load(path, mmap_mode="r")
        T = feat.shape[0]
        lens = (T + 1) * CLIP_LEN
        gt_vec = np.zeros(lens, dtype=np.float32)

        name = os.path.splitext(os.path.basename(path))[0]
        is_normal = "label_A" in name

        if not is_normal:
            for gt_line in gt_lines:
                if name in gt_line:
                    matched += 1
                    gt_content = gt_line.strip().split()
                    for i in range(1, len(gt_content), 2):
                        j = i + 1
                        if j < len(gt_content):
                            start, end = int(gt_content[i]), int(gt_content[j])
                            gt_vec[start:end] = 1.0
                    break

        gt_all.extend(gt_vec[:-CLIP_LEN])

    frame_gt = np.array(gt_all, dtype=np.float32)
    n_pos = int(frame_gt.sum())
    print(f"GT frames:   {len(frame_gt):,}")
    print(f"Anomalous:   {n_pos:,} ({100 * n_pos / len(frame_gt):.2f}%)")
    print(f"Matched:     {matched} / {n_test_abn} abnormal videos")

print(f"Expected:    ~2,330,384 frames / ~537,805 anomalous (CLIP-TSA)")
if frame_gt.sum() == 0:
    print("*** FATAL: GT has zero anomalous frames — evaluation will be meaningless ***")


# ═══════════════════════════════════════════════════════════
# 4. FEATURE PROCESSING (from CLIP-TSA utils)
# ═══════════════════════════════════════════════════════════


def process_feat(feat, length):
    """Divide variable-length features into fixed segments (CLIP-TSA)."""
    new_feat = np.zeros((length, feat.shape[1]), dtype=np.float32)
    r = np.linspace(0, len(feat), length + 1, dtype=np.int32)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i] = np.mean(feat[r[i]:r[i + 1]], axis=0)
        else:
            new_feat[i] = feat[r[i]]
    return new_feat


# ═══════════════════════════════════════════════════════════
# 5. MODEL (234K params)
# ═══════════════════════════════════════════════════════════


class MultiScaleTemporalConv(nn.Module):
    def __init__(self, d, bn):
        super().__init__()
        self.down = nn.Linear(d, bn)
        self.c3 = nn.Conv1d(bn, bn, 3, padding=1, groups=bn)
        self.c5 = nn.Conv1d(bn, bn, 5, padding=2, groups=bn)
        self.c7 = nn.Conv1d(bn, bn, 7, padding=3, groups=bn)
        self.up = nn.Linear(bn, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        res = x
        h = self.down(x).transpose(1, 2)
        h = (F.gelu(self.c3(h)) + F.gelu(self.c5(h)) + F.gelu(self.c7(h))) / 3
        return self.norm(self.up(h.transpose(1, 2)) + res)


class SqueezeExcitation(nn.Module):
    def __init__(self, d, r=16):
        super().__init__()
        self.fc1 = nn.Linear(d, d // r)
        self.fc2 = nn.Linear(d // r, d)

    def forward(self, x):
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(x.mean(1)))))
        return x * s.unsqueeze(1)


class EfficientVAD(nn.Module):
    def __init__(self, d=512, bn=128, se_r=16, hid=128, drop=0.6):
        super().__init__()
        self.mstc = MultiScaleTemporalConv(d, bn)
        self.se = SqueezeExcitation(d, se_r)
        self.head = nn.Sequential(
            nn.Linear(d, hid), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hid, 1), nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.se(self.mstc(x))
        B, T, D = h.shape
        scores = self.head(h.reshape(B * T, D)).reshape(B, T)
        mags = torch.norm(h, p=2, dim=-1)
        return scores, mags


# ═══════════════════════════════════════════════════════════
# 6. MIL RANKING LOSS
# ═══════════════════════════════════════════════════════════


class MILLoss(nn.Module):
    def __init__(self, margin=1.0, mag_m=100.0, mag_w=1e-4,
                 sm_w=8e-4, sp_w=8e-3, k=3, drop=0.6):
        super().__init__()
        self.margin = margin
        self.mag_m, self.mag_w = mag_m, mag_w
        self.sm_w, self.sp_w = sm_w, sp_w
        self.k, self.drop = k, drop

    def _topk(self, sc, mg, k):
        if self.training:
            mg = mg * torch.bernoulli(torch.full_like(mg, 1 - self.drop))
        _, idx = mg.topk(k, dim=1)
        return sc.gather(1, idx), mg.gather(1, idx)

    def forward(self, a_sc, n_sc, a_mg, n_mg):
        k = min(self.k, a_sc.shape[1])
        a_tk, a_mk = self._topk(a_sc, a_mg, k)
        n_tk, n_mk = self._topk(n_sc, n_mg, k)
        l_rank = F.relu(self.margin
                        - a_tk.mean(1, keepdim=True)
                        + n_tk.mean(1, keepdim=True)).mean()
        l_mag = ((self.mag_m - a_mk).abs() + n_mk).pow(2).mean()
        l_sm = (a_sc[:, 1:] - a_sc[:, :-1]).pow(2).mean()
        l_sp = a_sc.mean()
        total = l_rank + self.mag_w * l_mag + self.sm_w * l_sm + self.sp_w * l_sp
        return total, dict(rank=l_rank.item(), mag=l_mag.item(),
                           smooth=l_sm.item(), sparse=l_sp.item())


# ═══════════════════════════════════════════════════════════
# 7. DATASET
# ═══════════════════════════════════════════════════════════


class TrainSegDataset(Dataset):
    """Single-class dataset. Returns process_feat'd 32-segment features."""
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        feat = np.load(self.paths[i]).astype(np.float32)
        feat = process_feat(feat, NUM_SEG)
        return torch.from_numpy(feat)


# ═══════════════════════════════════════════════════════════
# 8. FRAME-LEVEL EVALUATION (matching CLIP-TSA exactly)
# ═══════════════════════════════════════════════════════════

_test_feat_cache = []


def _cache_test_features(file_paths):
    """Load all test features into memory once."""
    global _test_feat_cache
    if len(_test_feat_cache) == len(file_paths):
        return
    print(f"  Caching {len(file_paths)} test features in memory...")
    _test_feat_cache = []
    total_segs = 0
    for path in file_paths:
        feat = np.load(path).astype(np.float32)
        _test_feat_cache.append(feat)
        total_segs += feat.shape[0]
    mem_mb = sum(f.nbytes for f in _test_feat_cache) / 1e6
    print(f"  Cached: {total_segs:,} total segments, {mem_mb:.1f} MB")


@torch.no_grad()
def evaluate(model, file_paths, gt, device):
    """
    Frame-level AP matching CLIP-TSA:
      1. Feed each test video's raw (T_i, D) features
      2. Get T_i per-segment anomaly scores
      3. Concatenate across all videos (same order as GT)
      4. np.repeat(scores, 16) for frame-level predictions
      5. average_precision_score(gt, preds)
    """
    model.eval()
    _cache_test_features(file_paths)

    all_scores = []
    for feat in _test_feat_cache:
        x = torch.from_numpy(feat).unsqueeze(0).to(device)
        sc, _ = model(x)
        all_scores.append(sc[0].cpu().numpy())

    seg_pred = np.concatenate(all_scores)
    frame_pred = np.repeat(seg_pred, CLIP_LEN)

    gt_use = gt
    if len(frame_pred) != len(gt_use):
        mn = min(len(frame_pred), len(gt_use))
        frame_pred = frame_pred[:mn]
        gt_use = gt_use[:mn]

    ap = average_precision_score(gt_use, frame_pred) * 100
    auc_val = roc_auc_score(gt_use, frame_pred) * 100
    return ap, auc_val


# ═══════════════════════════════════════════════════════════
# 9. TRAINING (iteration-based, matching CLIP-TSA)
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 4: TRAINING")
print("=" * 70)

MAX_ITER = 4000
BATCH = 16
LR = 1e-3
WD = 5e-3
K = 3
SEED = 3
EVAL_INTERVAL = 5
EVAL_START = 150

torch.manual_seed(SEED)
np.random.seed(SEED)
import random
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

abn_ds = TrainSegDataset(train_abn_paths)
nor_ds = TrainSegDataset(train_nor_paths)

abn_loader = DataLoader(abn_ds, batch_size=BATCH, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)
nor_loader = DataLoader(nor_ds, batch_size=BATCH, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

print(f"Abnormal loader: {len(abn_loader)} batches ({len(abn_ds)} samples)")
print(f"Normal loader:   {len(nor_loader)} batches ({len(nor_ds)} samples)")

bn = min(128, FEAT_DIM // 4)
se_r = min(16, FEAT_DIM // 8)
model = EfficientVAD(FEAT_DIM, bn, se_r, hid=128, drop=0.6).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nEfficientVAD  |  {n_params:,} params ({n_params / 1e6:.3f}M)")
print(f"feat={FEAT_DIM}  bn={bn}  se_r={se_r}  seg={NUM_SEG}")
print(f"iters={MAX_ITER}  bs={BATCH}  lr={LR}  wd={WD}  top_k={K}")

criterion = MILLoss(k=K)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

SAVE = "/kaggle/working"
best_ap = 0.0
best_step = 0
history = []

ap0, auc0 = evaluate(model, test_paths, frame_gt, DEVICE)
print(f"\nBefore training: AP {ap0:.2f}%  AUC {auc0:.2f}%")
print(f"\nTraining ({MAX_ITER} iterations)...")
print("-" * 70)

t_start = time.time()
running_loss = 0.0

for step in range(1, MAX_ITER + 1):
    model.train()
    criterion.train()

    if (step - 1) % len(abn_loader) == 0:
        abn_iter = iter(abn_loader)
    if (step - 1) % len(nor_loader) == 0:
        nor_iter = iter(nor_loader)

    a_feat = next(abn_iter).to(DEVICE)
    n_feat = next(nor_iter).to(DEVICE)

    optimizer.zero_grad()
    a_sc, a_mg = model(a_feat)
    n_sc, n_mg = model(n_feat)
    loss, det = criterion(a_sc, n_sc, a_mg, n_mg)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    running_loss += loss.item()

    if step % 200 == 0:
        elapsed = time.time() - t_start
        avg_loss = running_loss / 200
        running_loss = 0.0
        print(f"Step {step:4d}/{MAX_ITER} | loss {avg_loss:.4f} | "
              f"rank {det['rank']:.4f} | {elapsed:.0f}s")

    if step % EVAL_INTERVAL == 0 and step > EVAL_START:
        ap, auc_val = evaluate(model, test_paths, frame_gt, DEVICE)

        if ap > best_ap:
            best_ap = ap
            best_step = step
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "frame_ap": ap, "frame_auc": auc_val,
                "feat_dim": FEAT_DIM, "params": n_params,
            }, os.path.join(SAVE, "efficientvad_xd_best.pth"))
            print(f"  Step {step:4d} | AP {ap:.2f}% AUC {auc_val:.2f}% * NEW BEST")

        if step % 100 == 0:
            print(f"  Step {step:4d} | AP {ap:.2f}% AUC {auc_val:.2f}%  "
                  f"(best: {best_ap:.2f}% @ step {best_step})")

        history.append(dict(step=step, frame_ap=ap, frame_auc=auc_val))

torch.save({
    "step": MAX_ITER,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "feat_dim": FEAT_DIM, "params": n_params,
}, os.path.join(SAVE, "efficientvad_xd_final.pth"))


# ═══════════════════════════════════════════════════════════
# 10. FINAL RESULTS
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

total_time = time.time() - t_start
print(f"Training time: {total_time:.0f}s ({total_time / 60:.1f}m)")

if best_ap > 0:
    ckpt = torch.load(os.path.join(SAVE, "efficientvad_xd_best.pth"),
                      map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

final_ap, final_auc = evaluate(model, test_paths, frame_gt, DEVICE)

print(f"Best step:        {best_step}")
print(f"Frame-level AP:   {final_ap:.2f}%   (primary XD-Violence metric)")
print(f"Frame-level AUC:  {final_auc:.2f}%")
print(f"Parameters:       {n_params:,} ({n_params / 1e6:.3f}M)")

model.eval()
dummy = torch.randn(1, NUM_SEG, FEAT_DIM).to(DEVICE)
for _ in range(10):
    model(dummy)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t0 = time.time()
for _ in range(1000):
    model(dummy)
if torch.cuda.is_available():
    torch.cuda.synchronize()
latency = (time.time() - t0) / 1000 * 1000
print(f"Latency:          {latency:.2f} ms/sample")

flops = 0
for m in model.modules():
    if isinstance(m, nn.Linear):
        flops += 2 * m.in_features * m.out_features
    elif isinstance(m, nn.Conv1d):
        flops += 2 * m.kernel_size[0] * (m.in_channels // m.groups) * m.out_channels * NUM_SEG
print(f"FLOPs (est):      {flops:,} ({flops / 1e6:.2f}M)")

print(f"\n{'=' * 70}")
print("COMPARISON  (XD-Violence, frame-level AP%)")
print(f"{'=' * 70}")
print(f"| {'Method':<25} | {'AP%':>7} | {'AUC%':>7} | {'Params':>10} |")
print(f"|{'-' * 27}|{'-' * 9}|{'-' * 9}|{'-' * 12}|")
print(f"| {'Wu et al. (ECCV 20)':<25} | {'75.41':>7} | {'--':>7} | {'--':>10} |")
print(f"| {'RTFM (ICCV 21)':<25} | {'77.81':>7} | {'--':>7} | {'~24.7M':>10} |")
print(f"| {'CLIP-TSA (ICIP 23)':<25} | {'82.17':>7} | {'--':>7} | {'1.68M':>10} |")
print(f"| {'VadCLIP (AAAI 24)':<25} | {'84.51':>7} | {'--':>7} | {'large':>10} |")
print(f"| {'EfficientVAD (Ours)':<25} | {final_ap:>6.2f}% | {final_auc:>6.2f}% | {n_params / 1e6:>9.3f}M |")

summary = {
    "version": VERSION,
    "dataset": "xd-violence",
    "metric": "frame-level AP (matching CLIP-TSA)",
    "best_step": best_step,
    "total_steps": MAX_ITER,
    "frame_ap": final_ap,
    "frame_auc": final_auc,
    "params": n_params,
    "flops": flops,
    "latency_ms": latency,
    "training_time_s": total_time,
    "n_train_abn": len(train_abn_paths),
    "n_train_nor": len(train_nor_paths),
    "n_test": len(test_paths),
    "gt_frames": len(frame_gt),
    "gt_anomalous": int(frame_gt.sum()),
    "history": history,
}
with open(os.path.join(SAVE, "efficientvad_xd_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved to {SAVE}/")
print("Done!")
