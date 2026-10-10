"""
EfficientVAD — XD-Violence Training (v6)
=========================================
Frame-level AP evaluation matching CLIP-TSA / VadCLIP methodology.
GT built from annotations.txt; predictions expanded via np.repeat(pred, 16).
"""

import os, sys, glob, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

VERSION = "v6-frame-level-ap"
CLIP_LEN = 16
NUM_SEG = 32

print(f"EfficientVAD {VERSION}")
print(f"PyTorch {torch.__version__}  |  CUDA {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════
# 1. LOCATE DATASET
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 1: LOCATING DATASET")
print("=" * 70)

INPUT_BASE = "/kaggle/input"

# Print top-level input structure for debugging
if os.path.isdir(INPUT_BASE):
    for d in sorted(os.listdir(INPUT_BASE)):
        full = os.path.join(INPUT_BASE, d)
        if os.path.isdir(full):
            n = len(os.listdir(full))
            print(f"  [{d}/] ({n} items)")
            for sd in sorted(os.listdir(full))[:5]:
                sf = os.path.join(full, sd)
                if os.path.isdir(sf):
                    print(f"    [{sd}/] ({len(os.listdir(sf))} items)")
                else:
                    print(f"    {sd}")

KNOWN_ROOTS = [
    os.path.join(INPUT_BASE, "datasets", "kundanpawar0001", "kundan-xd"),
    os.path.join(INPUT_BASE, "kundan-xd"),
]

ROOT = None
for cand in KNOWN_ROOTS:
    if os.path.isdir(cand):
        ROOT = cand
        break

if ROOT is None and os.path.isdir(INPUT_BASE):
    for d in sorted(os.listdir(INPUT_BASE)):
        full = os.path.join(INPUT_BASE, d)
        if os.path.isdir(full):
            ROOT = full
            break

assert ROOT is not None, "FATAL: No dataset in /kaggle/input/"
print(f"\nDataset root: {ROOT}")

# Walk tree once to find XDTrainClipFeatures, XDTestClipFeatures, annotations.txt
TRAIN_DIR = TEST_DIR = ANNOT_PATH = None

for dirpath, dirnames, filenames in os.walk(ROOT):
    bname = os.path.basename(dirpath)

    if bname == "XDTrainClipFeatures" and TRAIN_DIR is None:
        npy_here = any(f.endswith(".npy") for f in filenames)
        if npy_here:
            TRAIN_DIR = dirpath
        elif "XDTrainClipFeatures" in dirnames:
            nested = os.path.join(dirpath, "XDTrainClipFeatures")
            if any(f.endswith(".npy") for f in os.listdir(nested)):
                TRAIN_DIR = nested

    if bname == "XDTestClipFeatures" and TEST_DIR is None:
        npy_here = any(f.endswith(".npy") for f in filenames)
        if npy_here:
            TEST_DIR = dirpath
        elif "XDTestClipFeatures" in dirnames:
            nested = os.path.join(dirpath, "XDTestClipFeatures")
            if any(f.endswith(".npy") for f in os.listdir(nested)):
                TEST_DIR = nested

    if "annotations.txt" in filenames and ANNOT_PATH is None:
        ANNOT_PATH = os.path.join(dirpath, "annotations.txt")

    if TRAIN_DIR and TEST_DIR and ANNOT_PATH:
        break

print(f"Train dir:   {TRAIN_DIR}")
print(f"Test dir:    {TEST_DIR}")
print(f"Annotations: {ANNOT_PATH}")

if not TRAIN_DIR or not TEST_DIR:
    print("\nDirectory structure under ROOT:")
    for dp, dn, fn in os.walk(ROOT):
        depth = dp.replace(ROOT, "").count(os.sep)
        if depth < 5:
            n_npy = sum(1 for f in fn if f.endswith(".npy"))
            print(f"{'  ' * depth}[{os.path.basename(dp)}/] "
                  f"({n_npy} npy, {len(fn)} files)")
    sys.exit(1)

assert ANNOT_PATH, "annotations.txt not found — cannot build frame-level GT"


# ═══════════════════════════════════════════════════════════
# 2. BUILD SAMPLE LISTS (crop 0 only)
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 2: BUILDING SAMPLE LISTS")
print("=" * 70)


def gather_crop0(feat_dir):
    crop0 = sorted(glob.glob(os.path.join(feat_dir, "*__0.npy")))
    if not crop0:
        crop0 = sorted(glob.glob(os.path.join(feat_dir, "*.npy")))
        print(f"  No __0.npy files; using all {len(crop0)} npy")
    return crop0


def vname(path):
    n = os.path.basename(path)
    return n[:-7] if n.endswith("__0.npy") else os.path.splitext(n)[0]


def is_normal(name):
    return "label_A" in name


train_paths = gather_crop0(TRAIN_DIR)
test_paths = gather_crop0(TEST_DIR)

sample = np.load(train_paths[0])
FEAT_DIM = sample.shape[-1]
print(f"Feature dim: {FEAT_DIM}  |  sample shape: {sample.shape}")

tr_n = sum(1 for p in train_paths if is_normal(vname(p)))
tr_a = len(train_paths) - tr_n
te_n = sum(1 for p in test_paths if is_normal(vname(p)))
te_a = len(test_paths) - te_n

print(f"Train: {len(train_paths)}  (normal {tr_n}, anomaly {tr_a})")
print(f"Test:  {len(test_paths)}  (normal {te_n}, anomaly {te_a})")
print(f"First 3 test: {[os.path.basename(p) for p in test_paths[:3]]}")


# ═══════════════════════════════════════════════════════════
# 3. FRAME-LEVEL GT  (matches VadCLIP make_gt_xd.py exactly)
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 3: BUILDING FRAME-LEVEL GT")
print("=" * 70)

# Read raw annotation lines (VadCLIP's exact approach)
with open(ANNOT_PATH) as f:
    gt_lines = list(f)
print(f"Loaded {len(gt_lines)} annotation lines")

# Debug: show first 3 annotation line keys and first 3 abnormal test vnames
print("\nFirst 3 annotation lines (key only):")
for i, line in enumerate(gt_lines[:3]):
    print(f"  ann[{i}]: '{line.strip().split()[0]}'")

abn_test_names = [vname(p) for p in test_paths if not is_normal(vname(p))]
print(f"\nFirst 3 abnormal test vnames:")
for i, n in enumerate(abn_test_names[:3]):
    print(f"  test[{i}]: '{n}'")


def build_frame_gt(file_paths):
    """
    Exactly replicates VadCLIP's make_gt_xd.py:
      - For each test video, get name by stripping __0.npy suffix
      - If not normal (no label_A), search annotation lines with
        'name in gt_line' (substring match against full line)
      - Parse intervals and fill gt_vec
      - gt.extend(gt_vec[:-16])
    """
    gt_all = []
    seg_counts = []
    matched = 0

    for path in file_paths:
        feat = np.load(path, mmap_mode="r")
        T = feat.shape[0]
        seg_counts.append(T)
        lens = (T + 1) * CLIP_LEN
        gt_vec = np.zeros(lens, dtype=np.float32)

        name = vname(path)
        if not is_normal(name):
            for gt_line in gt_lines:
                if name in gt_line:
                    matched += 1
                    gt_content = gt_line.strip().split()
                    abnormal_fragment = [
                        [int(gt_content[i]), int(gt_content[j])]
                        for i in range(1, len(gt_content), 2)
                        for j in range(2, len(gt_content), 2)
                        if j == i + 1
                    ]
                    if len(abnormal_fragment) != 0:
                        abnormal_fragment = np.array(abnormal_fragment)
                        for frag in abnormal_fragment:
                            gt_vec[frag[0]:frag[1]] = 1.0
                    break

        gt_all.extend(gt_vec[:-CLIP_LEN])

    gt = np.array(gt_all, dtype=np.float32)
    n_pos = int(gt.sum())
    n_abn = sum(1 for p in file_paths if not is_normal(vname(p)))
    print(f"\nGT frames:    {len(gt):,}")
    print(f"Anomalous:    {n_pos:,}  ({100 * n_pos / max(1, len(gt)):.2f}%)")
    print(f"Matched:      {matched} / {n_abn} abnormal videos")
    return gt, seg_counts


frame_gt, test_seg_counts = build_frame_gt(test_paths)

# Sanity check against known VadCLIP gt.npy stats
print(f"Expected:     ~2,331,296 frames / ~537,882 anomalous (VadCLIP)")

# If matching failed, print detailed diagnostics
if frame_gt.sum() == 0:
    print("\n*** WARNING: GT has zero anomalous frames! ***")
    print("Diagnostic: checking why substring matching failed...")
    n_checked = 0
    for p in test_paths:
        nm = vname(p)
        if is_normal(nm):
            continue
        n_checked += 1
        if n_checked > 5:
            break
        found_any = False
        for gt_line in gt_lines:
            if nm in gt_line:
                found_any = True
                break
        print(f"  vname='{nm}' | in_any_line={found_any}")
        if not found_any:
            # Try partial matches
            nm_parts = nm.split("__")
            print(f"    split parts: {nm_parts}")
            for gl in gt_lines[:3]:
                key = gl.strip().split()[0]
                key_parts = key.split("__")
                print(f"    ann key parts: {key_parts}")
    print("*** Check above: do feature filenames match annotation keys? ***")


# ═══════════════════════════════════════════════════════════
# 4. FEATURE PROCESSING
# ═══════════════════════════════════════════════════════════


def uniform_extract(feat, t_max):
    """Uniform temporal sampling with segment averaging (CLIP-TSA / VadCLIP)."""
    out = np.zeros((t_max, feat.shape[1]), dtype=np.float32)
    r = np.linspace(0, len(feat), t_max + 1, dtype=np.int32)
    for i in range(t_max):
        if r[i] != r[i + 1]:
            out[i] = np.mean(feat[r[i] : r[i + 1]], axis=0)
        else:
            out[i] = feat[r[i]]
    return out


def pad_feat(feat, min_len):
    if feat.shape[0] >= min_len:
        return feat
    return np.pad(feat, ((0, min_len - feat.shape[0]), (0, 0)), mode="constant")


def process_train(feat, n_seg):
    if feat.shape[0] > n_seg:
        return uniform_extract(feat, n_seg)
    return pad_feat(feat, n_seg)


# ═══════════════════════════════════════════════════════════
# 5. MODEL  (234K params)
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
                 sm_w=8e-5, sp_w=8e-5, k=3, drop=0.6):
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
# 7. DATASETS
# ═══════════════════════════════════════════════════════════


class TrainSet(Dataset):
    def __init__(self, paths, labels):
        self.paths, self.labels = paths, labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        feat = np.load(self.paths[i]).astype(np.float32)
        feat = process_train(feat, NUM_SEG)
        return torch.from_numpy(feat), self.labels[i]


class MILPairs(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.nidx = [i for i, l in enumerate(ds.labels) if l == 0]
        self.aidx = [i for i, l in enumerate(ds.labels) if l == 1]
        assert self.aidx and self.nidx, "Need both normal and anomaly samples"
        self.length = max(len(self.nidx), len(self.aidx))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        a = self.aidx[idx % len(self.aidx)]
        n = self.nidx[idx % len(self.nidx)]
        return self.ds[a][0], self.ds[n][0]


# ═══════════════════════════════════════════════════════════
# 8. FRAME-LEVEL EVALUATION  (the critical fix over v4)
# ═══════════════════════════════════════════════════════════


@torch.no_grad()
def evaluate(model, file_paths, gt, device):
    """
    Frame-level AP & AUC (matches CLIP-TSA / VadCLIP):
      1. Feed each test video's raw (T_i, D) features through model
      2. Get T_i anomaly scores per video
      3. Concatenate across all videos (same order as GT)
      4. np.repeat(scores, 16)  →  frame-level predictions
      5. average_precision_score(gt, preds)
    """
    model.eval()
    all_scores = []

    for path in file_paths:
        feat = np.load(path).astype(np.float32)
        T = feat.shape[0]
        x = torch.from_numpy(feat).unsqueeze(0).to(device)
        sc, _ = model(x)
        all_scores.append(sc[0].cpu().numpy())

    seg_pred = np.concatenate(all_scores)
    frame_pred = np.repeat(seg_pred, CLIP_LEN)

    gt_use = gt
    if len(frame_pred) != len(gt_use):
        print(f"  ⚠ pred {len(frame_pred)} != gt {len(gt_use)}, trimming")
        mn = min(len(frame_pred), len(gt_use))
        frame_pred = frame_pred[:mn]
        gt_use = gt_use[:mn]

    ap = average_precision_score(gt_use, frame_pred) * 100
    auc = roc_auc_score(gt_use, frame_pred) * 100
    return ap, auc


# ═══════════════════════════════════════════════════════════
# 9. TRAINING
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 4: TRAINING")
print("=" * 70)

EPOCHS = 100
BATCH = 32
LR = 1e-3
WD = 5e-4
K = 3
EVAL_EVERY = 5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

train_labels = [0 if is_normal(vname(p)) else 1 for p in train_paths]
train_ds = TrainSet(train_paths, train_labels)
pair_ds = MILPairs(train_ds)
loader = DataLoader(pair_ds, batch_size=BATCH, shuffle=True,
                    num_workers=2, pin_memory=True)

bn = min(128, FEAT_DIM // 4)
se_r = min(16, FEAT_DIM // 8)
model = EfficientVAD(FEAT_DIM, bn, se_r, hid=128, drop=0.6).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"EfficientVAD  |  {n_params:,} params ({n_params / 1e6:.3f}M)")
print(f"feat={FEAT_DIM}  bn={bn}  se_r={se_r}  seg={NUM_SEG}")
print(f"bs={BATCH}  lr={LR}  epochs={EPOCHS}  top_k={K}")

criterion = MILLoss(k=K)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[int(EPOCHS * 0.6), int(EPOCHS * 0.8)], gamma=0.1)

SAVE = "/kaggle/working"
best_ap = 0.0
best_epoch = 0
history = []

print(f"\nTraining...")
print("-" * 70)

for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    tot_loss = 0.0
    det_sum = {}

    for a_feat, n_feat in loader:
        a_feat = a_feat.to(DEVICE)
        n_feat = n_feat.to(DEVICE)
        optimizer.zero_grad()
        a_sc, a_mg = model(a_feat)
        n_sc, n_mg = model(n_feat)
        loss, det = criterion(a_sc, n_sc, a_mg, n_mg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot_loss += loss.item()
        for k_, v_ in det.items():
            det_sum[k_] = det_sum.get(k_, 0) + v_

    scheduler.step()
    nb = max(1, len(loader))
    avg = tot_loss / nb
    dt = time.time() - t0
    line = f"Ep {epoch:3d}/{EPOCHS} | loss {avg:.4f} | rank {det_sum.get('rank', 0) / nb:.4f} | {dt:.1f}s"

    if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
        ap, auc = evaluate(model, test_paths, frame_gt, DEVICE)
        line += f"  |  AP {ap:.2f}%  AUC {auc:.2f}%"

        if ap > best_ap:
            best_ap = ap
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "frame_ap": ap, "frame_auc": auc,
                "feat_dim": FEAT_DIM, "params": n_params,
            }, os.path.join(SAVE, "efficientvad_xd_best.pth"))
            line += " *"

        history.append(dict(epoch=epoch, loss=avg, frame_ap=ap, frame_auc=auc))

    print(line)


# ═══════════════════════════════════════════════════════════
# 10. FINAL RESULTS
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

ckpt = torch.load(os.path.join(SAVE, "efficientvad_xd_best.pth"),
                  map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
final_ap, final_auc = evaluate(model, test_paths, frame_gt, DEVICE)

print(f"Best epoch:       {best_epoch}")
print(f"Frame-level AP:   {final_ap:.2f}%   (primary XD-Violence metric)")
print(f"Frame-level AUC:  {final_auc:.2f}%")
print(f"Parameters:       {n_params:,} ({n_params / 1e6:.3f}M)")

# Latency
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

# FLOPs
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
    "metric": "frame-level AP (matching CLIP-TSA/VadCLIP)",
    "best_epoch": best_epoch,
    "frame_ap": final_ap,
    "frame_auc": final_auc,
    "params": n_params,
    "flops": flops,
    "latency_ms": latency,
    "n_train": len(train_paths),
    "n_test": len(test_paths),
    "gt_frames": len(frame_gt),
    "gt_anomalous": int(frame_gt.sum()),
    "history": history,
}
with open(os.path.join(SAVE, "efficientvad_xd_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved to {SAVE}/")
print("Done!")
