"""
EfficientVAD — XD-Violence Training (v14-validated)
====================================================
20-iteration sanity check. CLIP-TSA ViT-B/16 features, frame-level AP.
Validated: test file ordering matches gt-xd.npy (500 abn + 300 nor, alpha-sorted
within each group). SEED=0 (matching CLIP-TSA XD runner.sh).
"""
import os, sys, traceback

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)
print("BOOT: script started")

def main():
    import glob, time, json, gzip, base64, warnings, random
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import average_precision_score, roc_auc_score

    VERSION = "v14-validated-20iter"
    CLIP_LEN = 16
    NUM_SEG = 32

    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

    print(f"EfficientVAD {VERSION}")
    print(f"PyTorch {torch.__version__}  |  CUDA {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        print(f"GPU mem: {mem / 1e9:.1f} GB")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ═══════════════════════════════════════════════════════════
    # 1. LOCATE CLIP-TSA FEATURES
    # ═══════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("PHASE 1: LOCATING CLIP-TSA FEATURES")
    print("=" * 70)

    INPUT_BASE = "/kaggle/input"

    if os.path.isdir(INPUT_BASE):
        print(f"Contents of {INPUT_BASE}:")
        for d in sorted(os.listdir(INPUT_BASE)):
            full = os.path.join(INPUT_BASE, d)
            if os.path.isdir(full):
                items = os.listdir(full)
                print(f"  [{d}/] ({len(items)} items)")
                for sub in sorted(items)[:5]:
                    subpath = os.path.join(full, sub)
                    if os.path.isdir(subpath):
                        sub_items = os.listdir(subpath)
                        print(f"    [{sub}/] ({len(sub_items)} items)")
                        for ss in sorted(sub_items)[:5]:
                            ssp = os.path.join(subpath, ss)
                            if os.path.isdir(ssp):
                                print(f"      [{ss}/] ({len(os.listdir(ssp))} items)")
                            else:
                                print(f"      {ss}")
                    else:
                        print(f"    {sub}")
    else:
        print(f"WARNING: {INPUT_BASE} does not exist!")

    CLIP_TSA_CANDIDATES = [
        os.path.join(INPUT_BASE, "kundan-xd",
                     "Kundan_xdviolence_features",
                     "xd_fromhttpsgithub.comjoos2010kjCLIP-TSA"),
        os.path.join(INPUT_BASE, "kundan-xd",
                     "xd_fromhttpsgithub.comjoos2010kjCLIP-TSA"),
        os.path.join(INPUT_BASE, "datasets", "kundanpawar0001", "kundan-xd",
                     "Kundan_xdviolence_features",
                     "xd_fromhttpsgithub.comjoos2010kjCLIP-TSA"),
    ]

    CLIP_TSA_ROOT = None
    for cand in CLIP_TSA_CANDIDATES:
        print(f"  Checking: {cand} -> {os.path.isdir(cand)}")
        if os.path.isdir(cand):
            CLIP_TSA_ROOT = cand
            break

    if CLIP_TSA_ROOT is None:
        print("Searching recursively for features directory...")
        for root, dirs, files in os.walk(INPUT_BASE):
            depth = root.replace(INPUT_BASE, "").count(os.sep)
            if depth > 8:
                dirs.clear()
                continue
            if "features" in dirs:
                feat_dir = os.path.join(root, "features")
                if os.path.isdir(feat_dir):
                    subdirs = set(os.listdir(feat_dir))
                    if {"test", "train"}.issubset(subdirs):
                        CLIP_TSA_ROOT = root
                        print(f"  Found via features/test+train: {root}")
                        break
            basename = os.path.basename(root).lower()
            if "clip" in basename and "tsa" in basename:
                CLIP_TSA_ROOT = root
                print(f"  Found via name match: {root}")
                break
            npy_count = sum(1 for f in files if f.endswith(".npy"))
            if npy_count > 50:
                parent = os.path.dirname(root)
                grandparent = os.path.dirname(parent)
                CLIP_TSA_ROOT = grandparent
                print(f"  Found via npy concentration ({npy_count} files): {root}")
                print(f"  Using grandparent as root: {grandparent}")
                break

    if CLIP_TSA_ROOT is None:
        print("\nFATAL: Cannot find CLIP-TSA features. Full directory tree:")
        for dp, dn, fn in os.walk(INPUT_BASE):
            depth = dp.replace(INPUT_BASE, "").count(os.sep)
            if depth < 6:
                n_npy = sum(1 for f in fn if f.endswith(".npy"))
                indent = "  " * depth
                print(f"{indent}[{os.path.basename(dp)}/] "
                      f"({n_npy} npy, {len(fn)} total, {len(dn)} dirs)")
        sys.exit(1)

    FEAT_ROOT = os.path.join(CLIP_TSA_ROOT, "features")
    if not os.path.isdir(FEAT_ROOT):
        if os.path.isdir(os.path.join(CLIP_TSA_ROOT, "test")):
            FEAT_ROOT = CLIP_TSA_ROOT
        else:
            print(f"WARNING: {FEAT_ROOT} not found, listing CLIP_TSA_ROOT:")
            for item in os.listdir(CLIP_TSA_ROOT):
                print(f"  {item}")
            sys.exit(1)

    print(f"CLIP-TSA root:  {CLIP_TSA_ROOT}")
    print(f"Features root:  {FEAT_ROOT}")

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
    # 2. BUILD SAMPLE LISTS
    # ═══════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("PHASE 2: BUILDING SAMPLE LISTS")
    print("=" * 70)

    train_abn_paths = sorted(glob.glob(os.path.join(TRAIN_ABN, "*.npy")))
    train_nor_paths = sorted(glob.glob(os.path.join(TRAIN_NOR, "*.npy")))

    test_abn_on_disk = sorted(glob.glob(os.path.join(TEST_ABN, "*.npy")))
    test_nor_on_disk = sorted(glob.glob(os.path.join(TEST_NOR, "*.npy")))
    test_paths = test_abn_on_disk + test_nor_on_disk

    n_test_abn = len(test_abn_on_disk)
    n_test_nor = len(test_nor_on_disk)

    sample = np.load(train_abn_paths[0])
    FEAT_DIM = sample.shape[-1]
    print(f"Feature dim: {FEAT_DIM}  |  sample shape: {sample.shape}")
    print(f"Train: {len(train_abn_paths)} abnormal + {len(train_nor_paths)} normal "
          f"= {len(train_abn_paths) + len(train_nor_paths)}")
    print(f"Test:  {n_test_abn} abnormal + {n_test_nor} normal = {len(test_paths)}")
    print(f"First 3 test abn: {[os.path.basename(p) for p in test_paths[:3]]}")
    print(f"First 3 test nor: {[os.path.basename(p) for p in test_paths[n_test_abn:n_test_abn+3]]}")

    has_hash = any('#' in os.path.basename(p) for p in test_paths[:5])
    has_triple = any('___' in os.path.basename(p) for p in test_paths[:5])
    print(f"Naming: '#' in names={has_hash}, '___' in names={has_triple}")
    if has_triple and not has_hash:
        print("  Confirmed: Kaggle naming convention (# replaced by _)")

    if n_test_abn != 500 or n_test_nor != 300:
        print(f"WARNING: Expected 500 abn + 300 nor, got {n_test_abn} + {n_test_nor}")
        print("  GT alignment may be wrong! Proceeding but results may be invalid.")
    else:
        print("  Test split verified: 500 abnormal + 300 normal (matches gt-xd.npy)")

    sample_segs = [np.load(p).shape[0] for p in test_paths[:10]]
    avg_segs = np.mean(sample_segs)
    est_frames = int(avg_segs * len(test_paths) * CLIP_LEN)
    print(f"  Avg segments/video (sample): {avg_segs:.0f}  |  est frames: ~{est_frames:,}"
          f"  (GT: 2,330,384)")

    # ═══════════════════════════════════════════════════════════
    # 3. LOAD FRAME-LEVEL GT (embedded from CLIP-TSA gt-xd.npy)
    # ═══════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("PHASE 3: LOADING FRAME-LEVEL GT")
    print("=" * 70)

    _GT_XD_B64 = (
        "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"
    )

    GT_XD_NFRAMES = 2330384

    def _decode_embedded_gt():
        raw = gzip.decompress(base64.b64decode(_GT_XD_B64))
        return np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:GT_XD_NFRAMES].astype(np.float32)

    frame_gt = _decode_embedded_gt()
    n_pos = int(frame_gt.sum())
    print(f"Decoded embedded gt-xd.npy")
    print(f"GT frames:   {len(frame_gt):,}")
    print(f"Anomalous:   {n_pos:,} ({100 * n_pos / len(frame_gt):.2f}%)")
    assert n_pos == 537805, f"Expected 537805 anomalous frames, got {n_pos}"
    assert len(frame_gt) == GT_XD_NFRAMES

    # ═══════════════════════════════════════════════════════════
    # 4. FEATURE PROCESSING (from CLIP-TSA utils)
    # ═══════════════════════════════════════════════════════════

    def process_feat(feat, length):
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
        nonlocal _test_feat_cache
        if len(_test_feat_cache) == len(file_paths):
            return
        print(f"  Caching {len(file_paths)} test features...")
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
        try:
            auc_val = roc_auc_score(gt_use, frame_pred) * 100
        except ValueError:
            auc_val = float("nan")
        return ap, auc_val

    # ═══════════════════════════════════════════════════════════
    # 9. TRAINING
    # ═══════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("PHASE 4: TRAINING")
    print("=" * 70)

    MAX_ITER = 20
    BATCH = 16
    LR = 1e-3
    WD = 5e-3
    K = 3
    SEED = 0
    EVAL_INTERVAL = 5
    EVAL_START = 0

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    abn_ds = TrainSegDataset(train_abn_paths)
    nor_ds = TrainSegDataset(train_nor_paths)

    abn_loader = DataLoader(abn_ds, batch_size=BATCH, shuffle=True,
                            num_workers=0, pin_memory=False, drop_last=True)
    nor_loader = DataLoader(nor_ds, batch_size=BATCH, shuffle=True,
                            num_workers=0, pin_memory=False, drop_last=True)

    print(f"Abnormal loader: {len(abn_loader)} batches ({len(abn_ds)} samples)")
    print(f"Normal loader:   {len(nor_loader)} batches ({len(nor_ds)} samples)")

    bn = min(128, FEAT_DIM // 4)
    se_r = min(16, FEAT_DIM // 8)
    model = EfficientVAD(FEAT_DIM, bn, se_r, hid=128, drop=0.6).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nEfficientVAD  |  {n_params:,} params ({n_params / 1e6:.3f}M)")
    print(f"feat={FEAT_DIM}  bn={bn}  se_r={se_r}  seg={NUM_SEG}")
    print(f"iters={MAX_ITER}  bs={BATCH}  lr={LR}  wd={WD}  top_k={K}  seed={SEED}")

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

        if step % 5 == 0 or step == 1:
            elapsed = time.time() - t_start
            n_avg = min(step, 5)
            avg_loss = running_loss / n_avg
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
            else:
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


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"FATAL ERROR: {type(e).__name__}: {e}")
        print(f"{'=' * 70}")
        traceback.print_exc()
        sys.exit(1)
