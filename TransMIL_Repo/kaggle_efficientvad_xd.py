"""
EfficientVAD vs CLIP-TSA -- XD-Violence Comparison (v16)
========================================================
Both trained 4000 iterations on same CLIP ViT-B/16 features.
Metric: frame-level AP (XD-Violence standard).
CLIP-TSA faithfully reimplemented from https://github.com/joos2010kj/CLIP-TSA.
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

    VERSION = "v16-comparison-4000iter"
    CLIP_LEN = 16
    NUM_SEG = 32

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    print(f"EfficientVAD vs CLIP-TSA  {VERSION}")
    print(f"PyTorch {torch.__version__}  |  CUDA {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        print(f"GPU mem: {mem / 1e9:.1f} GB")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================================================================
    # 1. LOCATE CLIP-TSA FEATURES
    # ================================================================

    print("\n" + "=" * 60)
    print("PHASE 1: LOCATING FEATURES")
    print("=" * 60)

    INPUT_BASE = "/kaggle/input"

    if os.path.isdir(INPUT_BASE):
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
                        for ss in sorted(sub_items)[:3]:
                            ssp = os.path.join(subpath, ss)
                            if os.path.isdir(ssp):
                                print(f"      [{ss}/] ({len(os.listdir(ssp))} items)")
                            else:
                                print(f"      {ss}")
                    else:
                        print(f"    {sub}")

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
        if os.path.isdir(cand):
            CLIP_TSA_ROOT = cand
            break

    if CLIP_TSA_ROOT is None:
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
                        break
            basename = os.path.basename(root).lower()
            if "clip" in basename and "tsa" in basename:
                CLIP_TSA_ROOT = root
                break
            npy_count = sum(1 for f in files if f.endswith(".npy"))
            if npy_count > 50:
                parent = os.path.dirname(root)
                CLIP_TSA_ROOT = os.path.dirname(parent)
                break

    if CLIP_TSA_ROOT is None:
        print("FATAL: Cannot find CLIP-TSA features.")
        for dp, dn, fn in os.walk(INPUT_BASE):
            depth = dp.replace(INPUT_BASE, "").count(os.sep)
            if depth < 6:
                n_npy = sum(1 for f in fn if f.endswith(".npy"))
                print(f"{'  ' * depth}[{os.path.basename(dp)}/] ({n_npy} npy)")
        sys.exit(1)

    FEAT_ROOT = os.path.join(CLIP_TSA_ROOT, "features")
    if not os.path.isdir(FEAT_ROOT):
        if os.path.isdir(os.path.join(CLIP_TSA_ROOT, "test")):
            FEAT_ROOT = CLIP_TSA_ROOT
        else:
            sys.exit(f"FATAL: {FEAT_ROOT} not found")

    print(f"Features root: {FEAT_ROOT}")

    TRAIN_ABN = os.path.join(FEAT_ROOT, "train", "abnormal")
    TRAIN_NOR = os.path.join(FEAT_ROOT, "train", "normal")
    TEST_ABN = os.path.join(FEAT_ROOT, "test", "abnormal")
    TEST_NOR = os.path.join(FEAT_ROOT, "test", "normal")

    for d, label in [(TRAIN_ABN, "train/abn"), (TRAIN_NOR, "train/nor"),
                      (TEST_ABN, "test/abn"), (TEST_NOR, "test/nor")]:
        if not os.path.isdir(d):
            sys.exit(f"FATAL: Missing {d}")
        n = len([f for f in os.listdir(d) if f.endswith(".npy")])
        print(f"  {label}: {n} files")

    # ================================================================
    # 2. BUILD SAMPLE LISTS
    # ================================================================

    print("\n" + "=" * 60)
    print("PHASE 2: SAMPLE LISTS")
    print("=" * 60)

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
    print(f"Train: {len(train_abn_paths)} abn + {len(train_nor_paths)} nor")
    print(f"Test:  {n_test_abn} abn + {n_test_nor} nor = {len(test_paths)}")

    if n_test_abn != 500 or n_test_nor != 300:
        print(f"WARNING: Expected 500+300, got {n_test_abn}+{n_test_nor}")

    # ================================================================
    # 3. LOAD FRAME-LEVEL GT (embedded)
    # ================================================================

    print("\n" + "=" * 60)
    print("PHASE 3: GROUND TRUTH")
    print("=" * 60)

    _GT_XD_B64 = (
        "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"
    )
    GT_XD_NFRAMES = 2330384

    def _decode_embedded_gt():
        raw = gzip.decompress(base64.b64decode(_GT_XD_B64))
        return np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:GT_XD_NFRAMES].astype(np.float32)

    frame_gt = _decode_embedded_gt()
    n_pos = int(frame_gt.sum())
    print(f"GT: {len(frame_gt):,} frames, {n_pos:,} anomalous ({100*n_pos/len(frame_gt):.2f}%)")
    assert n_pos == 537805 and len(frame_gt) == GT_XD_NFRAMES

    # ================================================================
    # 4. UTILS
    # ================================================================

    def process_feat(feat, length):
        new_feat = np.zeros((length, feat.shape[1]), dtype=np.float32)
        r = np.linspace(0, len(feat), length + 1, dtype=np.int32)
        for i in range(length):
            if r[i] != r[i + 1]:
                new_feat[i] = np.mean(feat[r[i]:r[i + 1]], axis=0)
            else:
                new_feat[i] = feat[r[i]]
        return new_feat

    # ================================================================
    # 5A. MODEL: EfficientVAD (234K params)
    # ================================================================

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

    # ================================================================
    # 5B. MODEL: CLIP-TSA (faithful reimplementation)
    # ================================================================

    class PerturbedTopKFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, k, num_samples=100, sigma=0.05, train_mode=True):
            b, d = x.shape
            k = int(d * k)
            noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)
            perturbed_x = x[:, None, :] + noise * sigma
            k = max(1, min(k, perturbed_x.shape[-1]))
            if not train_mode:
                k = min(1000, k)
            topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
            indices = torch.sort(topk_results.indices, dim=-1).values
            perturbed_output = F.one_hot(indices, num_classes=d).float()
            indicators = perturbed_output.mean(dim=1)
            ctx.k = k
            ctx.num_samples = num_samples
            ctx.sigma = sigma
            ctx.perturbed_output = perturbed_output
            ctx.noise = noise
            return indicators

        @staticmethod
        def backward(ctx, grad_output):
            if grad_output is None:
                return tuple([None] * 5)
            grad_expected = torch.einsum("bnkd,bne->bkde",
                                         ctx.perturbed_output, ctx.noise)
            grad_expected /= (ctx.num_samples * ctx.sigma)
            grad_input = torch.einsum("bkde,bke->bd",
                                       grad_expected, grad_output)
            return (grad_input,) + tuple([None] * 5)

    class PerturbedTopK(nn.Module):
        def __init__(self, k, num_samples=100, sigma=0.05):
            super().__init__()
            self.num_samples = num_samples
            self.sigma = sigma
            self.k = k

        def forward(self, x, train_mode):
            return PerturbedTopKFunction.apply(
                x, self.k, self.num_samples, self.sigma, train_mode)

    class HAScorer(nn.Module):
        def __init__(self, input_dim=512):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 256)
            self.fc2 = nn.Linear(256, 1)

        def forward(self, x):
            return torch.sigmoid(self.fc2(F.relu(self.fc1(x))))

    class HardAttention(nn.Module):
        def __init__(self, k=1.0, num_samples=100, input_dim=512):
            super().__init__()
            self.scorer = HAScorer(input_dim)
            self.hard_att = PerturbedTopK(k=k, num_samples=num_samples)

        def forward(self, inputs):
            scores = self.scorer(inputs)
            b = scores.size(0)
            topk = self.hard_att(scores.squeeze(-1), train_mode=(b > 1))
            out = topk.unsqueeze(-1) * inputs.unsqueeze(1)
            return torch.sum(out, dim=1)

    class NONLocalBlock1D(nn.Module):
        def __init__(self, in_channels, inter_channels=None,
                     sub_sample=False, bn_layer=True):
            super().__init__()
            self.in_channels = in_channels
            self.inter_channels = inter_channels or in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

            self.g = nn.Conv1d(self.in_channels, self.inter_channels, 1)
            if bn_layer:
                self.W = nn.Sequential(
                    nn.Conv1d(self.inter_channels, self.in_channels, 1),
                    nn.BatchNorm1d(self.in_channels))
                nn.init.constant_(self.W[1].weight, 0)
                nn.init.constant_(self.W[1].bias, 0)
            else:
                self.W = nn.Conv1d(self.inter_channels, self.in_channels, 1)
                nn.init.constant_(self.W.weight, 0)
                nn.init.constant_(self.W.bias, 0)
            self.theta = nn.Conv1d(self.in_channels, self.inter_channels, 1)
            self.phi = nn.Conv1d(self.in_channels, self.inter_channels, 1)

            if sub_sample:
                self.g = nn.Sequential(self.g, nn.MaxPool1d(2))
                self.phi = nn.Sequential(self.phi, nn.MaxPool1d(2))

        def forward(self, x):
            B = x.size(0)
            ic = self.inter_channels
            g_x = self.g(x).view(B, ic, -1).permute(0, 2, 1)
            theta_x = self.theta(x).view(B, ic, -1).permute(0, 2, 1)
            phi_x = self.phi(x).view(B, ic, -1)
            f = torch.matmul(theta_x, phi_x) / phi_x.size(-1)
            y = torch.matmul(f, g_x).permute(0, 2, 1).contiguous()
            y = y.view(B, ic, *x.size()[2:])
            return self.W(y) + x

    class Aggregate(nn.Module):
        def __init__(self, len_feature):
            super().__init__()
            div = 2048 // len_feature
            ch = 512 // div
            ch_full = 2048 // div
            self.conv_1 = nn.Sequential(
                nn.Conv1d(len_feature, ch, 3, padding=1, dilation=1),
                nn.ReLU(), nn.BatchNorm1d(ch))
            self.conv_2 = nn.Sequential(
                nn.Conv1d(len_feature, ch, 3, padding=2, dilation=2),
                nn.ReLU(), nn.BatchNorm1d(ch))
            self.conv_3 = nn.Sequential(
                nn.Conv1d(len_feature, ch, 3, padding=4, dilation=4),
                nn.ReLU(), nn.BatchNorm1d(ch))
            self.conv_4 = nn.Sequential(
                nn.Conv1d(ch_full, ch, 1, bias=False), nn.ReLU())
            self.conv_5 = nn.Sequential(
                nn.Conv1d(ch_full, ch_full, 3, padding=1, bias=False),
                nn.ReLU(), nn.BatchNorm1d(ch_full))
            self.non_local = NONLocalBlock1D(ch, sub_sample=False, bn_layer=True)

        def forward(self, x):
            out = x.permute(0, 2, 1)
            residual = out
            out1 = self.conv_1(out)
            out2 = self.conv_2(out)
            out3 = self.conv_3(out)
            out_d = torch.cat((out1, out2, out3), dim=1)
            out = self.conv_4(out)
            out = self.non_local(out)
            out = torch.cat((out_d, out), dim=1)
            out = self.conv_5(out)
            out = out + residual
            return out.permute(0, 2, 1)

    class DimReductionMLP(nn.Module):
        def __init__(self, input_dim=512):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 1024)
            self.fc2 = nn.Linear(1024, 512)

        def forward(self, x):
            return torch.sigmoid(self.fc2(F.relu(self.fc1(x))))

    def _cliptsa_weight_init(m):
        cn = m.__class__.__name__
        if cn.find('Conv') != -1 or cn.find('Linear') != -1:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)

    class CLIPTSAModel(nn.Module):
        def __init__(self, n_features, batch_size, k=0.9,
                     num_samples=100, apply_HA=True):
            super().__init__()
            self.batch_size = batch_size
            self.k_abn = NUM_SEG // 10
            self.k_nor = NUM_SEG // 10
            self.division = 2048 // n_features
            self.apply_HA = apply_HA
            d = self.division
            self.hard_attention = HardAttention(
                k=k, num_samples=num_samples, input_dim=n_features)
            self.Aggregate = Aggregate(len_feature=2048 // d)
            self.fc1 = nn.Linear(n_features, 512 // d)
            self.fc2 = nn.Linear(512 // d, 128 // d)
            self.fc3 = nn.Linear(128 // d, 1)
            self.drop_out = nn.Dropout(0.7)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.mlp = DimReductionMLP(input_dim=n_features)
            self.apply(_cliptsa_weight_init)

        def forward(self, inputs):
            k_abn = self.k_abn
            k_nor = self.k_nor
            out = inputs
            bs, ncrops, t, f = out.size()
            out = out.view(-1, t, f)

            if f > 512:
                out = self.mlp(out)
                f = 512

            if self.apply_HA:
                out = self.hard_attention(out)

            out = self.Aggregate(out)
            out = self.drop_out(out)
            features = out

            scores = self.relu(self.fc1(features))
            scores = self.drop_out(scores)
            scores = self.relu(self.fc2(scores))
            scores = self.drop_out(scores)
            scores = self.sigmoid(self.fc3(scores))
            scores = scores.view(bs, ncrops, -1).mean(1)
            scores = scores.unsqueeze(dim=2)

            normal_features = features[0:self.batch_size * ncrops]
            normal_scores = scores[0:self.batch_size]
            abnormal_features = features[self.batch_size * ncrops:]
            abnormal_scores = scores[self.batch_size:]

            feat_magnitudes = torch.norm(features, p=2, dim=2)
            feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)
            nfea_magnitudes = feat_magnitudes[0:self.batch_size]
            afea_magnitudes = feat_magnitudes[self.batch_size:]
            n_size = nfea_magnitudes.shape[0]

            if n_size == 1:
                afea_magnitudes = nfea_magnitudes
                abnormal_scores = normal_scores
                abnormal_features = normal_features

            select_idx = torch.ones_like(nfea_magnitudes)
            select_idx = self.drop_out(select_idx)
            afea_magnitudes_drop = afea_magnitudes * select_idx
            idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]
            idx_abn_feat = idx_abn.unsqueeze(2).expand(
                -1, -1, abnormal_features.shape[2])

            abn_reshaped = abnormal_features.view(n_size, ncrops, t, f)
            abn_reshaped = abn_reshaped.permute(1, 0, 2, 3)
            abn_list = []
            for abf in abn_reshaped:
                abn_list.append(torch.gather(abf, 1, idx_abn_feat))
            feat_select_abn = torch.cat(abn_list, dim=0)

            idx_abn_score = idx_abn.unsqueeze(2).expand(
                -1, -1, abnormal_scores.shape[2])
            score_abnormal = torch.mean(
                torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)

            select_idx_n = torch.ones_like(nfea_magnitudes)
            select_idx_n = self.drop_out(select_idx_n)
            nfea_magnitudes_drop = nfea_magnitudes * select_idx_n
            idx_nor = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]
            idx_nor_feat = idx_nor.unsqueeze(2).expand(
                -1, -1, normal_features.shape[2])

            nor_reshaped = normal_features.view(n_size, ncrops, t, f)
            nor_reshaped = nor_reshaped.permute(1, 0, 2, 3)
            nor_list = []
            for nf in nor_reshaped:
                nor_list.append(torch.gather(nf, 1, idx_nor_feat))
            feat_select_nor = torch.cat(nor_list, dim=0)

            idx_nor_score = idx_nor.unsqueeze(2).expand(
                -1, -1, normal_scores.shape[2])
            score_normal = torch.mean(
                torch.gather(normal_scores, 1, idx_nor_score), dim=1)

            return (score_abnormal, score_normal, feat_select_abn,
                    feat_select_nor, feat_select_abn, feat_select_abn,
                    scores, feat_select_abn, feat_select_abn,
                    feat_magnitudes)

    # ================================================================
    # 6A. LOSS: EfficientVAD MIL Ranking
    # ================================================================

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

    # ================================================================
    # 6B. LOSS: CLIP-TSA RTFM
    # ================================================================

    class RTFMLoss(nn.Module):
        def __init__(self, alpha=0.0001, margin=100):
            super().__init__()
            self.alpha = alpha
            self.margin = margin
            self.criterion = nn.BCELoss()

        def forward(self, score_normal, score_abnormal, nlabel, alabel,
                    feat_n, feat_a):
            label = torch.cat((nlabel, alabel), 0)
            score = torch.cat((score_normal, score_abnormal), 0).squeeze()
            loss_cls = self.criterion(score, label)
            loss_abn = torch.abs(
                self.margin -
                torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1))
            loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)
            loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)
            return loss_cls + self.alpha * loss_rtfm

    def ctsa_sparsity(arr, lamda2=8e-3):
        return lamda2 * torch.mean(torch.norm(arr, dim=0))

    def ctsa_smooth(arr, lamda1=8e-4):
        arr2 = torch.zeros_like(arr)
        arr2[:-1] = arr[1:]
        arr2[-1] = arr[-1]
        return lamda1 * torch.sum((arr2 - arr) ** 2)

    # ================================================================
    # 7. DATASET
    # ================================================================

    class TrainSegDataset(Dataset):
        def __init__(self, paths):
            self.paths = paths
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, i):
            feat = np.load(self.paths[i]).astype(np.float32)
            feat = process_feat(feat, NUM_SEG)
            return torch.from_numpy(feat)

    # ================================================================
    # 8. EVALUATION
    # ================================================================

    _test_feat_cache = []

    def _cache_test_features(file_paths):
        nonlocal _test_feat_cache
        if len(_test_feat_cache) == len(file_paths):
            return
        print(f"  Caching {len(file_paths)} test features...")
        _test_feat_cache = []
        for path in file_paths:
            _test_feat_cache.append(np.load(path).astype(np.float32))
        mem_mb = sum(f.nbytes for f in _test_feat_cache) / 1e6
        print(f"  Cached: {mem_mb:.1f} MB")

    def _frame_metrics(all_scores, gt):
        seg_pred = np.concatenate(all_scores)
        frame_pred = np.repeat(seg_pred, CLIP_LEN)
        mn = min(len(frame_pred), len(gt))
        frame_pred, gt_use = frame_pred[:mn], gt[:mn]
        ap = average_precision_score(gt_use, frame_pred) * 100
        try:
            auc_val = roc_auc_score(gt_use, frame_pred) * 100
        except ValueError:
            auc_val = float("nan")
        return ap, auc_val

    @torch.no_grad()
    def evaluate_evad(model, file_paths, gt, device):
        model.eval()
        _cache_test_features(file_paths)
        all_scores = []
        for feat in _test_feat_cache:
            x = torch.from_numpy(feat).unsqueeze(0).to(device)
            sc, _ = model(x)
            all_scores.append(sc[0].cpu().numpy())
        return _frame_metrics(all_scores, gt)

    @torch.no_grad()
    def evaluate_ctsa(model, file_paths, gt, device):
        model.eval()
        _cache_test_features(file_paths)
        all_scores = []
        for feat in _test_feat_cache:
            x = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).to(device)
            outputs = model(x)
            sc = outputs[6].squeeze().cpu().numpy()
            all_scores.append(sc)
        return _frame_metrics(all_scores, gt)

    # ================================================================
    # 9. SHARED HYPERPARAMS
    # ================================================================

    MAX_ITER = 4000
    BATCH = 16
    LR = 1e-3
    WD = 5e-3
    SEED = 0
    EVAL_INTERVAL = 50
    EVAL_START = 200
    SAVE = "/kaggle/working"

    # ================================================================
    # 10. TRAIN EFFICIENTVAD
    # ================================================================

    print("\n" + "=" * 60)
    print("PHASE A: TRAINING EfficientVAD (4000 iter)")
    print("=" * 60)

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

    print(f"Abn loader: {len(abn_loader)} batches  |  "
          f"Nor loader: {len(nor_loader)} batches")

    bn = min(128, FEAT_DIM // 4)
    se_r = min(16, FEAT_DIM // 8)
    evad_model = EfficientVAD(FEAT_DIM, bn, se_r, hid=128, drop=0.6).to(DEVICE)
    evad_params = sum(p.numel() for p in evad_model.parameters())
    print(f"EfficientVAD: {evad_params:,} params ({evad_params/1e6:.3f}M)")

    evad_criterion = MILLoss(k=3)
    evad_optimizer = optim.Adam(evad_model.parameters(), lr=LR, weight_decay=WD)

    evad_best_ap = 0.0
    evad_best_step = 0
    evad_history = []

    ap0, auc0 = evaluate_evad(evad_model, test_paths, frame_gt, DEVICE)
    print(f"Before training: AP {ap0:.2f}% AUC {auc0:.2f}%")

    t_start = time.time()
    running_loss = 0.0

    for step in range(1, MAX_ITER + 1):
        evad_model.train()
        evad_criterion.train()

        if (step - 1) % len(abn_loader) == 0:
            abn_iter = iter(abn_loader)
        if (step - 1) % len(nor_loader) == 0:
            nor_iter = iter(nor_loader)

        a_feat = next(abn_iter).to(DEVICE)
        n_feat = next(nor_iter).to(DEVICE)

        evad_optimizer.zero_grad()
        a_sc, a_mg = evad_model(a_feat)
        n_sc, n_mg = evad_model(n_feat)
        loss, det = evad_criterion(a_sc, n_sc, a_mg, n_mg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(evad_model.parameters(), 1.0)
        evad_optimizer.step()
        running_loss += loss.item()

        if step % 200 == 0 or step == 1:
            elapsed = time.time() - t_start
            avg = running_loss / min(step, 200)
            running_loss = 0.0
            print(f"[EVAD] Step {step:4d}/{MAX_ITER} | loss {avg:.4f} | "
                  f"{elapsed:.0f}s")

        should_eval = ((step >= EVAL_START and step % EVAL_INTERVAL == 0)
                       or step == MAX_ITER)
        if should_eval:
            ap, auc_val = evaluate_evad(
                evad_model, test_paths, frame_gt, DEVICE)
            if ap > evad_best_ap:
                evad_best_ap = ap
                evad_best_step = step
                torch.save({
                    "step": step,
                    "model_state_dict": evad_model.state_dict(),
                    "frame_ap": float(ap), "frame_auc": float(auc_val),
                    "feat_dim": int(FEAT_DIM), "params": int(evad_params),
                }, os.path.join(SAVE, "efficientvad_xd_best.pth"))
                print(f"[EVAD] Step {step:4d} | AP {ap:.2f}% AUC {auc_val:.2f}%"
                      f" * BEST")
            elif step % 200 == 0 or step == MAX_ITER:
                print(f"[EVAD] Step {step:4d} | AP {ap:.2f}% AUC {auc_val:.2f}%"
                      f"  (best: {evad_best_ap:.2f}% @{evad_best_step})")
            evad_history.append(dict(step=step, ap=ap, auc=auc_val))

    evad_time = time.time() - t_start

    if evad_best_ap > 0:
        ckpt = torch.load(os.path.join(SAVE, "efficientvad_xd_best.pth"),
                          map_location=DEVICE, weights_only=False)
        evad_model.load_state_dict(ckpt["model_state_dict"])
    evad_final_ap, evad_final_auc = evaluate_evad(
        evad_model, test_paths, frame_gt, DEVICE)
    print(f"\n[EVAD] DONE: AP {evad_final_ap:.2f}% AUC {evad_final_auc:.2f}%"
          f" | best@{evad_best_step} | {evad_time:.0f}s")

    del evad_model, evad_criterion, evad_optimizer
    torch.cuda.empty_cache()

    # ================================================================
    # 11. TRAIN CLIP-TSA
    # ================================================================

    print("\n" + "=" * 60)
    print("PHASE B: TRAINING CLIP-TSA (4000 iter)")
    print("=" * 60)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    abn_loader2 = DataLoader(abn_ds, batch_size=BATCH, shuffle=True,
                             num_workers=0, pin_memory=False, drop_last=True)
    nor_loader2 = DataLoader(nor_ds, batch_size=BATCH, shuffle=True,
                             num_workers=0, pin_memory=False, drop_last=True)

    ctsa_model = CLIPTSAModel(
        FEAT_DIM, BATCH, k=0.9, num_samples=100, apply_HA=True).to(DEVICE)
    ctsa_total_params = sum(p.numel() for p in ctsa_model.parameters())
    ctsa_active_params = ctsa_total_params - sum(
        p.numel() for p in ctsa_model.mlp.parameters())
    print(f"CLIP-TSA: {ctsa_total_params:,} total params "
          f"({ctsa_active_params:,} active, MLP unused for ViT)")

    rtfm_loss_fn = RTFMLoss(alpha=0.0001, margin=100)
    ctsa_optimizer = optim.Adam(
        ctsa_model.parameters(), lr=LR, weight_decay=WD)

    ctsa_best_ap = 0.0
    ctsa_best_step = 0
    ctsa_history = []

    ap0, auc0 = evaluate_ctsa(ctsa_model, test_paths, frame_gt, DEVICE)
    print(f"Before training: AP {ap0:.2f}% AUC {auc0:.2f}%")

    nlabel_t = torch.zeros(BATCH).to(DEVICE)
    alabel_t = torch.ones(BATCH).to(DEVICE)

    t_start2 = time.time()
    running_loss2 = 0.0

    for step in range(1, MAX_ITER + 1):
        ctsa_model.train()

        if (step - 1) % len(abn_loader2) == 0:
            abn_iter2 = iter(abn_loader2)
        if (step - 1) % len(nor_loader2) == 0:
            nor_iter2 = iter(nor_loader2)

        a_feat = next(abn_iter2).to(DEVICE)
        n_feat = next(nor_iter2).to(DEVICE)

        inp = torch.cat((n_feat.unsqueeze(1), a_feat.unsqueeze(1)), 0)

        (score_abn, score_nor, feat_sel_abn, feat_sel_nor,
         _, _, scores_all, _, _, _) = ctsa_model(inp)

        scores_flat = scores_all.view(BATCH * NUM_SEG * 2, -1).squeeze()
        abn_scores_flat = scores_flat[BATCH * NUM_SEG:]

        loss_rtfm = rtfm_loss_fn(
            score_nor, score_abn, nlabel_t, alabel_t,
            feat_sel_nor, feat_sel_abn)
        loss_sp = ctsa_sparsity(abn_scores_flat)
        loss_sm = ctsa_smooth(abn_scores_flat)
        cost = loss_rtfm + loss_sp + loss_sm

        ctsa_optimizer.zero_grad()
        cost.backward()
        ctsa_optimizer.step()
        running_loss2 += cost.item()

        if step % 200 == 0 or step == 1:
            elapsed = time.time() - t_start2
            avg = running_loss2 / min(step, 200)
            running_loss2 = 0.0
            print(f"[CTSA] Step {step:4d}/{MAX_ITER} | loss {avg:.4f} | "
                  f"{elapsed:.0f}s")

        should_eval = ((step >= EVAL_START and step % EVAL_INTERVAL == 0)
                       or step == MAX_ITER)
        if should_eval:
            ap, auc_val = evaluate_ctsa(
                ctsa_model, test_paths, frame_gt, DEVICE)
            if ap > ctsa_best_ap:
                ctsa_best_ap = ap
                ctsa_best_step = step
                torch.save({
                    "step": step,
                    "model_state_dict": ctsa_model.state_dict(),
                    "frame_ap": float(ap), "frame_auc": float(auc_val),
                    "params": int(ctsa_active_params),
                }, os.path.join(SAVE, "cliptsa_xd_best.pth"))
                print(f"[CTSA] Step {step:4d} | AP {ap:.2f}% AUC {auc_val:.2f}%"
                      f" * BEST")
            elif step % 200 == 0 or step == MAX_ITER:
                print(f"[CTSA] Step {step:4d} | AP {ap:.2f}% AUC {auc_val:.2f}%"
                      f"  (best: {ctsa_best_ap:.2f}% @{ctsa_best_step})")
            ctsa_history.append(dict(step=step, ap=ap, auc=auc_val))

    ctsa_time = time.time() - t_start2

    if ctsa_best_ap > 0:
        ckpt = torch.load(os.path.join(SAVE, "cliptsa_xd_best.pth"),
                          map_location=DEVICE, weights_only=False)
        ctsa_model.load_state_dict(ckpt["model_state_dict"])
    ctsa_final_ap, ctsa_final_auc = evaluate_ctsa(
        ctsa_model, test_paths, frame_gt, DEVICE)
    print(f"\n[CTSA] DONE: AP {ctsa_final_ap:.2f}% AUC {ctsa_final_auc:.2f}%"
          f" | best@{ctsa_best_step} | {ctsa_time:.0f}s")

    # ================================================================
    # 12. LATENCY & FLOPS (EfficientVAD only — our contribution)
    # ================================================================

    print("\n" + "=" * 60)
    print("EFFICIENCY METRICS (EfficientVAD)")
    print("=" * 60)

    if evad_best_ap > 0:
        ckpt = torch.load(os.path.join(SAVE, "efficientvad_xd_best.pth"),
                          map_location=DEVICE, weights_only=False)
    evad_bench = EfficientVAD(FEAT_DIM, bn, se_r, hid=128, drop=0.6).to(DEVICE)
    if evad_best_ap > 0:
        evad_bench.load_state_dict(ckpt["model_state_dict"])
    evad_bench.eval()

    dummy = torch.randn(1, NUM_SEG, FEAT_DIM).to(DEVICE)
    for _ in range(10):
        evad_bench(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(1000):
        evad_bench(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = (time.time() - t0) / 1000 * 1000
    print(f"Latency: {latency:.2f} ms/sample (32-seg input)")

    flops = 0
    for m in evad_bench.modules():
        if isinstance(m, nn.Linear):
            flops += 2 * m.in_features * m.out_features
        elif isinstance(m, nn.Conv1d):
            flops += (2 * m.kernel_size[0] * (m.in_channels // m.groups)
                      * m.out_channels * NUM_SEG)
    print(f"FLOPs (est): {flops:,} ({flops/1e6:.2f}M)")
    del evad_bench

    # ================================================================
    # 13. FINAL COMPARISON
    # ================================================================

    print("\n" + "=" * 60)
    print("FINAL COMPARISON  (XD-Violence, frame-level AP%)")
    print("=" * 60)
    print(f"| {'Method':<25} | {'AP%':>7} | {'AUC%':>7} | {'Params':>10} |"
          f" {'Time':>6} |")
    print(f"|{'-'*27}|{'-'*9}|{'-'*9}|{'-'*12}|{'-'*8}|")
    print(f"| {'Wu et al. (ECCV 20)':<25} | {'75.41':>7} | {'--':>7} |"
          f" {'--':>10} | {'--':>6} |")
    print(f"| {'RTFM (ICCV 21)':<25} | {'77.81':>7} | {'--':>7} |"
          f" {'~24.7M':>10} | {'--':>6} |")
    print(f"| {'CLIP-TSA (paper)':<25} | {'82.17':>7} | {'--':>7} |"
          f" {'1.68M':>10} | {'--':>6} |")
    print(f"| {'CLIP-TSA (ours)':<25} | {ctsa_final_ap:>6.2f}% |"
          f" {ctsa_final_auc:>6.2f}% | {ctsa_active_params/1e6:>9.3f}M |"
          f" {ctsa_time/60:>5.1f}m |")
    print(f"| {'EfficientVAD (ours)':<25} | {evad_final_ap:>6.2f}% |"
          f" {evad_final_auc:>6.2f}% | {evad_params/1e6:>9.3f}M |"
          f" {evad_time/60:>5.1f}m |")

    gap = evad_final_ap - ctsa_final_ap
    param_ratio = ctsa_active_params / evad_params
    print(f"\nAP gap: {gap:+.2f}% (EfficientVAD vs CLIP-TSA ours)")
    print(f"Param ratio: CLIP-TSA is {param_ratio:.1f}x larger")
    print(f"EfficientVAD: {evad_params:,} params | "
          f"Latency: {latency:.2f}ms | FLOPs: {flops/1e6:.2f}M")

    summary = {
        "version": VERSION,
        "dataset": "xd-violence",
        "metric": "frame-level AP",
        "efficientvad": {
            "best_step": evad_best_step,
            "frame_ap": evad_final_ap,
            "frame_auc": evad_final_auc,
            "params": evad_params,
            "flops": flops,
            "latency_ms": latency,
            "training_time_s": evad_time,
            "history": evad_history,
        },
        "cliptsa": {
            "best_step": ctsa_best_step,
            "frame_ap": ctsa_final_ap,
            "frame_auc": ctsa_final_auc,
            "params_active": ctsa_active_params,
            "params_total": ctsa_total_params,
            "training_time_s": ctsa_time,
            "history": ctsa_history,
        },
        "total_steps": MAX_ITER,
        "seed": SEED,
        "batch_size": BATCH,
        "n_train_abn": len(train_abn_paths),
        "n_train_nor": len(train_nor_paths),
        "n_test": len(test_paths),
    }
    with open(os.path.join(SAVE, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {SAVE}/")
    print("Done!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"FATAL ERROR: {type(e).__name__}: {e}")
        print(f"{'=' * 60}")
        traceback.print_exc()
        sys.exit(1)
