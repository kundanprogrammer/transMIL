print("ALIVE", flush=True)
"""
CLIP-TSA Original Code — XD-Violence (Kaggle T4)
=================================================
This runs the EXACT original CLIP-TSA code from
https://github.com/joos2010kj/CLIP-TSA
with only path adaptations for Kaggle environment.
No reimplementation — code is copy-pasted from the repo.
"""
import os, sys, traceback

# Write a sentinel file IMMEDIATELY so we know the script started
try:
    with open("/kaggle/working/_boot.txt", "w") as _bf:
        _bf.write("boot_started\n")
except Exception:
    pass

print("BOOT: CLIP-TSA original code started", flush=True)
sys.stdout.flush()
sys.stderr.flush()

def main():
    print("MAIN: entered main()", flush=True)
    import glob, time, json, gzip, base64, warnings, random
    print("MAIN: stdlib imports done", flush=True)
    import numpy as np
    print("MAIN: numpy imported", flush=True)
    import torch
    print(f"MAIN: torch {torch.__version__} imported", flush=True)
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.nn.init as torch_init
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.nn import MSELoss
    from sklearn.metrics import auc, roc_curve, precision_recall_curve
    print("MAIN: all imports done", flush=True)

    np.int = int

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    print(f"PyTorch {torch.__version__}  |  CUDA {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        print(f"GPU mem: {mem / 1e9:.1f} GB")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CLIP_LEN = 16

    # Set default tensor type as original code does
    # torch.set_default_tensor_type deprecated in PyTorch 2.x, use alternatives
    try:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    except Exception:
        torch.set_default_device('cuda')
        torch.set_default_dtype(torch.float32)

    # ================================================================
    # LOCATE FEATURES (same as our v15/v16 logic)
    # ================================================================

    print("\n" + "=" * 60)
    print("PHASE 1: LOCATING FEATURES")
    print("=" * 60)

    INPUT_BASE = "/kaggle/input"

    # Robust feature locating (same logic as v16)
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

    if CLIP_TSA_ROOT is None:
        print("FATAL: Cannot find CLIP-TSA features. Scanning tree:")
        for dp, dn, fn in os.walk(INPUT_BASE):
            depth = dp.replace(INPUT_BASE, "").count(os.sep)
            if depth < 6:
                n_npy = sum(1 for f in fn if f.endswith(".npy"))
                print(f"{'  ' * depth}[{os.path.basename(dp)}/] ({n_npy} npy)")
        raise RuntimeError("Cannot find CLIP-TSA features")

    FEAT_ROOT = os.path.join(CLIP_TSA_ROOT, "features")
    if not os.path.isdir(FEAT_ROOT):
        if os.path.isdir(os.path.join(CLIP_TSA_ROOT, "test")):
            FEAT_ROOT = CLIP_TSA_ROOT
        else:
            raise RuntimeError(f"{FEAT_ROOT} not found")

    print(f"Features root: {FEAT_ROOT}")

    # Dataset uses "abnormal"/"normal" (full names), not "abn"/"nor"
    train_abn_dir = os.path.join(FEAT_ROOT, "train", "abnormal")
    train_nor_dir = os.path.join(FEAT_ROOT, "train", "normal")
    test_abn_dir = os.path.join(FEAT_ROOT, "test", "abnormal")
    test_nor_dir = os.path.join(FEAT_ROOT, "test", "normal")
    # Fallback to short names if full names don't exist
    if not os.path.isdir(train_abn_dir):
        train_abn_dir = os.path.join(FEAT_ROOT, "train", "abn")
        train_nor_dir = os.path.join(FEAT_ROOT, "train", "nor")
        test_abn_dir = os.path.join(FEAT_ROOT, "test", "abn")
        test_nor_dir = os.path.join(FEAT_ROOT, "test", "nor")

    for d, label in [(train_abn_dir, "train/abn"), (train_nor_dir, "train/nor"),
                     (test_abn_dir, "test/abn"), (test_nor_dir, "test/nor")]:
        n = len([f for f in os.listdir(d) if f.endswith(".npy")])
        print(f"  {label}: {n} files")

    # ================================================================
    # BUILD FILE LISTS (matching original dataset.py format)
    # ================================================================

    print("\n" + "=" * 60)
    print("PHASE 2: SAMPLE LISTS")
    print("=" * 60)

    train_abn_paths = sorted(glob.glob(os.path.join(train_abn_dir, "*.npy")))
    train_nor_paths = sorted(glob.glob(os.path.join(train_nor_dir, "*.npy")))
    test_abn_paths = sorted(glob.glob(os.path.join(test_abn_dir, "*.npy")))
    test_nor_paths = sorted(glob.glob(os.path.join(test_nor_dir, "*.npy")))

    sample = np.load(train_abn_paths[0])
    FEAT_DIM = sample.shape[-1]
    print(f"Feature dim: {FEAT_DIM}  |  sample shape: {sample.shape}")
    print(f"Train: {len(train_abn_paths)} abn + {len(train_nor_paths)} nor")
    print(f"Test:  {len(test_abn_paths)} abn + {len(test_nor_paths)} nor = "
          f"{len(test_abn_paths) + len(test_nor_paths)}")

    # Test list: abnormal first, then normal (same as original dataset.py)
    test_paths = test_abn_paths + test_nor_paths

    # ================================================================
    # GROUND TRUTH (embedded gt-xd.npy)
    # ================================================================

    print("\n" + "=" * 60)
    print("PHASE 3: GROUND TRUTH")
    print("=" * 60)

    GT_B64 = (
        "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"
    )
    gt_bytes = gzip.decompress(base64.b64decode(GT_B64))
    frame_gt = np.unpackbits(np.frombuffer(gt_bytes, dtype=np.uint8))
    frame_gt = frame_gt[:2330384]
    anom_count = int(frame_gt.sum())
    print(f"GT: {len(frame_gt):,} frames, {anom_count:,} anomalous "
          f"({anom_count/len(frame_gt)*100:.2f}%)")

    # ================================================================
    # ORIGINAL CLIP-TSA CODE — UTILITIES
    # (from utils/utils.py — verbatim)
    # ================================================================

    def process_feat(feat, length):
        if type(feat) == torch.Tensor:
            new_feat = torch.zeros((length, feat.shape[1])).type(torch.float32)
            r = torch.linspace(0, len(feat), length+1, dtype=torch.int)
            for i in range(length):
                if r[i]!=r[i+1]:
                    new_feat[i,:] = torch.mean(feat[r[i]:r[i+1],:], 0)
                else:
                    new_feat[i,:] = feat[r[i],:]
            return new_feat
        else:
            new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
            r = np.linspace(0, len(feat), length+1, dtype=int)
            for i in range(length):
                if r[i]!=r[i+1]:
                    new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
                else:
                    new_feat[i,:] = feat[r[i],:]
            return new_feat

    # ================================================================
    # ORIGINAL CLIP-TSA CODE — HARD ATTENTION
    # (from utils/hard_attention.py — verbatim)
    # ================================================================

    class PerturbedTopKFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, k: float, num_samples:int=1000, sigma: float=0.05, train_mode: bool=True):
            b, d = x.shape
            k = int(d * k)
            noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)
            perturbed_x = x[:, None, :] + noise * sigma
            if k > perturbed_x.shape[-1]:
                k = perturbed_x.shape[-1]
            elif k == 0:
                k = 1
            if not train_mode:
                k = min(1000, k)
            topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
            indices = topk_results.indices
            indices = torch.sort(indices, dim=-1).values
            perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
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
                return tuple([ None ] * 5)
            grad_expected = torch.einsum("bnkd,bne->bkde", ctx.perturbed_output, ctx.noise)
            grad_expected /= (ctx.num_samples * ctx.sigma)
            grad_input = torch.einsum("bkde,bke->bd", grad_expected, grad_output)
            return (grad_input,) + tuple([ None ] * 5)

    class PerturbedTopK(nn.Module):
        def __init__(self, k: float, num_samples: int=1000, sigma: float=0.05):
            super(PerturbedTopK, self).__init__()
            self.num_samples = num_samples
            self.sigma = sigma
            self.k = k

        def __call__(self, x, train_mode):
            return PerturbedTopKFunction.apply(x, self.k, self.num_samples, self.sigma, train_mode)

    # MLP scorer inside HardAttention (from hard_attention.py)
    class HA_MLP(nn.Module):
        def __init__(self, input_dim=512):
            super(HA_MLP, self).__init__()
            self.fc1 = nn.Linear(input_dim, 256)
            self.fc2 = nn.Linear(256, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            out = self.sigmoid(self.fc2(x))
            return out

    class HardAttention(nn.Module):
        def __init__(self, k=1.0, num_samples=100, input_dim=512):
            super(HardAttention, self).__init__()
            self.scorer = HA_MLP(input_dim)
            self.hard_att = PerturbedTopK(k=k, num_samples=num_samples)

        def forward(self, inputs):
            scores = self.scorer(inputs)
            b, t, d = scores.shape
            if b > 1:
                train_mode = True
            else:
                train_mode = False
            topk = self.hard_att(scores.squeeze(-1), train_mode)
            out = topk.unsqueeze(-1) * inputs.unsqueeze(1)
            out = torch.sum(out, dim=1)
            return out

    # ================================================================
    # ORIGINAL CLIP-TSA CODE — MODEL
    # (from model.py — verbatim, with imports inlined)
    # ================================================================

    def weight_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            torch_init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)

    # Dim-reduction MLP (from model.py, class MLP)
    class DimReductionMLP(nn.Module):
        def __init__(self, input_dim=512):
            super(DimReductionMLP, self).__init__()
            self.fc1 = nn.Linear(input_dim, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            out = self.sigmoid(self.fc2(x))
            return out

    class _NonLocalBlockND(nn.Module):
        def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
            super(_NonLocalBlockND, self).__init__()
            assert dimension in [1, 2, 3]
            self.dimension = dimension
            self.sub_sample = sub_sample
            self.in_channels = in_channels
            self.inter_channels = inter_channels
            if self.inter_channels is None:
                self.inter_channels = in_channels // 2
                if self.inter_channels == 0:
                    self.inter_channels = 1
            if dimension == 3:
                conv_nd = nn.Conv3d
                max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
                bn = nn.BatchNorm3d
            elif dimension == 2:
                conv_nd = nn.Conv2d
                max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
                bn = nn.BatchNorm2d
            else:
                conv_nd = nn.Conv1d
                max_pool_layer = nn.MaxPool1d(kernel_size=(2))
                bn = nn.BatchNorm1d
            self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
            if bn_layer:
                self.W = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                            kernel_size=1, stride=1, padding=0),
                    bn(self.in_channels)
                )
                nn.init.constant_(self.W[1].weight, 0)
                nn.init.constant_(self.W[1].bias, 0)
            else:
                self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                                 kernel_size=1, stride=1, padding=0)
                nn.init.constant_(self.W.weight, 0)
                nn.init.constant_(self.W.bias, 0)
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
            if sub_sample:
                self.g = nn.Sequential(self.g, max_pool_layer)
                self.phi = nn.Sequential(self.phi, max_pool_layer)

        def forward(self, x, return_nl_map=False):
            batch_size = x.size(0)
            g_x = self.g(x).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)
            N = f.size(-1)
            f_div_C = f / N
            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *x.size()[2:])
            W_y = self.W(y)
            z = W_y + x
            if return_nl_map:
                return z, f_div_C
            return z

    class NONLocalBlock1D(_NonLocalBlockND):
        def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
            super(NONLocalBlock1D, self).__init__(in_channels,
                                                  inter_channels=inter_channels,
                                                  dimension=1, sub_sample=sub_sample,
                                                  bn_layer=bn_layer)

    class Aggregate(nn.Module):
        def __init__(self, len_feature):
            super(Aggregate, self).__init__()
            bn = nn.BatchNorm1d
            self.len_feature = len_feature
            self.division = 2048 // len_feature
            self.conv_1 = nn.Sequential(
                nn.Conv1d(in_channels=len_feature, out_channels=512 // self.division, kernel_size=3,
                          stride=1,dilation=1, padding=1),
                nn.ReLU(),
                bn(512 // self.division)
            )
            self.conv_2 = nn.Sequential(
                nn.Conv1d(in_channels=len_feature, out_channels=512 // self.division, kernel_size=3,
                          stride=1, dilation=2, padding=2),
                nn.ReLU(),
                bn(512 // self.division)
            )
            self.conv_3 = nn.Sequential(
                nn.Conv1d(in_channels=len_feature, out_channels=512 // self.division, kernel_size=3,
                          stride=1, dilation=4, padding=4),
                nn.ReLU(),
                bn(512 // self.division)
            )
            self.conv_4 = nn.Sequential(
                nn.Conv1d(in_channels=2048 // self.division, out_channels=512 // self.division, kernel_size=1,
                          stride=1, padding=0, bias = False),
                nn.ReLU(),
            )
            self.conv_5 = nn.Sequential(
                nn.Conv1d(in_channels=2048 // self.division, out_channels=2048 // self.division, kernel_size=3,
                          stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.BatchNorm1d(2048 // self.division),
            )
            self.non_local = NONLocalBlock1D(512 // self.division, sub_sample=False, bn_layer=True)

        def forward(self, x):
                out = x.permute(0, 2, 1)
                residual = out
                out1 = self.conv_1(out)
                out2 = self.conv_2(out)
                out3 = self.conv_3(out)
                out_d = torch.cat((out1, out2, out3), dim = 1)
                out = self.conv_4(out)
                out = self.non_local(out)
                out = torch.cat((out_d, out), dim=1)
                out = self.conv_5(out)
                out = out + residual
                out = out.permute(0, 2, 1)
                return out

    class Model(nn.Module):
        def __init__(self, n_features, batch_size, k=0.0, num_samples=100, apply_HA=True, visual="vit", gpu="0"):
            super(Model, self).__init__()

            OG_feat = n_features
            if visual.upper() in ["I3D", "C3D"]:
                n_features = 512

            self.hard_attention = HardAttention(k=k, num_samples=num_samples, input_dim=n_features)
            self.apply_HA = apply_HA
            self.batch_size = batch_size
            self.num_segments = 32
            self.k_abn = self.num_segments // 10
            self.k_nor = self.num_segments // 10

            self.division = 2048 // n_features

            self.Aggregate = Aggregate(len_feature=2048 // self.division)
            self.fc1 = nn.Linear(n_features, 512 // self.division)
            self.fc2 = nn.Linear(512 // self.division, 128 // self.division)
            self.fc3 = nn.Linear(128 // self.division, 1)

            self.drop_out = nn.Dropout(0.7)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.apply(weight_init)

            self.parallel = 0.5 if "," in gpu else 1
            self.visual = visual

            self.mlp = DimReductionMLP(input_dim=OG_feat)

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
                if self.visual != "vit" and out.shape[0] > 1 and out.shape[1] != 32:
                    concat = []
                    for i in out:
                        concat.append(self.hard_attention(i.unsqueeze(0)))
                    out = torch.cat(concat, dim=0)
                else:
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

            adjusted_scoremag_batch_size = int(self.batch_size * self.parallel)
            adjusted_feat_batch_size = int(self.batch_size*ncrops * self.parallel)

            normal_features = features[0:adjusted_feat_batch_size]
            normal_scores = scores[0:adjusted_scoremag_batch_size]

            abnormal_features = features[adjusted_feat_batch_size:]
            abnormal_scores = scores[adjusted_scoremag_batch_size:]

            feat_magnitudes = torch.norm(features, p=2, dim=2)
            feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)
            nfea_magnitudes = feat_magnitudes[0:adjusted_scoremag_batch_size]
            afea_magnitudes = feat_magnitudes[adjusted_scoremag_batch_size:]
            n_size = nfea_magnitudes.shape[0]

            if nfea_magnitudes.shape[0] == 1:
                afea_magnitudes = nfea_magnitudes
                abnormal_scores = normal_scores
                abnormal_features = normal_features

            select_idx = torch.ones_like(nfea_magnitudes).cuda()
            select_idx = self.drop_out(select_idx)

            afea_magnitudes_drop = afea_magnitudes * select_idx

            idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]

            idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])
            abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
            abnormal_features = abnormal_features.permute(1, 0, 2, 3)

            total_select_abn_feature = torch.zeros(0)
            for abnormal_feature in abnormal_features:
                feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)
                total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

            idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
            score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)

            select_idx_normal = torch.ones_like(nfea_magnitudes).cuda()
            select_idx_normal = self.drop_out(select_idx_normal)
            nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal

            idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]

            idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

            normal_features = normal_features.view(n_size, ncrops, t, f)
            normal_features = normal_features.permute(1, 0, 2, 3)

            total_select_nor_feature = torch.zeros(0)
            for nor_fea in normal_features:
                feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)
                total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

            idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
            score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1)

            feat_select_abn = total_select_abn_feature
            feat_select_normal = total_select_nor_feature

            return score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_select_abn, feat_select_abn, scores, feat_select_abn, feat_select_abn, feat_magnitudes

    # ================================================================
    # ORIGINAL CLIP-TSA CODE — TRAIN
    # (from train.py — verbatim)
    # ================================================================

    def sparsity(arr, batch_size, lamda2):
        loss = torch.mean(torch.norm(arr, dim=0))
        return lamda2*loss

    def smooth(arr, lamda1):
        arr2 = torch.zeros_like(arr)
        arr2[:-1] = arr[1:]
        arr2[-1] = arr[-1]
        loss = torch.sum((arr2-arr)**2)
        return lamda1*loss

    class SigmoidMAELoss(torch.nn.Module):
        def __init__(self):
            super(SigmoidMAELoss, self).__init__()
            self.__l1_loss__ = MSELoss()
        def forward(self, pred, target):
            return self.__l1_loss__(pred, target)

    class RTFM_loss(torch.nn.Module):
        def __init__(self, alpha, margin):
            super(RTFM_loss, self).__init__()
            self.alpha = alpha
            self.margin = margin
            self.sigmoid = torch.nn.Sigmoid()
            self.mae_criterion = SigmoidMAELoss()
            self.criterion = torch.nn.BCELoss()

        def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
            label = torch.cat((nlabel, alabel), 0)
            score_abnormal = score_abnormal
            score_normal = score_normal
            score = torch.cat((score_normal, score_abnormal), 0)
            score = score.squeeze()
            label = label.cuda()
            loss_cls = self.criterion(score, label)
            loss_abn = torch.abs(self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1))
            loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)
            loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)
            loss_total = loss_cls + self.alpha * loss_rtfm
            return loss_total

    # ================================================================
    # ORIGINAL CLIP-TSA CODE — DATASET
    # (from dataset.py — adapted for Kaggle paths only)
    # ================================================================

    import torch.utils.data as data

    class CLIPTSADataset(data.Dataset):
        def __init__(self, file_paths, is_normal=True, test_mode=False, visual="vit"):
            self.is_normal = is_normal
            self.test_mode = test_mode
            self.visual = visual
            self.list = file_paths

        def __getitem__(self, index):
            path = self.list[index]
            label = self.get_label()
            features = np.load(path, allow_pickle=True)
            features = np.array(features, dtype=np.float32)

            # Original dataset.py logic for ViT:
            # "If it enters, it doesn't do tencrop"
            if self.visual.lower() not in ["i3d", "c3d"] and len(features.shape) != 3:
                features = features.reshape((*features.shape[:-1], 1, features.shape[-1]))

            if self.test_mode:
                return features
            else:
                # process 10-cropped snippet feature
                features = features.transpose(1, 0, 2)  # [10, B, T, F]
                divided_features = []
                for feature in features:
                    feature = process_feat(feature, 32)
                    divided_features.append(feature)
                divided_features = np.array(divided_features, dtype=np.float32)
                return divided_features, label

        def get_label(self):
            if self.is_normal:
                label = torch.tensor(0.0)
            else:
                label = torch.tensor(1.0)
            return label

        def __len__(self):
            return len(self.list)

    # ================================================================
    # ORIGINAL CLIP-TSA CODE — TEST
    # (from test_10crop.py — verbatim, adapted for embedded GT)
    # ================================================================

    def test(dataloader, model, device):
        with torch.no_grad():
            model.eval()
            pred = torch.zeros(0)

            for i, input in enumerate(dataloader):
                input = input.to(device)
                input = input.permute(0, 2, 1, 3)

                score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
                scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
                logits = torch.squeeze(logits, 1)
                logits = torch.mean(logits, 0)
                sig = logits
                pred = torch.cat((pred, sig))

            pred = list(pred.cpu().detach().numpy())
            pred = np.repeat(np.array(pred), 16)

            gt = frame_gt

            fpr, tpr, threshold = roc_curve(list(gt), pred)
            rec_auc = auc(fpr, tpr)

            precision, recall, th = precision_recall_curve(list(gt), pred)
            pr_auc = auc(recall, precision)

            print(f'  AP: {pr_auc*100:.2f}%  AUC: {rec_auc*100:.2f}%')
            return pr_auc

    # ================================================================
    # ORIGINAL CLIP-TSA TRAINING LOOP
    # (from main.py — verbatim logic)
    # ================================================================

    print("\n" + "=" * 60)
    print("TRAINING: CLIP-TSA Original (4000 iter)")
    print("=" * 60)

    # Hyperparams from runner.sh (xd config)
    BATCH_SIZE = 16
    MAX_EPOCH = 4000
    K = 0.9
    SEED = 0
    LR = 0.001
    WD = 0.005
    NUM_SAMPLES = 100
    VISUAL = "vit"
    GPU = "0"

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Create datasets exactly as original main.py does
    train_nloader = DataLoader(
        CLIPTSADataset(train_nor_paths, is_normal=True, test_mode=False, visual=VISUAL),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(
        CLIPTSADataset(train_abn_paths, is_normal=False, test_mode=False, visual=VISUAL),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(
        CLIPTSADataset(test_paths, test_mode=True, visual=VISUAL),
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False)

    print(f"Train normal: {len(train_nloader.dataset)} videos, {len(train_nloader)} batches")
    print(f"Train abnormal: {len(train_aloader.dataset)} videos, {len(train_aloader)} batches")
    print(f"Test: {len(test_loader.dataset)} videos")

    feature_size = train_aloader.dataset[0][0].shape[-1]
    print(f"Feature size from dataset: {feature_size}")

    model = Model(feature_size, BATCH_SIZE, K, NUM_SAMPLES, True, visual=VISUAL, gpu=GPU)
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    test_info = { "epoch": [], "test_AUC": [], "best_test_AUC": [] }
    best_AUC = -1

    # Initial test
    print("Before training:")
    auc_val = test(test_loader, model, DEVICE)

    SAVE = "/kaggle/working"

    t_start = time.time()

    for step in range(1, MAX_EPOCH + 1):
        # Reset iterators as original code does
        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)
        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        # === Original train() function logic (from train.py) ===
        model.train()
        ninput, nlabel = next(loadern_iter)
        ainput, alabel = next(loadera_iter)

        input = torch.cat((ninput, ainput), 0).to(DEVICE)

        score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
        feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, _ = model(input)

        scores = scores.view(BATCH_SIZE * 32 * 2, -1)
        scores = scores.squeeze()
        abn_scores = scores[BATCH_SIZE * 32:]

        nlabel = nlabel[0:BATCH_SIZE]
        alabel = alabel[0:BATCH_SIZE]

        loss_criterion = RTFM_loss(0.0001, 100)
        loss_sparse = sparsity(abn_scores, BATCH_SIZE, 8e-3)
        loss_smooth = smooth(abn_scores, 8e-4)
        cost = loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn) + loss_smooth + loss_sparse

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # === Logging (every 200 steps) ===
        if step % 200 == 0 or step == 1:
            elapsed = time.time() - t_start
            print(f"[Step {step:4d}/{MAX_EPOCH}] loss {cost.item():.4f} | {elapsed:.0f}s")

        # === Original eval schedule: every 5 steps after 150 ===
        # (770 evals total - faithful to original code)
        if step % 5 == 0 and step > 150:
            auc_val = test(test_loader, model, DEVICE)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc_val)

            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, os.path.join(SAVE, f"cliptsa_original_best.pkl"),
                )
                print(f"  *** BEST AP: {best_AUC*100:.2f}% @ step {step}")

            test_info["best_test_AUC"].append(best_AUC)

    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE: Best AP = {best_AUC*100:.2f}%")
    print(f"Total training time: {elapsed_total:.0f}s ({elapsed_total/60:.1f}m)")
    print(f"{'='*60}")

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, os.path.join(SAVE, "cliptsa_original_final.pkl"))

    # Final summary
    summary = {
        "model": "CLIP-TSA (original code)",
        "dataset": "XD-Violence",
        "features": "CLIP ViT-B/16",
        "best_ap": float(best_AUC * 100),
        "total_params": int(total_params),
        "training_time_s": float(elapsed_total),
        "max_epoch": MAX_EPOCH,
        "k": K,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
    }
    with open(os.path.join(SAVE, "cliptsa_original_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("\nDone!")

try:
    main()
except BaseException as e:
    tb = traceback.format_exc()
    msg = f"FATAL: {type(e).__name__}: {e}\n{tb}"
    print(msg, flush=True)
    try:
        with open("/kaggle/working/crash_log.txt", "w") as _f:
            _f.write(msg)
    except Exception:
        pass
    if not isinstance(e, (KeyboardInterrupt, SystemExit)):
        sys.exit(1)
    raise
