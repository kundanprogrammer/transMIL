"""
CLIP-TSA Original Code — XD-Violence (Kaggle T4)
=================================================
This runs the EXACT original CLIP-TSA code from
https://github.com/joos2010kj/CLIP-TSA
with only path adaptations for Kaggle environment.
No reimplementation — code is copy-pasted from the repo.
"""
import os, sys, traceback

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)
print("BOOT: CLIP-TSA original code started")

def main():
    import glob, time, json, gzip, base64, warnings, random
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.nn.init as torch_init
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.nn import MSELoss
    from sklearn.metrics import auc, roc_curve, precision_recall_curve

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
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # ================================================================
    # LOCATE FEATURES (same as our v15/v16 logic)
    # ================================================================

    print("\n" + "=" * 60)
    print("PHASE 1: LOCATING FEATURES")
    print("=" * 60)

    INPUT_BASE = "/kaggle/input"
    FEAT_ROOT = None

    if os.path.isdir(INPUT_BASE):
        def _scan(d, depth=0):
            global FEAT_ROOT
            items = sorted(os.listdir(d))
            if depth < 4:
                print("  " + "  " * depth + f"[{os.path.basename(d)}/] ({len(items)} items)")
            for item in items:
                full = os.path.join(d, item)
                if os.path.isdir(full):
                    if item == "features":
                        return full
                    r = _scan(full, depth + 1)
                    if r:
                        return r
            return None
        FEAT_ROOT = _scan(INPUT_BASE)

    if FEAT_ROOT is None:
        raise RuntimeError("Cannot find features directory")

    print(f"Features root: {FEAT_ROOT}")

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
        "H4sIAAAAAAAAA+3dB3xN5x8G8JOb5CYRkUSIFbH3LKW2Wqu0tHZttVRRbdHSalFaW7X2"
        "3iq1V6ldu5TYRIns5N7/e96Te+N/+y8S9Pu+n0/OPee8OUbufe7vnPecGAwGAwAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB4Ihn+8x9PnPR/d31/dw"
        "EAAAAAAADwBChevHjQXxcuXIiq/kSf4ubvw1C+Z4l+7u4CAAAAAAAAYH9lO3bsaFx34MC7"
        "7u4IAAAAAAAA4B5FihRpVapkyTpOuxoYDIY07u4SAAAAAAAAYD9FS5UqVf7Zxo0X6nJDvZ"
        "o16/Z0d4cAAAAAAAAA+ylevHjQ6T17FuiS4sU/7ebuDgEAAAAAAAD2FRQUFHH5woV5v/bo"
        "0adFq1bjf1y6dNIFdY/bJv6bxb8MAAAAAACwm4CAAENcXFy8u/sBAADwUMjg7g4AAADci"
        "wQAACRFAgAAkiQBAECSJAAAAEmSAAAAJEkCAACQJAkAAABJkgAAAEiSJAAAAEmSAAAAJEkC"
        "AACQJAkAAECSJAAAAEmSAAAAJEkCAACQJAkAAABJkgAAAEiSJAAAAEmSAAAAJEkCAACAJEk"
        "AAACQJAkAAABJkgAAAEiSJAAAAEmSAAAAJEkCAACQJAkAAABJkgAAACBJkgAAAEiSJAAAA"
        "EmSAAAAJEkCAACQJAkAAABJkgAAAEiSJAAAAEmSAAAAIEmSAAAAJEkCAACAJEkAAACQJAkA"
        "AABJkgAAACBJEgAAAJIkAQAASJIkAAAASZIEAAAgSRIAAAAkSQIAAIAkSQAAAJIkAQAAQJI"
        "kAAAASJIEAAAgeRIAAABIkgQAAABJkgAAACBJEgAAACRJAgAAgCRJAAAAkCQJAAAAkiQJAA"
        "AAkiQBAABAkiQAAABIkgQAAABJkgAAACRJAgAAgCRJAAAAkCQJAAAAkiQBAABIkiQAAABIk"
        "gQAAABJkgAAACRJAgAAgCRJAAAAkCQJAAAAkiQBAAAAAJAkCQAAAJIkAQAAQJIkAAAASJIE"
        "AAAASZIEAAAgSRIAAAAkSQIAAIAkSQAAAJAkCQAAAJIkAQAASJIEAAAgSRIAAAAkSQIAAIA"
        "kSQAAABIAAABJkgQAAABJkgAAACRJAgAAgCRJAAAAkCQJAAAAkiQBAAAAAJAkCQAAAJIkAQ"
        "AASJIEAAAgSRIAAAAkSQIAAJAkSQAAAJIkAQAAQJIkAAAASJIEAAAgSRIAAIAkSQIAAIAkS"
        "QAAAJIkAQAASJIEAAAgSRIAAAAkSQIAAJAkSQAAAJAkCQAAAJIkAQAASJIEAAAgSRIAAIAk"
        "SQIAAIAkSQAAAJIkAQAASJIEAAAgSRIAAAAkSQIAAJAkSQAAAABIkgQAAABJkgAAACRJAgA"
        "AgCRJAAAAkCQJAAAAkiQBAABIkiQAAABIkgQAAABJkgAAACRJAgAAgCRJAAAAkCQJAAAAki"
        "QBAAAAAJAkCQAAAJIkAQAASJIEAAAgSRIAAAAkSQIAAIAkSQAAABIAAIAkSQIAAIAkSQAAA"
        "JIkAQAASJIEAAAgSRIAAAAkSQIAAJAkSQAAAJIkAQAAQJIkAAAASJIEAAAgSRIAAAAkSQIA"
        "AJAkSQAAAJIkAQAASJIEAAAgSRIAAIAkSQIAAIAkSQAAAJIkAQAASJIEAAAgSRIAAAAkSQI"
        "AAJAkSQAAAJAkCQAAAJIkAQAASJIEAAAgSRIAAIAkSQIAAJAkSQAAAJIkAQAAQJIkAAAASJ"
        "IEAAAgSRIAAAAkSQIAAJAkSQAAAABIkgQAAABJkgAAACRJAgAAgCRJAAAAkCQJAAAAkiQBA"
        "ABIkiQAAABIkgQAAABJkgAAACRJAgAAgCRJAAAAkCQJAAAAkiQBAABIAAAAAJAkCQAAAJIk"
        "AQAASJIEAAAgSRIAAAAkSQIAAIAkSQAAAJIkAQAAQJIkAAAASJIEAAAgSRIAAAAkSQIAAJA"
        "kSQAAAJIkAQAASJIEAAAgSRIAAIAkSQIAAIAkSQAAAJIkAQAASJIEAAAgSRIAAAAkSQIAAJ"
        "AkSQAAAJAkCQAAAJIkAQAASJIEAAAgSRIAAIAkSQIAAJAkSQAAAJIkAQAAQJIkAAAASJIEA"
        "AAgSRIAAAAkSQIAAJAkSQAAAABIkgQAACB5EgAAAACSJAEAAECSJAAAAEmSAAAAJEkCAACA"
        "JEkAAACQJAkAAACSJAEAAECSJAAAAEmSAAAAJEkCAACQJAkAAECSJAAAAEmSAAAAJEkCAAC"
        "AJEkAAACQJAkAAACSJAEAAECSJAAAAEmSAAAAJEkCAACQJAkAAECSJAEAAECSJAAAAEmSAA"
        "AAJEkCAACQJAkAAECSJAAAAEmSAAAAJEkCAACQJAkAAABIkgQAAABJkgAAACBJEgAAAJIkA"
        "QAAQJIkAAAASJIEAAAgSRIAAAAkSQIAAIAkSQAAAJAkCQAAAJIkAQAAQJIkAAAASJIEAAAg"
        "SRIAAAAkSQIAAIAkSQAAAJIkAQAAQJIkAAAASJIEAAAgSRIAAAAkSQIAAIAkSQAAABIAAAA"
        "AkCQJAAAAkiQBAABAkiQAAABIkgQAAABJkgAAAEiSJAAAAEmSAAAAJEkCAACQJAkAAABIkg"
        "QAAABJkgAAAEiSJAAAAEmSAAAAJEkCAACQJAkAAABIkgQAAABJkgAAACBJEgAAACRJAgAAg"
        "CRJAAAAkCQJAAAAkiQBAAAAAJAkCQAAAJIkAQAAQJIkAAAASJIEAAAgSRIAAAAkSQIAAIAk"
        "SQAAABIAAIAkSQIAAIAkSQAAAJIkAQAASJIEAAAgSRIAAAAkSQIAAJAkSQAAAJIkAQAAQJI"
        "kAAAASJIEAAAgSRIAAAAkSQIAAJAkSQAAAJIkAQAAQJIkAAAASJIEAAAgSRIAAIAkSQIAAJ"
        "AkSQAAAABJkgAAACRJAgAAgCRJAAAAkCQJAAAAkiQBAAAAAJAkCQAAAJIkAQAASJIEAAAg"
        "SRIAAAAkSQIAAIAkSQAAABIAAIAkSQIAAIAkSQAAAJIkAQAASJIEAAAgSRIAAAAkSQIAAJA"
        "kSQAAAJIkAQAAQJIkAAAASJIEAAAgSRIAAAAkSQIAAJAkSQAAAJIkAQAAQJIkAAAASJIEAA"
        "AgSRIAAIAkSQIAAJAkSQAAAJAkCQAAAJIkAQAASJIEAAAgSRIAAIAkSQIAAJAkSQAAAJIkA"
        "QAAQJIkAAAASJIEAAAgSRIAAAAkSQIAAJAkSQAAAABIkgQAAABJkgAAACRJAgAAgCRJAAAA"
        "kCQJAAAAkiQBAABIkiQAAABIkgQAAABJkgAAACRJAgAAgCRJAAAAkCQJAAAAkiQBAABIAAA"
        "AAJAkCQAAAJIkAQAASJIEAAAgSRIAAAAkSQIAAIAkSQAAAJIkAQAAQJIkAAAASJIEAAAgSR"
        "IAAAAkSQIAAJAkSQAAAJIkAQAASJIEAAAgSRIAAIAkSQIAAIAkSQAAAJIkAQAASJIEAAAg"
        "SRIAAAAkSQIAAJAkSQAAAJAkCQAAAJIkAQAASJIEAAAgSRIAAIAkSQIAAJAkSQAAAJIkAQ"
        "AAQJIkAAAASJIEAAAgSRIAAAAkSQIAAJAkSQAAAABIkgQAAABJkgAAACRJAgAAgCRJAAAAk"
        "CQJAAAAkiQBAABIkiQAAABIkgQAAABJkgAAACRJAgAAgCRJAAAAkCQJAAAAkiQBAABIAAAA"
        "AJAkCQAAAJIkAQAASJIEAAAgSRIAAAAkSQIAAIAkSQAAAJIkAQAAQJIkAAAASJIEAAAgSRI"
        "AAAAkSQIAAJAkSQAAAJIkAQAASJIEAAAgSRIAAIAkSQIAAIAkSQAAAJIkAQAASJIEAAAgSR"
        "IAAAAkSQIAAJAkSQAAAJAkCQAAAJIkAQAASJIEAAAgSRIAAIAkSQIAAJAkSQAAAJIkAQAAQ"
        "JIkAAAASJIEAAAgSRIAAAAkSQIAAJAkSQAAAABIkgQAAABJkgAAACRJAgAAgCRJAAAAkCQJ"
        "AAAAkiQBAABIkiQAAABIkgQAAABJkgAAACRJAgAAgCRJAAAAkCQJAAAAkiQBAABIAAAAAJA"
        "kCQAAAJIkAQAASJIEAAAgSRIAAAAkSQIAAIAkSQAAAJIkAQAAQJIkAAAASJIEAAAgSRIAAA"
        "AkSQIAAJAkSQAAAJIkAQAASJIEAAAgSRIAAIAkSQIAAIAkSQAAAJIkAQAASJIEAAAgSRIAA"
        "AAkSQIAAJAkSQAAAJAkCQAAAJIkAQAASJIEAAAgSRIAAIAkSQIAAJAkSQAAAJIkAQAAQJIk"
        "AAAASJIEAAAgSRIAAAAkSQIAAJAkSQAAAGBfxYsXD3J3HwAAAB4KBnf/AgAAAPcaAQAASJI"
        "EAAAgSRIAAAAkSQIAAIAkSQAAAJIkAQAASJIEAAAgeRIAAOD/V4kSJUq5uw8AAACPDIPBYD"
        "AkJCQkuLsfAAAAjwwAAAAPPX4GAQAAAAAAAAAAAAAAAAAAAAAAAACAe/wH+cPLMQBYAAA="
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
except Exception as e:
    traceback.print_exc()
    print(f"\nFATAL: {e}")
    sys.exit(1)
