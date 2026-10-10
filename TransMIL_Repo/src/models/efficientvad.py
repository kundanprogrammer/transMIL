import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # x: (B, T, D)
        residual = x
        h = self.proj_down(x)           # (B, T, bottleneck)
        h = h.transpose(1, 2)           # (B, bottleneck, T)

        h3 = F.gelu(self.dw_conv3(h))
        h5 = F.gelu(self.dw_conv5(h))
        h7 = F.gelu(self.dw_conv7(h))
        h = (h3 + h5 + h7) / 3.0       # element-wise mean

        h = h.transpose(1, 2)           # (B, T, bottleneck)
        h = self.proj_up(h)             # (B, T, D)
        return self.norm(h + residual)


class SqueezeExcitation(nn.Module):
    """Channel attention via global average pooling + FC bottleneck."""
    def __init__(self, dim=512, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.fc2 = nn.Linear(dim // reduction, dim)

    def forward(self, x):
        # x: (B, T, D)
        s = x.mean(dim=1)              # (B, D) global avg pool over time
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))  # (B, D)
        return x * s.unsqueeze(1)       # broadcast over T


class EfficientVAD(nn.Module):
    """
    Lightweight CLIP-adapted video anomaly detector.
    Operates on frozen CLIP ViT-B/16 features (512-D, pre-extracted).

    Architecture:
      MSTC (multi-scale temporal conv) → SE attention → FC scoring head
    Plus a parallel feature magnitude branch used in the loss.

    Input:  (B, T, 512)  — pre-extracted CLIP features
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
        # x: (B, T, D) — CLIP features
        B, T, D = x.shape
        h = self.mstc(x)                                # (B, T, D)
        h = self.se(h)                                   # (B, T, D)
        magnitudes = torch.norm(h, p=2, dim=-1)          # (B, T)
        scores = self.score_head(h.reshape(B * T, D))    # (B*T, 1)
        scores = scores.reshape(B, T)                    # (B, T)
        return scores, magnitudes
