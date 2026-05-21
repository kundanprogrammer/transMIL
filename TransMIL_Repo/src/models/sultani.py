import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MILClassifier(nn.Module):
    """
    Simple MLP classifier for segment-level anomaly scoring.
    Input: (B, 32, 1024) I3D features
    Output: (B, 32) anomaly scores per segment
    """
    def __init__(self, feat_dim=1024, hidden_dim=512, dropout=0.6):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # x: (B, 32, 1024)
        B, T, D = x.shape
        scores = self.classifier(x.reshape(B * T, D))  # (B*32, 1)
        scores = scores.reshape(B, T)                    # (B, 32)
        return scores


# ── 7. RTFM Model (Tian et al., ICCV 2021) ────────────────────────────────