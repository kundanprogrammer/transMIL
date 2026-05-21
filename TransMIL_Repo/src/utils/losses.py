import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MILRankingLoss(nn.Module):
    """
    Max-margin ranking loss for MIL-based VAD.
    For each (anomaly_bag, normal_bag) pair:
      max(0, 1 - max(anomaly_scores) + max(normal_scores))
    Plus sparsity and smoothness regularization.
    """
    def __init__(self, margin=1.0, sparsity_weight=8e-3, smooth_weight=8e-4):
        super().__init__()
        self.margin = margin
        self.sparsity_weight = sparsity_weight
        self.smooth_weight = smooth_weight
    
    def forward(self, anomaly_scores, normal_scores):
        """
        anomaly_scores: (B_a, 32) — scores for anomaly videos
        normal_scores:  (B_n, 32) — scores for normal videos
        """
        # Top-k selection (k=1 for simplicity, k=3 for RTFM-style)
        max_anomaly = anomaly_scores.max(dim=1)[0]  # (B_a,)
        max_normal = normal_scores.max(dim=1)[0]     # (B_n,)
        
        # Pairwise ranking loss
        # Broadcast: (B_a, 1) - (1, B_n)
        loss_rank = F.relu(self.margin - max_anomaly.unsqueeze(1) + max_normal.unsqueeze(0))
        loss_rank = loss_rank.mean()
        
        # Sparsity: anomaly scores should be sparse (few segments anomalous)
        loss_sparse = anomaly_scores.mean()
        
        # Smoothness: adjacent segments should have similar scores
        loss_smooth = (anomaly_scores[:, 1:] - anomaly_scores[:, :-1]).pow(2).mean()
        
        total = loss_rank + self.sparsity_weight * loss_sparse + self.smooth_weight * loss_smooth
        return total, loss_rank.item(), loss_sparse.item(), loss_smooth.item()


# ── 6. Model: MIL Anomaly Classifier ──────────────────────────────────────