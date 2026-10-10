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
        max_anomaly = anomaly_scores.max(dim=1)[0]
        max_normal = normal_scores.max(dim=1)[0]

        loss_rank = F.relu(self.margin - max_anomaly.unsqueeze(1) + max_normal.unsqueeze(0))
        loss_rank = loss_rank.mean()

        loss_sparse = anomaly_scores.mean()
        loss_smooth = (anomaly_scores[:, 1:] - anomaly_scores[:, :-1]).pow(2).mean()

        total = loss_rank + self.sparsity_weight * loss_sparse + self.smooth_weight * loss_smooth
        return total, loss_rank.item(), loss_sparse.item(), loss_smooth.item()


class EfficientVADLoss(nn.Module):
    """
    Combined loss for EfficientVAD:
      L = L_rank + lambda_mag * L_mag + lambda_smooth * L_smooth + lambda_sparse * L_sparse

    L_rank:   top-k MIL ranking with stochastic selection via dropout mask
    L_mag:    feature magnitude margin loss (adapted from RTFM for post-MSTC features)
    L_smooth: temporal smoothness on scores
    L_sparse: sparsity on anomaly scores
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
        """Stochastic top-k selection via dropout mask on magnitudes (RTFM-style)."""
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
        """
        anom_scores/norm_scores: (B, T)
        anom_mags/norm_mags:     (B, T) — L2 norms of temporal features
        """
        k = min(self.top_k, anom_scores.shape[1])

        anom_topk_scores, anom_topk_mags = self._topk_with_dropout(anom_scores, anom_mags, k)
        norm_topk_scores, norm_topk_mags = self._topk_with_dropout(norm_scores, norm_mags, k)

        # Ranking loss on top-k means
        anom_max = anom_topk_scores.mean(dim=1)
        norm_max = norm_topk_scores.mean(dim=1)
        loss_rank = F.relu(self.margin - anom_max.unsqueeze(1) + norm_max.unsqueeze(0)).mean()

        # Feature magnitude loss: push abn magnitudes toward margin, suppress normal
        loss_mag = ((self.mag_margin - anom_topk_mags).abs() + norm_topk_mags).pow(2).mean()

        # Temporal smoothness
        loss_smooth = (anom_scores[:, 1:] - anom_scores[:, :-1]).pow(2).mean()

        # Sparsity
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
