import torch.nn as nn
import torch.nn.functional as F

class MILRankingLoss(nn.Module):
    """
    Max-margin ranking loss for MIL-based VAD.
    """
    def __init__(self, margin=1.0, sparsity_weight=8e-3, smooth_weight=8e-4):
        super().__init__()
        self.margin = margin
        self.sparsity_weight = sparsity_weight
        self.smooth_weight = smooth_weight
    
    def forward(self, anomaly_scores, normal_scores):
        max_anomaly = anomaly_scores.max(dim=1)[0]
        max_normal = normal_scores.max(dim=1)[0]
        
        loss_rank = F.relu(self.margin - max_anomaly.unsqueeze(1) + max_normal.unsqueeze(0))
        loss_rank = loss_rank.mean()
        
        loss_sparse = anomaly_scores.mean()
        loss_smooth = (anomaly_scores[:, 1:] - anomaly_scores[:, :-1]).pow(2).mean()
        
        total = loss_rank + self.sparsity_weight * loss_sparse + self.smooth_weight * loss_smooth
        return total, loss_rank.item(), loss_sparse.item(), loss_smooth.item()
