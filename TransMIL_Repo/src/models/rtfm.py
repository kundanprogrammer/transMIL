import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RTFMClassifier(nn.Module):
    """
    RTFM-style classifier with feature magnitude learning.
    Uses shared embedding → two branches (magnitude + classifier).
    Magnitude modulates attention over segments rather than scaling scores.
    """
    def __init__(self, feat_dim=1024, hidden_dim=512, dropout=0.6, topk=3):
        super().__init__()
        self.topk = topk
        
        # Shared feature embedding
        self.embed = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Magnitude branch — learns temporal attention weights
        self.mag_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        
        # Score branch — anomaly classifier
        self.score_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x):
        # x: (B, 32, 1024)
        B, T, D = x.shape
        
        h = self.embed(x.reshape(B * T, D))       # (B*T, hidden)
        
        # Magnitude as attention: sigmoid-bounded [0, 1]
        mag = torch.sigmoid(self.mag_branch(h))    # (B*T, 1)
        mag = mag.reshape(B, T)                     # (B, T)
        
        # Raw logits — no sigmoid here
        logits = self.score_branch(h)               # (B*T, 1)
        logits = logits.reshape(B, T)               # (B, T)
        
        # Magnitude-weighted sigmoid: mag amplifies important segments
        # Scale logits by magnitude before sigmoid for gradient-friendly gating
        scores = torch.sigmoid(logits * (1.0 + mag))
        
        return scores


# ── 8. MIL DataLoader (pairs anomaly + normal) ────────────────────────────