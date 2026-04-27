import torch
import torch.nn as nn

class RTFMClassifier(nn.Module):
    """
    RTFM-style classifier with feature magnitude learning.
    """
    def __init__(self, feat_dim=1024, hidden_dim=512, dropout=0.6, topk=3):
        super().__init__()
        self.topk = topk
        
        self.embed = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.mag_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        
        self.score_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x):
        B, T, D = x.shape
        h = self.embed(x.reshape(B * T, D))
        
        mag = torch.sigmoid(self.mag_branch(h))
        mag = mag.reshape(B, T)
        
        logits = self.score_branch(h)
        logits = logits.reshape(B, T)
        
        scores = torch.sigmoid(logits * (1.0 + mag))
        return scores
