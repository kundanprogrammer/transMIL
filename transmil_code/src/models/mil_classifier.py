import torch.nn as nn
import torch

class MILClassifier(nn.Module):
    """
    Simple MLP classifier for segment-level anomaly scoring.
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
        B, T, D = x.shape
        scores = self.classifier(x.reshape(B * T, D))
        scores = scores.reshape(B, T)
        return scores
