import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Learnable positional encoding for temporal segments."""
    def __init__(self, d_model, max_len=32):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x):
        return x + self.pos_embed[:, :x.size(1), :]



class TransMILClassifier(nn.Module):
    """
    Temporal Transformer for MIL-based Video Anomaly Detection.
    
    NOVEL CONTRIBUTION: Unlike Sultani-MIL (independent per-segment MLP) and
    RTFM (magnitude-weighted MLP), TransMIL uses multi-head self-attention to
    model temporal dependencies between the 32 video segments.
    
    Key insight: Anomalies are contextual — a "running" segment is normal in
    a park but anomalous in a bank lobby. Self-attention captures these
    inter-segment relationships.
    
    Architecture:
      Input (32, 1024) → Linear projection → Positional encoding →
      Transformer encoder (N layers) → Segment-wise classifier → Scores (32,)
    """
    def __init__(self, feat_dim=1024, hidden_dim=256, dropout=0.5,
                 n_heads=4, n_layers=1):
        super().__init__()
        
        # Project I3D features to transformer dimension
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        
        # Positional encoding for temporal order
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=32)
        
        # Temporal transformer encoder — lightweight (1 layer, 4 heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Anomaly score head — simple to prevent overfitting
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # x: (B, 32, 1024)
        B, T, D = x.shape
        
        h = self.input_proj(x)          # (B, 32, hidden_dim)
        h = self.pos_enc(h)             # + positional encoding
        h = self.transformer(h)         # (B, 32, hidden_dim) — with temporal attention
        h = self.norm(h)                # post-norm for stability
        
        scores = self.score_head(h.reshape(B * T, -1))  # (B*32, 1)
        scores = scores.reshape(B, T)                     # (B, 32)
        return scores


# ── 17. Dual-Stream Dataset Loaders ───────────────────────────────────────