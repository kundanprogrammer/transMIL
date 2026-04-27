import torch
import torch.nn as nn

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
        
        # Anomaly score head — MLP to prevent overfitting
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        B, T, D = x.shape
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.norm(x)
        # Apply score head to each segment
        scores = self.score_head(x.reshape(B * T, -1))
        scores = scores.reshape(B, T)
        return scores
