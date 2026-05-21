import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PEMILClassifier(nn.Module):
    """
    PE-MIL: Prompt-Enhanced MIL (Chen et al., CVPR 2024).
    I3D-compatible approximation using learnable semantic prototypes.

    Architecture:
      1. Visual projection:  I3D features → hidden_dim (with LayerNorm + GELU)
      2. Prototype bank:     n_prototypes learnable vectors in hidden_dim space
                             (simulate CLIP text embeddings of crime categories)
      3. Cross-attention:    segments (query) attend to prototypes (key, value)
                             → prototype-guided segment representations
      4. Gating:             learned gate controls how much prototype info to add
      5. Score head:         concatenate [original, guided] → anomaly score

    Input:  (B, 32, feat_dim)  — I3D features
    Output: (B, 32)             — segment anomaly scores in [0,1]
    """
    def __init__(self, feat_dim=1024, hidden_dim=512, dropout=0.6,
                 n_prototypes=13, n_heads=4):
        super().__init__()
        self.n_prototypes = n_prototypes

        # ── 1. Visual projection
        self.visual_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # ── 2. Learnable semantic prototypes
        #    Initialised with small random values; updated end-to-end
        #    Each prototype learns to "describe" one anomaly pattern
        self.prototypes = nn.Parameter(
            torch.randn(n_prototypes, hidden_dim) * (1.0 / (hidden_dim ** 0.5))
        )

        # ── 3. Cross-attention: segments attend to prototypes
        #    Query = projected video segments
        #    Key = Value = prototype bank
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout * 0.3,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)

        # ── 4. Learned gate: controls how much prototype guidance to inject
        #    If gate=0: pure visual features; if gate=1: full prototype guidance
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # ── 5. Score head: takes gated fusion → anomaly score
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, feat_dim) — batch of I3D feature tensors

        Returns:
            scores: (B, T) — segment-level anomaly scores
        """
        B, T, D = x.shape

        # 1. Project visual features
        v = self.visual_proj(x)                                 # (B, T, hidden)

        # 2. Expand prototype bank for batch
        proto = self.prototypes.unsqueeze(0).expand(B, -1, -1) # (B, n_proto, hidden)

        # 3. Cross-attention: segments query prototypes
        v_guided, _ = self.cross_attn(v, proto, proto)         # (B, T, hidden)
        v_guided = self.cross_attn_norm(v + v_guided)          # residual + norm

        # 4. Gated fusion: learn how much to trust prototype guidance per segment
        gate_input = torch.cat([v, v_guided], dim=-1)          # (B, T, hidden*2)
        gate = self.gate(gate_input)                            # (B, T, hidden) ∈ [0,1]
        v_fused = gate * v_guided + (1 - gate) * v             # (B, T, hidden)

        # 5. Score
        scores = self.score_head(
            v_fused.reshape(B * T, -1)
        ).reshape(B, T)                                         # (B, T)

        return scores


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART 3C: RUN ALL EXPERIMENTS                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
