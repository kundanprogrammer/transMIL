import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class UMILClassifier(nn.Module):
    """
    UMIL: Unbiased Multiple Instance Learning (Lv et al., CVPR 2023).

    Architecture vs Sultani-MIL:
      - Same base encoder (Linear → ReLU → Dropout → Linear → ReLU)
      - Same score head (Linear → Sigmoid)
      - ADDED: projection head for contrastive learning (128-dim L2-normalised)

    The projection head is only used during training (for contrastive loss).
    At inference, only scores are used — identical interface to MILClassifier.

    Input:  (B, 32, 1024)  — batch of I3D features
    Output: (B, 32)        — segment anomaly scores in [0,1]
    """
    def __init__(self, feat_dim=1024, hidden_dim=512, dropout=0.6, proj_dim=128):
        super().__init__()
        self.proj_dim = proj_dim

        # ── Shared feature encoder (same as Sultani-MIL for fair comparison)
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # ── Score head: maps encoded features → anomaly score per segment
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # ── Projection head: maps encoded features → contrastive embedding
        #    L2-normalised 128-dim vectors, NOT used at inference
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),   # two-layer projection (SimCLR-style)
        )

    def forward(self, x, return_proj=False):
        """
        Args:
            x: (B, T, D) — batch of I3D feature tensors
            return_proj: if True, also return L2-normalised projections for UMIL loss

        Returns:
            scores: (B, T) — segment-level anomaly scores
            proj:   (B, T, proj_dim) — optional, only if return_proj=True
        """
        B, T, D = x.shape
        h = self.encoder(x.reshape(B * T, D))           # (B*T, hidden//2)
        scores = self.score_head(h).reshape(B, T)        # (B, T)

        if return_proj:
            proj = self.proj_head(h)                     # (B*T, proj_dim)
            proj = F.normalize(proj, dim=-1)             # L2 normalise
            proj = proj.reshape(B, T, self.proj_dim)     # (B, T, proj_dim)
            return scores, proj

        return scores



class UMILLoss(nn.Module):
    """
    UMIL Composite Loss = MIL Ranking Loss + Contrastive Unbiasing Loss.

    Contrastive objective (simplified from paper):
      1. Select top-K highest-scoring segments in each anomaly bag
         → these are the model's current best guess at anomalous segments
      2. Compute cosine similarity between these pseudo-anomalies and
         ALL segments from normal bags
      3. Penalise high similarity with a margin: the anomaly feature should
         be at least (margin) less similar to normal features than to itself
         (i.e., they must be in different regions of the embedding sphere)

    Weight lambda controls the trade-off. We use 0.1 (conservative) so that
    the contrastive term guides but doesn't dominate the ranking loss.
    """
    def __init__(self, margin=1.0, sparsity_weight=8e-3, smooth_weight=8e-4,
                 contrastive_weight=0.1, contrastive_margin=0.3, top_k=3):
        super().__init__()
        self.mil = MILRankingLoss(margin, sparsity_weight, smooth_weight)
        self.contrastive_weight = contrastive_weight
        self.contrastive_margin = contrastive_margin
        self.top_k = top_k

    def _unbiasing_loss(self, anom_scores, norm_scores, anom_proj, norm_proj):
        """
        Push top-K anomaly projections away from all normal projections.

        anom_scores: (B_a, T)        anom_proj:   (B_a, T, D)
        norm_scores: (B_n, T)        norm_proj:   (B_n, T, D)
        """
        B_a, T, D = anom_proj.shape
        B_n = norm_proj.shape[0]

        # Select top-K scoring segments per anomaly bag → (B_a * K, D)
        k = min(self.top_k, T)
        topk_idx = anom_scores.topk(k, dim=1).indices          # (B_a, K)
        anom_sel = torch.stack(
            [anom_proj[b, topk_idx[b]] for b in range(B_a)], dim=0
        ).reshape(-1, D)                                        # (B_a*K, D) — L2-normalised

        # All normal segments → (B_n * T, D)
        norm_all = norm_proj.reshape(B_n * T, D)               # (B_n*T, D) — L2-normalised

        # Cosine similarity matrix (values in [-1, 1] because both are L2-normalised)
        sim = torch.mm(anom_sel, norm_all.T)                   # (B_a*K, B_n*T)

        # Margin loss: penalise similarity above (-contrastive_margin)
        # i.e., anomaly features should be at most -margin similar to normal ones
        # relu(sim - (-margin)) = relu(sim + margin)
        loss = F.relu(sim + self.contrastive_margin).mean()
        return loss

    def forward(self, anom_scores, norm_scores, anom_proj=None, norm_proj=None):
        mil_loss, l_rank, l_sp, l_sm = self.mil(anom_scores, norm_scores)

        if anom_proj is not None and norm_proj is not None:
            cont_loss = self._unbiasing_loss(
                anom_scores.detach(), norm_scores.detach(),  # detach scores for index selection
                anom_proj, norm_proj
            )
            total = mil_loss + self.contrastive_weight * cont_loss
            cont_val = cont_loss.item()
        else:
            total = mil_loss
            cont_val = 0.0

        return total, l_rank, l_sp, l_sm, cont_val

