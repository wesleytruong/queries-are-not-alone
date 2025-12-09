import torch
import torch.nn as nn
from typing import Tuple

class VTCAttention(nn.Module):
    """
    Video-Text Cluster Attention (VTC-Att).

    Inputs:
        F:  (B, T, D)         # video frame embeddings
        N:  (B, K+1, D)       # text cluster embeddings [t_i; neighbors]
        h:  (B, K+1, D)       # Sweeper semantic signals

    Output:
        t_hat: (B, D)         # refined text embedding
    """

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim

        # First MHA: Q from F, K/V from h
        self.mha_q = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

        # Second MHA: Q from N, K/V from h
        self.mha_k = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

        # Final MHA: Q from Q_out, K from K_out, V from N
        self.mha_final = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

        # Projections (W_Q, W_K, W_V, etc. in paper)
        self.proj_q_Q = nn.Linear(embed_dim, embed_dim)
        self.proj_q_K = nn.Linear(embed_dim, embed_dim)
        self.proj_q_V = nn.Linear(embed_dim, embed_dim)

        self.proj_k_Q = nn.Linear(embed_dim, embed_dim)
        self.proj_k_K = nn.Linear(embed_dim, embed_dim)
        self.proj_k_V = nn.Linear(embed_dim, embed_dim)

        self.proj_f_Q = nn.Linear(embed_dim, embed_dim)
        self.proj_f_K = nn.Linear(embed_dim, embed_dim)
        self.proj_f_V = nn.Linear(embed_dim, embed_dim)

        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_k = nn.LayerNorm(embed_dim)
        self.ln_final = nn.LayerNorm(embed_dim)

    def forward(self, F: torch.Tensor, N: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        F: (B, T, D)
        N: (B, K+1, D)
        h: (B, K+1, D)
        Return:
            t_hat: (B, D)
        """
        B, T, D = F.shape

        # --- First MHA: Q from F, K/V from h  ---
        q1 = self.proj_q_Q(F)      # (B, T, D)
        k1 = self.proj_q_K(h)      # (B, K+1, D)
        v1 = self.proj_q_V(h)      # (B, K+1, D)

        Q_out, _ = self.mha_q(q1, k1, v1)   # (B, T, D)
        Q_out = self.ln_q(Q_out)

        # Compress Q_out over time dimension T -> single vector per batch
        # e.g., mean pooling over frames
        Q_pooled = Q_out.mean(dim=1, keepdim=True)   # (B, 1, D)

        # --- Second MHA: Q from N, K/V from h ---
        q2 = self.proj_k_Q(N)      # (B, K+1, D)
        k2 = self.proj_k_K(h)      # (B, K+1, D)
        v2 = self.proj_k_V(h)      # (B, K+1, D)

        K_out, _ = self.mha_k(q2, k2, v2)   # (B, K+1, D)
        K_out = self.ln_k(K_out)

        # --- Final MHA: Q = Q_pooled, K = K_out, V = N ---
        q_final = self.proj_f_Q(Q_pooled)   # (B, 1, D)
        k_final = self.proj_f_K(K_out)      # (B, K+1, D)
        v_final = self.proj_f_V(N)          # (B, K+1, D)

        out, _ = self.mha_final(q_final, k_final, v_final)  # (B, 1, D)
        out = self.ln_final(out)                             # (B, 1, D)

        t_hat = out.squeeze(1)  # (B, D)
        return t_hat

