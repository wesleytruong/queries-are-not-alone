import torch
import torch.nn as nn

class TextVideoFusion(nn.Module):
    """
    Fuse refined text embedding t_hat with video frame embeddings F.

    Inputs:
        t_hat: (B, D)
        F:     (B, T, D)

    Output:
        v:     (B, D)  # final video embedding
    """

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

        self.proj_Q = nn.Linear(embed_dim, embed_dim)
        self.proj_K = nn.Linear(embed_dim, embed_dim)
        self.proj_V = nn.Linear(embed_dim, embed_dim)

        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, t_hat: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """
        t_hat: (B, D)
        F: (B, T, D)
        """
        B, T, D = F.shape

        q = self.proj_Q(t_hat).unsqueeze(1)  # (B, 1, D)
        k = self.proj_K(F)                   # (B, T, D)
        v = self.proj_V(F)                   # (B, T, D)

        out, _ = self.mha(q, k, v)           # (B, 1, D)
        out = self.ln(out)                   # (B, 1, D)

        v_emb = out.squeeze(1)               # (B, D)
        return v_emb

