import torch
import torch.nn as nn
from typing import Tuple

class Sweeper(nn.Module):
    """
    Sweeper: takes query + its K neighbors (text embeddings),
    and outputs semantic signals h_i for each element in the cluster.

    Input:
        cluster_embeds: (B, K+1, D)  # [query; neighbors]
    Output:
        h: (B, K+1, D)  # semantic signals per caption
        logits: (B, K+1, num_segments)  # if you do segment classification
    """

    def __init__(self, embed_dim: int, num_layers: int = 2,
                 num_heads: int = 8, ff_dim: int = 2048,
                 num_segments: int = 4):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_segments)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, cluster_embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        cluster_embeds: (B, K+1, D)
        """
        x = self.encoder(cluster_embeds)            # (B, K+1, D)
        h = self.layer_norm(self.proj(x))          # (B, K+1, D)
        logits = self.classifier(self.activation(h))  # (B, K+1, num_segments)
        return h, logits

