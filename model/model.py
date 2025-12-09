import torch
import torch.nn as nn
from typing import Dict, Any, List, Union

# OpenAI CLIP import with packaging fallback
try:
    import clip
except Exception:
    import packaging
    import pkg_resources

    pkg_resources.packaging = packaging
    import clip

from .sweeper import Sweeper
from .vtc_att import VTCAttention
from .fusion import TextVideoFusion

class RetrievalModel(nn.Module):
    """
    Wraps:
        - text encoder (CLIP text)
        - video encoder (CLIP image)
        - sweeper
        - VTC-Att
        - fusion
    """

    def __init__(self, cfg):
        super().__init__()
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # Load CLIP once during model construction (no deferred/lazy load)
        clip_model_name = getattr(cfg, "clip_model_name", "ViT-B/16")
        self.clip_model, self.clip_preprocess = clip.load(
            clip_model_name, device=device, jit=False
        )
        clip_dim = self.clip_model.text_projection.shape[1]

        D = getattr(cfg, "hidden_dim", clip_dim)
        if D != clip_dim:
            raise ValueError(
                f"hidden_dim ({D}) must match CLIP embed dim ({clip_dim}); "
                "update config or add a projection layer."
            )

        self.device = device

        self.sweeper = Sweeper(embed_dim=D)
        self.vtc_att = VTCAttention(embed_dim=D)
        self.fusion = TextVideoFusion(embed_dim=D)

        # learnable temperature for InfoNCE
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(cfg.temperature_init))
        )

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def encode_video(self, frame_feats: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames with CLIP.

        frame_feats can be:
            - (B, T, 3, H, W) raw frames already resized/normalized for CLIP
            - (B, T, D) precomputed CLIP embeddings (returned as-is)

        returns: F (B, T, D)
        """
        if frame_feats.ndim == 3:
            # Already in embedding space; just normalize to be safe
            F = frame_feats
        elif frame_feats.ndim == 5:
            B, T = frame_feats.shape[:2]
            flat_frames = frame_feats.reshape(B * T, *frame_feats.shape[2:])
            flat_frames = flat_frames.to(self.device)
            embeds = self.clip_model.encode_image(flat_frames)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            F = embeds.view(B, T, -1)
        else:
            raise ValueError(
                f"Expected frame_feats with ndim 3 (embeds) or 5 (frames), got {frame_feats.ndim}"
            )

        F = F.to(self.device, dtype=torch.float32)
        norm = F.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return F / norm

    def encode_text_cluster(self, cluster_feats: Union[torch.Tensor, List[str]]) -> torch.Tensor:
        """
        cluster_feats:
            - tensor: (B, K+1, D) precomputed embeddings (returned as-is)
            - list[str] / tuple[str]: flat captions to encode with CLIP text encoder
        """
        if isinstance(cluster_feats, torch.Tensor):
            feats = cluster_feats.to(self.device, dtype=torch.float32)
            norm = feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            return feats / norm

        if isinstance(cluster_feats, (list, tuple)):
            tokens = clip.tokenize(list(cluster_feats)).to(self.device)
            embeds = self.clip_model.encode_text(tokens)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            return embeds

        raise TypeError(
            "cluster_feats must be a torch.Tensor or list/tuple of strings"
        )

    def forward(
        self,
        batch: Dict[str, Any],
        cluster_embeds: torch.Tensor,
    ):
        """
        batch should contain:
            - 'frame_feats': (B, T, frame_feat_dim)
        cluster_embeds:
            - (B, K+1, D) text embeddings for [query; neighbors]

        Returns:
            t_hat: (B, D)
            v_emb: (B, D)
            h: (B, K+1, D)
            sweeper_logits: (B, K+1, num_segments)
        """
        F = self.encode_video(batch["frame_feats"])      # (B, T, D)

        # Sweeper over text cluster embeddings
        h, sweeper_logits = self.sweeper(cluster_embeds)  # (B, K+1, D), (B, K+1, S)

        # VTC-Att to refine text embedding
        t_hat = self.vtc_att(F, cluster_embeds, h)        # (B, D)

        # Text-Video fusion for video embedding
        v_emb = self.fusion(t_hat, F)                     # (B, D)

        return t_hat, v_emb, h, sweeper_logits
