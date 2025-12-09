import torch
import torch.nn as nn
import torch.nn.functional as F

def info_nce_loss(
    t_emb: torch.Tensor,
    v_emb: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """
    InfoNCE loss over batch.

    t_emb: (B, D)
    v_emb: (B, D)
    temperature: scalar tensor
    """
    t_norm = F.normalize(t_emb, dim=-1)
    v_norm = F.normalize(v_emb, dim=-1)

    logits = torch.matmul(t_norm, v_norm.t())  # (B, B)
    logits = logits / temperature

    labels = torch.arange(t_emb.size(0), device=t_emb.device)
    loss_t2v = F.cross_entropy(logits, labels)
    loss_v2t = F.cross_entropy(logits.t(), labels)

    return (loss_t2v + loss_v2t) / 2.0

def sweeper_loss(
    sweeper_logits: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    sweeper_logits: (B, K+1, num_segments)
    targets: (B, K+1)   # int segment index per caption
    """
    B, K1, S = sweeper_logits.shape
    sweeper_logits = sweeper_logits.view(B * K1, S)
    targets = targets.view(B * K1)

    if label_smoothing > 0.0:
        # simple label smoothing wrapper
        with torch.no_grad():
            true_dist = torch.zeros_like(sweeper_logits)
            true_dist.fill_(label_smoothing / (S - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1 - label_smoothing)
        log_probs = F.log_softmax(sweeper_logits, dim=1)
        loss = -(true_dist * log_probs).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(sweeper_logits, targets)

    return loss

def triplet_loss(anchor: torch.Tensor,
                 positive: torch.Tensor,
                 negative: torch.Tensor,
                 margin: float,
                 debug=False, 
                 batch_idx=None) -> torch.Tensor:
    """
    L_C = max(0, U(a,p) - U(a,n) + margin)
    """
    # how far anchor is from same caption (with dropout)
    d_pos = cosine_distance(anchor, positive)
    # how far anchor is from different caption
    d_neg = cosine_distance(anchor, negative)
    # d_pos - d_neg + margin is negative if anchor is closer to positive than negative
    # it is positive if anchor is relatively close to negative 
    # margin is how far apart we want d_pos and d_neg to be; if closer, (d_pos - d_neg + margin) is gonna be positive
    raw = d_pos - d_neg + margin
    loss = F.relu(raw)

    # print the values for debugging every 10 batches
    if debug and batch_idx is not None and batch_idx % 10 == 0:
        print(
            f"[Batch {batch_idx}] "
            f"d_pos (cos(anchor, pos))={d_pos.mean().item():.4f}, "
            f"d_neg (cos(anchor, neg))={d_neg.mean().item():.4f}, "
            f"raw={raw.mean().item():.4f}, "
            f"loss={loss.mean().item():.4f}"
        )
    return loss.mean()
