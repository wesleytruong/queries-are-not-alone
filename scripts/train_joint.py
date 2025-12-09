import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from datasets import load_from_disk

from config import TrainJointConfig, load_joint_config
from dataclasses import asdict
from model import RetrievalModel
from utils.loss import info_nce_loss, sweeper_loss


class JointRetrievalDataset(Dataset):
    """
    Thin wrapper around the HF dataset produced by preprocess_joint_dataset.py.

    Each item returns:
        frame_feats: (T, D) float32 video embeddings
        text_cluster_embeds: (K+1, D) float32 text embeddings
        sweeper_labels: (K+1,) int64 labels for sweeper classifier
    """

    def __init__(self, path: str):
        ds_path = Path(path).expanduser()
        if not ds_path.exists():
            raise FileNotFoundError(f"Joint dataset path not found: {ds_path}")
        self.ds = load_from_disk(str(ds_path))

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.ds[idx]
        frame_feats = torch.tensor(row["video_embeds"], dtype=torch.float32)
        text_cluster_embeds = torch.tensor(row["text_cluster_embeds"], dtype=torch.float32)
        sweeper_labels = torch.tensor(row["sweeper_labels"], dtype=torch.long)
        return {
            "frame_feats": frame_feats,
            "text_cluster_embeds": text_cluster_embeds,
            "sweeper_labels": sweeper_labels,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    frame_feats = torch.stack([b["frame_feats"] for b in batch], dim=0)
    text_cluster_embeds = torch.stack([b["text_cluster_embeds"] for b in batch], dim=0)
    sweeper_labels = torch.stack([b["sweeper_labels"] for b in batch], dim=0)
    return {
        "frame_feats": frame_feats,
        "text_cluster_embeds": text_cluster_embeds,
        "sweeper_labels": sweeper_labels,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train retrieval model on joint dataset.")
    parser.add_argument(
        "--config",
        help="Path to TOML config for joint training.",
    )
    return parser.parse_args()


def train():
    args = parse_args()
    cfg = load_joint_config(args.config) if args.config else TrainJointConfig()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    dataset = JointRetrievalDataset(cfg.joint_dataset_path)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    model = RetrievalModel(cfg).to(device)
    # Force CLIP backbone to run in float32 to avoid mixed-precision dtype issues
    model.clip_model.float()

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.num_epochs):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            cluster_embeds = batch["text_cluster_embeds"]

            t_hat, v_emb, h, sweeper_logits = model(batch, cluster_embeds)

            L_ret = info_nce_loss(t_hat, v_emb, model.temperature)
            L_sweep = sweeper_loss(
                sweeper_logits,
                batch["sweeper_labels"],
                label_smoothing=cfg.sweeper_label_smoothing,
            )
            L_total = L_ret + L_sweep

            optimizer.zero_grad(set_to_none=True)
            L_total.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            running_loss += L_total.item()
            if (step + 1) % 10 == 0 or (step + 1) == len(loader):
                avg_loss = running_loss / (step + 1)
                print(
                    f"Epoch {epoch+1}/{cfg.num_epochs} "
                    f"Step {step+1}/{len(loader)} "
                    f"Loss {L_total.item():.4f} (avg {avg_loss:.4f}) "
                    f"L_ret {L_ret.item():.4f} L_sweep {L_sweep.item():.4f}"
                )

    ckpt_path = Path(cfg.checkpoint_path).expanduser()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": asdict(cfg),
        },
        ckpt_path,
    )
    print(f"Saved retrieval model checkpoint to {ckpt_path}")


if __name__ == "__main__":
    train()
