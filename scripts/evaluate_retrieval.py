"""
Evaluation script for MSR-VTT text-to-video retrieval.

Pipeline (matches discussion):
- Build a FAISS KNN index over training captions (text embeddings from CLIP).
- For each test sample, form a text cluster [query + K neighbors] using that index.
- Encode the sample's frames with the RetrievalModel vision encoder.
- Query embedding t_q: run Sweeper + VTC-Att (stop before fusion).
- Video embedding v_i: run fusion(t_q, F) for each test video.
- Search video embeddings with FAISS (or dense matmul) and report Recall@{1,5,10}.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import faiss
import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm

from config import (
    DataConfig,
    EvaluationConfig,
    TrainJointConfig,
    load_data_config,
    load_evaluation_config,
)
from model import RetrievalModel

# CLIP import with packaging fallback (mirrors other scripts)
try:
    import clip
except Exception:
    import packaging
    import pkg_resources

    pkg_resources.packaging = packaging
    import clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RetrievalModel on MSR-VTT.")
    parser.add_argument(
        "--config",
        default="config/evaluation_configs/msrvtt.toml",
        help="Path to evaluation config TOML file.",
    )
    return parser.parse_args()


def load_retrieval_model(ckpt_path: Path, device: torch.device) -> Tuple[RetrievalModel, TrainJointConfig]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    cfg_dict = state.get("cfg", {})
    cfg = TrainJointConfig(**cfg_dict) if cfg_dict else TrainJointConfig()

    model = RetrievalModel(cfg).to(device)
    missing, unexpected = model.load_state_dict(state["model_state"], strict=False)
    if missing:
        print(f"[warn] Missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"[warn] Unexpected keys when loading checkpoint: {unexpected}")
    model.eval()
    return model, cfg


def encode_captions(
    clip_model,
    texts: List[str],
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Encode captions with CLIP text encoder; returns [N, D] float32 normalized."""
    all_embeds: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(device)
            feats = clip_model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            all_embeds.append(feats)
    return torch.cat(all_embeds, dim=0)


def build_text_index(
    model: RetrievalModel,
    captions: List[str],
    device: torch.device,
    batch_size: int,
    use_gpu: bool,
) -> Tuple[faiss.Index, np.ndarray]:
    text_embeds = encode_captions(model.clip_model, captions, device, batch_size)
    vecs = text_embeds.cpu().numpy().astype("float32")
    faiss.normalize_L2(vecs)

    index: faiss.Index = faiss.IndexFlatIP(vecs.shape[1])
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(vecs)
    return index, vecs


def encode_video_frames(model: RetrievalModel, frames: Iterable, device: torch.device) -> torch.Tensor:
    """Encode a list of PIL frames to shape [1, T, D] normalized embeddings."""
    tensors = [model.clip_preprocess(f).unsqueeze(0) for f in frames]
    batch = torch.cat(tensors, dim=0).to(device)
    with torch.no_grad():
        feats = model.clip_model.encode_image(batch)
        feats = feats.float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return feats.unsqueeze(0)  # (1, T, D)


def recall_at_k(sim: torch.Tensor, gt: torch.Tensor, k: int) -> float:
    topk = sim.topk(k, dim=1).indices
    hits = (topk == gt.unsqueeze(1)).any(dim=1).float()
    return hits.mean().item()


def main():
    args = parse_args()
    eval_cfg: EvaluationConfig = load_evaluation_config(args.config)
    device = torch.device(eval_cfg.device if torch.cuda.is_available() else "cpu")

    model, joint_cfg = load_retrieval_model(Path(eval_cfg.checkpoint), device)
    # ensure model + CLIP backbone run in float32 to avoid mixed dtype errors
    model.float()
    model.clip_model.float()
    knn_k = eval_cfg.k if eval_cfg.k is not None else joint_cfg.knn_k

    train_cfg: DataConfig = load_data_config(eval_cfg.train_config)
    test_cfg: DataConfig = load_data_config(eval_cfg.test_config)

    train_ds_path = Path(train_cfg.out_dir) / f"{train_cfg.dataset_tag}_{train_cfg.split}"
    test_ds_path = Path(test_cfg.out_dir) / f"{test_cfg.dataset_tag}_{test_cfg.split}"
    print(f"Loading train split from {train_ds_path}")
    train_ds = load_from_disk(str(train_ds_path))
    print(f"Loading test split from {test_ds_path}")
    test_ds = load_from_disk(str(test_ds_path))

    print("Encoding training captions and building FAISS index...")
    train_captions = [row["caption"] for row in train_ds]
    text_index, train_text_vecs = build_text_index(
        model,
        train_captions,
        device=device,
        batch_size=eval_cfg.text_batch_size,
        use_gpu=eval_cfg.use_faiss_gpu,
    )

    query_embeds: List[torch.Tensor] = []
    video_embeds: List[torch.Tensor] = []
    video_ids: List[str] = []

    print("Encoding test queries and videos...")
    for row in tqdm(test_ds, desc="Test samples"):
        caption: str = row["caption"]
        frames = row["frames"]

        # Query caption embedding
        cap_emb = encode_captions(
            model.clip_model, [caption], device, batch_size=1
        ).squeeze(0)  # (D,)

        # Retrieve neighbors from training captions
        q_vec = cap_emb.cpu().numpy().astype("float32").reshape(1, -1)
        faiss.normalize_L2(q_vec)
        _, neighbor_idx = text_index.search(q_vec, knn_k)
        neighbors = train_text_vecs[neighbor_idx[0]]  # (K, D)

        cluster = np.vstack([q_vec, neighbors])  # (K+1, D)
        cluster_tensor = torch.from_numpy(cluster).to(device=device, dtype=torch.float32)
        cluster_tensor = cluster_tensor.unsqueeze(0)  # (1, K+1, D)

        frame_feats = encode_video_frames(model, frames, device)  # (1, T, D)

        with torch.no_grad():
            h, _ = model.sweeper(cluster_tensor)
            t_hat = model.vtc_att(frame_feats, cluster_tensor, h)  # (1, D)
            v_emb = model.fusion(t_hat, frame_feats)              # (1, D)

        query_embeds.append(t_hat.squeeze(0).cpu())
        video_embeds.append(v_emb.squeeze(0).cpu())
        video_ids.append(row.get("video_id", str(len(video_ids))))

    query_mat = torch.stack(query_embeds, dim=0)   # (N, D)
    video_mat = torch.stack(video_embeds, dim=0)   # (N, D)
    query_mat = query_mat / query_mat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    video_mat = video_mat / video_mat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    sim = query_mat @ video_mat.T

    id_to_idx = {vid: i for i, vid in enumerate(video_ids)}
    gt_indices = torch.tensor([id_to_idx[vid] for vid in video_ids], dtype=torch.long)

    r1 = recall_at_k(sim, gt_indices, 1)
    r5 = recall_at_k(sim, gt_indices, 5)
    r10 = recall_at_k(sim, gt_indices, 10)

    print(f"Recall@1 : {r1:.4f}")
    print(f"Recall@5 : {r5:.4f}")
    print(f"Recall@10: {r10:.4f}")

    if eval_cfg.save_embeds:
        out_path = Path(eval_cfg.save_embeds)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "query_embeds": query_mat,
                "video_embeds": video_mat,
                "video_ids": video_ids,
                "recall": {"r1": r1, "r5": r5, "r10": r10},
                "k": knn_k,
            },
            out_path,
        )
        print(f"Saved embeddings and metrics to {out_path}")


if __name__ == "__main__":
    main()
