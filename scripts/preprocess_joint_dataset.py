"""
Build joint-training data from the preprocessed video-caption dataset.

Steps:
1) Load HF dataset saved by preprocess_dataset.py (contains frames + caption).
2) Encode all captions with the trained text clusterer (CLIP text encoder).
3) Build a FAISS index over text embeddings and retrieve K nearest neighbors.
4) Compute Jaccard similarity labels using CLIP tokenization for Sweeper loss.
5) Encode the sampled frames to CLIP image embeddings.
6) Save a new HF dataset with text clusters, labels, and video embeddings.
"""

import argparse
import os
from pathlib import Path
from typing import List

import faiss
import torch
import torch.nn.functional as F
from datasets import Dataset, Features, Sequence, Value, Array2D, load_from_disk
from tqdm import tqdm

# CLIP import with packaging fallback
try:
    import clip
except Exception:
    import packaging
    import pkg_resources

    pkg_resources.packaging = packaging
    import clip

from config import load_data_config


def load_clip_text_encoder(model_name: str, checkpoint: str | None, device: torch.device):
    model, preprocess = clip.load(model_name, device=device, jit=False)
    if checkpoint and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"Warning: missing keys {missing}, unexpected keys {unexpected} when loading {checkpoint}")
    model.eval()
    return model, preprocess


def encode_captions(model, texts: List[str], device: torch.device, batch_size: int = 256) -> torch.Tensor:
    all_embeds: List[torch.Tensor] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding captions"):
            batch = texts[i : i + batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(device)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embeds.append(emb.cpu())
    return torch.cat(all_embeds, dim=0)


def build_faiss_index(embeds: torch.Tensor, use_gpu: bool) -> faiss.Index:
    vecs = embeds.numpy().astype("float32")
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(vecs)
    return index


def tokens_to_set(text: str) -> set[str]:
    # Simple word-level tokenization for Jaccard similarity
    return set(text.lower().strip().split())


def jaccard_label(a_tokens: set[str], b_tokens: set[str], num_buckets: int) -> int:
    """
    Bucket Jaccard similarity into num_buckets evenly spaced bins over [0, 1].
    """
    if num_buckets < 1:
        raise ValueError("num_buckets must be >= 1")
    if not a_tokens and not b_tokens:
        return num_buckets - 1

    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    sim = inter / union if union else 0.0
    sim = max(0.0, min(sim, 1.0))

    bucket = int(sim * num_buckets)
    return min(bucket, num_buckets - 1)


def encode_frames(model, preprocess, frames: List, device: torch.device) -> torch.Tensor:
    tensors = [preprocess(f).unsqueeze(0) for f in frames]
    batch = torch.cat(tensors, dim=0).to(device)
    with torch.no_grad():
        emb = model.encode_image(batch)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build joint-training dataset with text/video embeddings."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to TOML config (DataConfig with joint_* fields).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Config carries joint-preprocessing params (see DataConfig joint_* fields).
    cfg_path = args.config

    cfg = load_data_config(cfg_path)
    device = torch.device(cfg.joint_device if torch.cuda.is_available() else "cpu")
    k = cfg.joint_k
    text_ckpt = cfg.joint_text_checkpoint
    clip_model_name = cfg.joint_clip_model
    use_faiss_gpu = cfg.joint_use_faiss_gpu
    out_dir = Path(cfg.joint_out_dir)
    text_batch_size = cfg.joint_text_batch_size
    jaccard_bins = cfg.joint_jaccard_bins

    print("Loading preprocessed dataset...")
    base_ds = load_from_disk(os.path.join(cfg.out_dir, f"{cfg.dataset_tag}_{cfg.split}"))

    print("Loading CLIP (text clusterer) ...")
    text_model, preprocess = load_clip_text_encoder(clip_model_name, text_ckpt, device)

    print("Encoding captions...")
    captions = [row["caption"] for row in base_ds]
    text_embeds = encode_captions(text_model, captions, device=device, batch_size=text_batch_size)
    dim = text_embeds.shape[1]

    print("Building FAISS index...")
    index = build_faiss_index(text_embeds, use_gpu=use_faiss_gpu)

    print("Preparing generator...")
    token_cache = [tokens_to_set(c) for c in captions]

    def gen():
        for idx, row in enumerate(tqdm(base_ds, desc="Creating joint dataset")):
            query_emb = text_embeds[idx]
            q = query_emb.unsqueeze(0).numpy().astype("float32")
            faiss.normalize_L2(q)
            _, neighbors = index.search(q, k + 1)
            neighbor_ids = neighbors[0].tolist()

            cluster_embeds = text_embeds[neighbor_ids].numpy().astype("float32")

            labels = [
                jaccard_label(token_cache[idx], token_cache[n_id], jaccard_bins)
                for n_id in neighbor_ids
            ]

            frames = row["frames"]
            video_embeds = encode_frames(text_model, preprocess, frames, device=device).numpy().astype("float32")

            yield {
                "query_id": idx,
                "caption": row["caption"],
                "text_cluster_embeds": cluster_embeds,
                "sweeper_labels": labels,
                "video_embeds": video_embeds,
            }

    features = Features(
        {
            "query_id": Value("int64"),
            "caption": Value("string"),
            "text_cluster_embeds": Array2D(shape=(k + 1, dim), dtype="float32"),
            "sweeper_labels": Sequence(Value("int64"), length=k + 1),
            "video_embeds": Array2D(shape=(cfg.num_frames, dim), dtype="float32"),
        }
    )

    print("Building HF dataset with embeddings...")
    joint_ds = Dataset.from_generator(gen, features=features)

    out_dir.mkdir(parents=True, exist_ok=True)
    joint_ds.save_to_disk(str(out_dir))
    print(f"Saved joint dataset to {out_dir}")


if __name__ == "__main__":
    main()
