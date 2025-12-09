# scripts/train_text_clusterer.py

import argparse
import random
from pathlib import Path
from typing import List, Optional

import ast
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
try:
    import clip
except Exception:
    import packaging
    import pkg_resources

    pkg_resources.packaging = packaging
    import clip

from config import load_clusterer_config


#  keep track of the index in the entire text corpus


class TextCorpusDataset(Dataset):
    """
    simple text-only corpus dataset.
    each item: {"idx": index_in_corpus, "text": caption_string}
    need this so we can sample negatives from the dataset without replacement
    """
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return {
            "idx": idx,
            "text": self.texts[idx],
        }


def collate_batch(batch):
    """
    collate function for DataLoader, can't handle lists of dicts natively
    input: list of {"idx": int, "text": str}
    output: {"idxs": [..], "texts": [..]}
    """
    idxs = [b["idx"] for b in batch]
    texts = [b["text"] for b in batch]
    return {"idxs": idxs, "texts": texts}


#  loss function

def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    cosine distance U(a, b) = 1 - cos(a, b)
    a, b: [B, d]
    returns: [B]
    """
    return 1.0 - F.cosine_similarity(a, b, dim=-1)


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

#  data loading from Arrow

def load_captions_from_arrow(
    hf_path: str, max_captions_per_video: int, max_texts: Optional[int] = None
) -> List[str]:
    """
    load captions from a HuggingFace dataset saved with save_to_disk
    each row has:
      - "caption": either a stringified Python list of captions, or a list[str]

    returns a flat list of cleaned caption strings
    """
    ds = load_from_disk(hf_path)
    texts: List[str] = []

    for row in ds:
        raw = row["caption"]

        # if the caption is a string that looks like a list, parse it
        if isinstance(raw, str) and raw.startswith("["):
            try:
                caps = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                caps = []
        elif isinstance(raw, list):
            caps = raw
        else:
            caps = []

        cap_limit = max_captions_per_video if max_captions_per_video and max_captions_per_video > 0 else None
        if cap_limit and len(caps) > cap_limit:
            caps = random.sample(caps, cap_limit)

        # clean whitespace and filter out empty captions
        for c in caps:
            if isinstance(c, str):
                c = c.strip()
                if c:
                    texts.append(c)

    if max_texts and max_texts > 0 and len(texts) > max_texts:
        texts = random.sample(texts, max_texts)

    return texts


#  CLIP text encoder wrapper

def encode_texts(model,
                 texts: List[str],
                 device: str,
                 dropout_p: float = 0.0) -> torch.Tensor:
    """
    encode a batch of texts with CLIP and optionally apply dropout
    on the normalized embeddings to simulate Φ_C(x, z)

    returns: [B, d] tensor of L2-normalized embeddings
    """
    tokens = clip.tokenize(texts).to(device)
    feats = model.encode_text(tokens)                 # [B, d]
    # normalize with small epsilon to avoid div-by-zero
    norm = feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    feats = feats / norm  # normalize

    if dropout_p and dropout_p > 0.0:
        feats = F.dropout(feats, p=dropout_p, training=True)

    return feats


#  main training loop

def parse_args():
    parser = argparse.ArgumentParser(description="Train the CLIP-based text clusterer.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the TOML config file with clusterer training parameters.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_clusterer_config(args.config)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    requested_device = cfg.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; using CPU instead.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    hf_path = Path(cfg.dataset_path).expanduser()
    if not hf_path.exists():
        raise FileNotFoundError(f"HuggingFace dataset path not found: {hf_path}")
    texts = load_captions_from_arrow(
        str(hf_path),
        max_captions_per_video=cfg.max_captions_per_video,
        max_texts=cfg.max_texts,
    )
    print(f"Loaded {len(texts)} captions from {hf_path}")

    dataset = TextCorpusDataset(texts)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_batch,
        drop_last=cfg.drop_last,
    )

    # load OpenAI CLIP model (text encoder)
    model, _ = clip.load(cfg.clip_model_name, device=device, jit=False)
    model = model.float()  # avoid fp16 overflows while training
    model.train()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.optimizer_lr,
        weight_decay=cfg.optimizer_weight_decay,
    )

    margin = cfg.margin
    num_epochs = cfg.num_epochs

    for epoch in range(num_epochs):
        running_loss = 0.0

        batch_idx = 0

        for batch in loader:
            idxs = batch["idxs"]       # [B]
            texts_batch = batch["texts"]

            anchor = encode_texts(model, texts_batch, device, dropout_p=cfg.anchor_dropout)  # Φ_C(x_i, z)
            # positive with different dropout masks on embeddings
            positive = encode_texts(model, texts_batch, device, dropout_p=cfg.positive_dropout)  # Φ_C(x_i, z')

            # sample random negatives from D \ {x_i}
            neg_texts = []
            for i in idxs:
                j = i
                while j == i:
                    j = random.randint(0, len(dataset) - 1)
                neg_texts.append(dataset.texts[j])

            negative = encode_texts(model, neg_texts, device)  # t*_q_i

            loss = triplet_loss(anchor, positive, negative, margin=margin)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            print(f"Batch {batch_idx}, Avg loss: {avg_loss:.6f}")
            batch_idx += 1

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}/{num_epochs} - L_C = {avg_loss:.4f}")

    checkpoint_path = Path(cfg.checkpoint_path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved fine-tuned CLIP text encoder to {checkpoint_path}")


if __name__ == "__main__":
    main()
