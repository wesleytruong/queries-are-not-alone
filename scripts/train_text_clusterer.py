# scripts/train_text_clusterer.py

import os
import random
from typing import List

import ast
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import clip


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

def load_captions_from_arrow(hf_path: str) -> List[str]:
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

        # cap to at most 1 captions per video
        MAX_CAPS = 1
        if len(caps) > MAX_CAPS:
            caps = random.sample(caps, MAX_CAPS)

        # clean whitespace and filter out empty captions
        for c in caps:
            if isinstance(c, str):
                c = c.strip()
                if c:
                    texts.append(c)

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
    feats = feats / feats.norm(dim=-1, keepdim=True)  # normalize

    if dropout_p and dropout_p > 0.0:
        feats = F.dropout(feats, p=dropout_p, training=True)

    return feats


#  main training loop

def main():
    random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_path = "data/msrvtt_hf/msrvtt_train" 
    texts = load_captions_from_arrow(hf_path)
    print(f"Loaded {len(texts)} captions")

    dataset = TextCorpusDataset(texts)
    loader = DataLoader(
        dataset,
        batch_size=32, # change if ur pc can handle more
        shuffle=True,
        num_workers=0, # change if ur pc can handle more my laptop can't
        collate_fn=collate_batch,
        drop_last=True,
    )

    # load OpenAI CLIP model (text encoder)
    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model.train() 

    # If you only care about text, you can freeze the visual branch:
    # for name, param in model.named_parameters():
    #     if "visual" in name:
    #         param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5,
        weight_decay=0.01,
    )

    margin = 0.2 # how far at least apart we want the pos and neg cos dist to be 
    num_epochs = 1 

    for epoch in range(num_epochs):
        running_loss = 0.0

        batch_idx = 0

        for batch in loader:
            idxs = batch["idxs"]       # [B]
            texts_batch = batch["texts"]

            anchor = encode_texts(model, texts_batch, device)  # Φ_C(x_i, z)
            # positive with different dropout masks on embeddings 
            positive = encode_texts(model, texts_batch, device, dropout_p=0.1)  # Φ_C(x_i, z')

            # sample random negatives from D \ {x_i}
            neg_texts = []
            for i in idxs:
                j = i
                while j == i:
                    j = random.randint(0, len(dataset) - 1)
                neg_texts.append(dataset.texts[j])

            negative = encode_texts(model, neg_texts, device)  # t*_q_i

            loss = triplet_loss(anchor, positive, negative, margin=margin, 
            debug=True, batch_idx=batch_idx # rm these later when don't want so many prints
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            print(f"Batch {batch_idx}, Avg loss: {avg_loss:.6f}")
            batch_idx += 1

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}/{num_epochs} - L_C = {avg_loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    out_path = "checkpoints/text_clusterer_clip.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Saved fine-tuned CLIP text encoder to {out_path}")


if __name__ == "__main__":
    main()
