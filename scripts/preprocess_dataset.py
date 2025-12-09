import argparse
import os
from pathlib import Path
from typing import Dict, Any, Iterator, List

from config import load_data_config
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, Video, Dataset, Features, Sequence, Value, Image as HFImage

def sample_equal_frames(video_decoder, num_frames: int) -> np.ndarray:
    """
    Sample `num_frames` frames at equal(ish) timesteps from a VideoDecoder.

    Returns: [num_frames, H, W, C] uint8
    """

    # Decode ALL frames in one go: [T, 3, H, W]
    frames_obj = video_decoder.get_frames_in_range(0, 10**9, 1)
    frames_tensor = frames_obj.data  # torch.Tensor [T, 3, H, W]
    T = frames_tensor.shape[0]

    if T == 0:
        raise ValueError("Video has zero frames")

    # Compute indices exactly like your original function
    if T >= num_frames:
        idx = np.linspace(0, T - 1, num_frames, dtype=int)
    else:
        idx = np.linspace(0, T - 1, num_frames, dtype=float)
        idx = np.round(idx).astype(int)

    # Subsample on the tensor, then go to numpy [F, H, W, C]
    frames_tensor = frames_tensor[idx]                # [F, 3, H, W]
    frames_np = frames_tensor.permute(0, 2, 3, 1)     # [F, H, W, C]
    frames_np = frames_np.cpu().numpy().astype(np.uint8)

    return frames_np


def frames_to_pil_list(frames: np.ndarray, resize: int) -> List[Image.Image]:
    """
    frames: [F, H, W, C] uint8
    returns: list of PIL.Image
    """
    assert frames.ndim == 4, f"Expected [F,H,W,C], got shape {frames.shape}"
    F = frames.shape[0]
    pil_frames: List[Image.Image] = []

    for i in range(F):
        img = Image.fromarray(frames[i])
        if resize and resize > 0:
            img = img.resize((resize, resize), resample=Image.BICUBIC)
        img = img.convert("RGB")
        pil_frames.append(img)

    return pil_frames


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an HF video-caption dataset into sampled frame sequences."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a TOML config file describing preprocessing options.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_data_config(args.config)

    os.makedirs(cfg.out_dir, exist_ok=True)
    if not cfg.video_root:
        raise ValueError("Config must define a 'video_root' directory containing MSRVTT videos.")
    video_root = Path(cfg.video_root).expanduser()
    if not video_root.exists():
        raise FileNotFoundError(f"Video root {video_root} does not exist")
    normalized_ext = cfg.video_ext if cfg.video_ext.startswith(".") else f".{cfg.video_ext}"

    # Load dataset metadata in streaming mode
    stream_ds = load_dataset(
        cfg.dataset_name,
        name=cfg.dataset_config,
        split=cfg.split,
        streaming=True,
    )
    video_feature = Video()

    # Define HF dataset schema
    features = Features(
        {
            "video_id": Value("string"),
            "caption": Value("string"),
            "frames": Sequence(HFImage()),  # list of images
        }
    )

    def gen() -> Iterator[Dict[str, Any]]:
        for idx, example in enumerate(tqdm(stream_ds, desc=f"Preprocessing {cfg.split}")):
            caption = example.get(cfg.caption_column)
            if caption is None:
                continue

            raw_id = example.get(cfg.id_column)
            video_id = str(raw_id) if raw_id is not None else str(idx)

            candidates: List[Path] = []
            filename = example.get(cfg.video_column)
            if filename:
                filename_str = str(filename)
                path = Path(filename_str)
                if path.is_absolute():
                    candidates.append(path)
                else:
                    candidates.append(video_root / filename_str)
                    if not filename_str.endswith(normalized_ext):
                        candidates.append((video_root / filename_str).with_suffix(normalized_ext))

            video_path = next((c for c in candidates if c.exists()), None)
            if video_path is None:
                continue

            video_info = video_feature.decode_example({"path": str(video_path), "bytes": None})

            try:
                sampled = sample_equal_frames(video_info, cfg.num_frames)
            except ValueError:
                # skip broken / empty videos
                continue

            pil_frames = frames_to_pil_list(sampled, resize=cfg.resize)

            yield {
                "video_id": video_id,
                "caption": caption,
                "frames": pil_frames,
            }

    # Build an in-memory HF Dataset from generator
    ds = Dataset.from_generator(gen, features=features)

    out_path = os.path.join(cfg.out_dir, f"{cfg.dataset_tag}_{cfg.split}")
    ds.save_to_disk(out_path)
    print(f"Saved HF dataset split to: {out_path}")
    print(ds)


if __name__ == "__main__":
    main()
