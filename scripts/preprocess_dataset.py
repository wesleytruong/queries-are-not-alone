# scripts/preprocess_didemo_hf.py
import argparse
import os
from typing import Dict, Any, Iterator, List

import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, Video, Dataset, Features, Sequence, Value, Image as HFImage

def sample_equal_frames(video_array: np.ndarray, num_frames: int) -> np.ndarray:
    """
    video_array: [T, H, W, C] uint8
    return: [num_frames, H, W, C] sampled at equal timesteps
    """
    assert video_array.ndim == 4, f"Expected [T,H,W,C], got {video_array.shape}"
    T = video_array.shape[0]
    if T == 0:
        raise ValueError("Video has zero frames")

    if T >= num_frames:
        idx = np.linspace(0, T - 1, num_frames, dtype=int)
    else:
        idx = np.linspace(0, T - 1, num_frames, dtype=float)
        idx = np.round(idx).astype(int)

    return video_array[idx]


def frames_to_pil_list(frames: np.ndarray, resize: int) -> List[Image.Image]:
    """
    frames: [F, H, W, C] uint8
    returns: list of PIL.Image
    """
    pil_frames: List[Image.Image] = []
    F = frames.shape[0]
    for i in range(F):
        img = Image.fromarray(frames[i])
        if resize and resize > 0:
            img = img.resize((resize, resize), resample=Image.BICUBIC)
        img = img.convert("RGB")
        pil_frames.append(img)
    return pil_frames

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/didemo_hf",
        help="Directory where the HF dataset will be saved",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split of DiDeMo to preprocess",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of frames to sample uniformly per video",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=224,
        help="Resize frames to square [resize x resize]",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="friedrichor/DiDeMo",cat DiDeMo_Videos_mp4_train.tar.part-* | tar -vxf -
tar -xvf DiDeMo_Videos_mp4_test.tar
        help="Hugging Face dataset id for DiDeMo",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load DiDeMo in streaming mode
    stream_ds = load_dataset(
        args.dataset_name,
        split=args.split,
        streaming=True,
    )
    stream_ds = stream_ds.cast_column("video", Video())

    # Define HF dataset schema
    features = Features(
        {
            "video_id": Value("int32"),
            "caption": Value("string"),
            "frames": Sequence(HFImage()),  # list of images
        }
    )

    def gen() -> Iterator[Dict[str, Any]]:
        for idx, example in enumerate(tqdm(stream_ds, desc=f"Preprocessing {args.split}")):
            caption = example["caption"]
            video_info = example["video"]
            video_array = video_info["array"]  # [T, H, W, C]

            try:
                sampled = sample_equal_frames(video_array, args.num_frames)
            except ValueError:
                # skip broken / empty videos
                continue

            pil_frames = frames_to_pil_list(sampled, resize=args.resize)

            yield {
                "video_id": idx,
                "caption": caption,
                "frames": pil_frames,
            }

    # Build an in-memory HF Dataset from generator
    ds = Dataset.from_generator(gen, features=features)

    out_path = os.path.join(args.out_dir, f"didemo_{args.split}")
    ds.save_to_disk(out_path)
    print(f"Saved HF dataset split to: {out_path}")
    print(ds)


if __name__ == "__main__":
    main()

