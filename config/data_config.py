import tomllib
from typing import Dict, Any
from types import SimpleNamespace
from pathlib import Path

DEFAULT_CONFIG: Dict[str, Any] = {
    "out_dir": "data/msrvtt_hf",
    "dataset_config": "train_7k",
    "split": "train",
    "num_frames": 8,
    "resize": 224,
    "dataset_name": "friedrichor/MSR-VTT",
    "video_column": "video",
    "caption_column": "caption",
    "id_column": "video_id",
    "dataset_tag": "msrvtt",
    "video_root": "data/msrvtt_raw/video",
    "video_ext": ".mp4",
}


def load_config(path: str) -> SimpleNamespace:
    cfg_path = Path(path).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file {cfg_path} does not exist")
    with cfg_path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {cfg_path} must contain a TOML table.")
    merged = dict(DEFAULT_CONFIG)
    merged.update(data)
    return SimpleNamespace(**merged)

