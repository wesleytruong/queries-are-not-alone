from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Optional, Type, TypeVar
import tomllib

@dataclass
class DataConfig:
    out_dir: str = "assets/datasets/msrvtt_hf"
    dataset_config: str = "train_7k"
    split: str = "train"
    num_frames: int = 8
    resize: int = 224
    dataset_name: str = "friedrichor/MSR-VTT"
    video_column: str = "video"
    caption_column: str = "caption"
    id_column: str = "video_id"
    dataset_tag: str = "msrvtt"
    video_root: str = "data/msrvtt_raw/video"
    video_ext: str = ".mp4"
    # joint preprocessing (for preprocess_joint_dataset.py)
    joint_out_dir: str = "data/msrvtt_joint_preprocessed"
    joint_text_checkpoint: str = "checkpoints/text_clusterer_clip.pt"
    joint_k: int = 5
    joint_clip_model: str = "ViT-B/16"
    joint_use_faiss_gpu: bool = False
    joint_text_batch_size: int = 256
    joint_device: str = "cuda"

@dataclass
class TrainJointConfig:
    # data
    dataset_name: str = "MSRVTT"
    num_frames: int = 8
    text_embed_dim: int = 512
    video_embed_dim: int = 512
    hidden_dim: int = 512
    knn_k: int = 5
    joint_dataset_path: str = "assets/datasets/msrvtt_joint_preprocessed"

    # training
    batch_size: int = 64
    num_epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cuda"
    num_workers: int = 4
    grad_clip: float | None = None

    # losses
    temperature_init: float = 0.07
    sweeper_label_smoothing: float = 0.0

    # paths (you’ll adapt these)
    clip_model_name: str = "ViT-B/16"
    clip_pretrained: str = "openai"
    knn_index_path: str = "indices/caption_index.faiss"
    caption_embeddings_path: str = "indices/caption_embeds.pt"
    checkpoint_path: str = "assets/checkpoints/retrieval_model.pt"

@dataclass
class TrainClustererConfig:
    # data
    dataset_path: str = "assets/datasets/msrvtt_processed/msrvtt_train"
    max_captions_per_video: int = 1
    max_texts: Optional[int] = None

    # dataloader
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    drop_last: bool = True

    # model / training
    clip_model_name: str = "ViT-B/16"
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 0.01
    margin: float = 0.2
    num_epochs: int = 1
    positive_dropout: float = 0.1
    anchor_dropout: float = 0.0
    checkpoint_path: str = "assets/checkpoints/text_clusterer_clip.pt"
    device: str = "cuda"
    seed: int = 42

@dataclass
class EvaluationConfig:
    # which datasets/checkpoint to use
    train_config: str = "config/data_configs/msrvtt.toml"
    test_config: str = "config/data_configs/msrvtt_test.toml"
    checkpoint: str = "checkpoints/retrieval_model.pt"

    # evaluation options
    k: int | None = None
    text_batch_size: int = 256
    use_faiss_gpu: bool = False
    device: str = "cuda"
    save_embeds: str | None = None


T = TypeVar("T")

def load_dataclass_from_toml(cls: Type[T], path: str) -> T:
    if not is_dataclass(cls):
        raise TypeError("cls must be a dataclass type")

    cfg_path = Path(path).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file {cfg_path} does not exist")

    with cfg_path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {cfg_path} must contain a TOML table.")

    defaults = asdict(cls())
    merged = {**defaults, **data}
    return cls(**merged)

def load_data_config(path: str) -> DataConfig:
    return load_dataclass_from_toml(DataConfig, path)

def load_joint_config(path: str) -> TrainJointConfig:
    return load_dataclass_from_toml(TrainJointConfig, path)

def load_clusterer_config(path: str) -> TrainClustererConfig:
    return load_dataclass_from_toml(TrainClustererConfig, path)

def load_evaluation_config(path: str) -> EvaluationConfig:
    return load_dataclass_from_toml(EvaluationConfig, path)
