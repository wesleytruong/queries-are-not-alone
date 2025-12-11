# queries-are-not-alone
Repo Authors: Wesley Truong and Adyah Rastogi 

Repository for reproducing the results of the paper 
**“Queries Are Not Alone: Clustering Text Embeddings for Video Search”** by Peiyang Liu, Xi Wang, Ziqiang Cui, Wei Ye.  
- ACM: https://dl.acm.org/doi/10.1145/3726302.3730066  
- arXiv: https://arxiv.org/pdf/2510.07720  

## Reproducability

Our project currently targets [MSR-VTT](https://huggingface.co/datasets/friedrichor/MSR-VTT) by default.
This dataset contains videos from ~20 categories and has 20 sentences per video through AMT. 

The data preprocessing is done under /scripts/preprocess_dataset.py and scripts/preprocess_joint_dataset.py, and all models are defined under /models. The configurations are written per script under /configs for ease of editing hyperparameters and any models required. 
We also use the openai-clip package, specifically using the CLIP-ViT-B/16 model by default.

More detailed explanation of our reproduction is in our [report](https://github.com/wesleytruong/queries-are-not-alone/blob/main/Reproduction%20Paper.pdf).

```bash
# If not installed, install PyTorch and ensure ffmpeg is installed on the machine
# PyTorch: https://pytorch.org/get-started/locally/
# Install uv: https://docs.astral.sh/uv/getting-started/installation/
# sync packages and activate venv
uv sync
uv pip install -e .
source ./.venv/bin/activate

# Download and unzip dataset files
python scripts/download_msrvtt.py

# Preprocess video files into sampled frames + captions
python ./scripts/preprocess_dataset.py --config ./config/data_configs/msrvtt.toml
python ./scripts/preprocess_dataset.py --config ./config/data_configs/msrvtt_test.toml

# train text clusterer
python ./scripts/train_clusterer.py --config ./config/clusterer_configs/msrvtt.toml

# precompute knn clusters for joint training and preprocess text and frame embeddings
python ./scripts/preprocess_joint_dataset.py --config ./config/data_configs/msrvtt.toml

# perform joint training on sweeper, vtc_attention, and text-video fusion layer
python ./scripts/train_joint.py --config ./config/joint_configs/msrvtt.toml

# perform evaluation script on msrvtt test set
python ./scripts/evaluate_retrieval.py --config ./config/evaluation_configs/msrvtt.toml
```

### Results

| test run | R@1 | R@5 | R@10 |
|-|-|-|-|
| 1 | 0.5230 | 0.7620 | 0.8300 |
| 2 | 0.5040 | 0.7530 | 0.8310 |
| 3 | 0.5580 | 0.7580 | 0.8150 |

Text-to-video retrieval results on MSRVTT using CLIP-ViT-B/16
