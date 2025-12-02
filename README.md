# queries-are-not-alone

Repository for reproducing the results of the paper 
**“Queries Are Not Alone: Clustering Text Embeddings for Video Search”** by Peiyang Liu, Xi Wang, Ziqiang Cui, Wei Ye.  
- ACM: https://dl.acm.org/doi/10.1145/3726302.3730066  
- arXiv: https://arxiv.org/pdf/2510.07720  

## Data preparation (MSR-VTT)

Our project currently targets MSR-VTT by default.

```bash
# Download and unzip dataset files
python scripts/download_msrvtt.py
# Preprocess video files into sampled frames + captions
python scripts/preprocess_dataset.py --config ./configs/msrvtt.toml
```

