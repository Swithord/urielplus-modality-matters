import torch

DATA_PATH = "data/"
OUTPUT_DIR = "results/"


TRAINING_PARAMS = {
    "dim": 10,
    "geometry": "hyperboloid",
    "lr": 0.3,
    "epochs": 1000,
    "batch_size": 1024,
    "K": 80,
    "seed": 2025,
    "hyperboloid_max_rie_grad_norm": 15.0,
    "hyperboloid_max_spatial_norm_points": 1e11,
    "warmup_epochs": 300,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "verbose": True,
}
