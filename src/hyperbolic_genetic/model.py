# src/model.py
import torch
from torch import nn

class GeometryEmbedding(nn.Module):
    def __init__(self, num_nodes: int, dim: int, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.weight = nn.Parameter(1e-3 * torch.randn(num_nodes, dim, dtype=dtype))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.weight[idx]
