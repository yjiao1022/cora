"""
GraphSAGE encoder (2-layer) for link prediction.
Produces node embeddings z in R^{N x H}.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    """
    2-layer GraphSAGE encoder.

    Args:
        in_dim : int
            Input feature dimension.
        hid : int
            Hidden / output embedding size.
        dropout : float
            Dropout after the first layer.

    Returns:
        z : Tensor [N, hid]
            Node embeddings.
    """
    def __init__(self, in_dim: int, hid: int = 64, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid)
        self.conv2 = SAGEConv(hid, hid)
        self.dropout = nn.Dropout(dropout)
        self.act = F.relu

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x 