"""
DGI components: a small 2‑layer GCN encoder that produces embeddings (not logits),
and a helper to wrap it with PyG's DeepGraphInfomax objective.

Notes
-----
- The encoder returns node embeddings R^{N x H} (no final softmax).
- Keep it simple: GCNConv -> ReLU -> Dropout -> GCNConv.
- This file only defines *models*. The training loop should live in tasks/.
"""

from __future__ import annotations
from typing import Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import DeepGraphInfomax

class SmallGCNEncoder(nn.Module):
    """
    Minimal 2-layer GCN encoder used inside DGI.

    Args:
        in_dim : int
            Input feature dimension (#features per node).
        hid : int, default 64
            Embedding dimension for the hidden (and output) layer.
            The encoder outputs z in R^{N x hid}.
        dropout : float, default 0.5
            Dropout rate applied after the first layer.
    """
    def __init__(self, in_dim: int, hid: int=64, dropout: float=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute node embeddings.

        Args:
            x:  [N, in_dim]
            edge_index:  [2, E] (COO)
            returns:  [N, hid] (embeddings) 
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        return x

def build_dgi(encoder: nn.Module, hidden_channels: int):
    """
    Wrap an encoder into a Deep Graph Infomax (DGI) objective.

    Args:
        encoder: nn.Module
        Node encoder that maps (x, edge_index) -> z in R^{N x hidden_channels}.
        hidden_channels: int
        Dimensionality of z: must match the encoder's output size.

    Returns:
        dgi: nn.Module
            A torch_geometric.nn.models.DeepGraphInfomax module.
    """
    def summary(z: torch.Tensor, * args) -> torch.Tensor:
        # standard DGI summary: sigmoid of mean-pooled embeddings
        return torch.sigmoid(z.mean(dim=0))
    
    def corruption(x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randperm(x.size(0))
        return x[idx], edge_index
    
    
    dgi = DeepGraphInfomax(hidden_channels=hidden_channels, encoder=encoder, summary=summary, corruption=corruption)
    return dgi