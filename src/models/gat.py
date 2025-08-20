import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    """
    2-layer Graph Attention Network (GAT) model for node classification.

    Args:
        x: [N, F],  Node features w/ N nodes and F features.
        edge_index: [2, E] Edge indices w/ E edges.
        in_dim: Number of input features.
        hid: Number of hidden units in the GAT layer.
        out_dim: Number of output classes.
        dropout: Dropout rate for the GAT layers.

    Returns:
        logits: [N, C], Logits for each node w/ C classes.
    """
    def __init__(self, in_dim: int, hid: int=8, heads: int=8, out_dim: int=7, dropout: float=0.5):
        super().__init__()

        self.g1 = GATConv(in_dim, hid, heads=heads, dropout=dropout)
        self.g2 = GATConv(hid*heads, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()

    def forward(self, x, edge_index):
        x = self.g1(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)
        x = self.g2(x, edge_index)
        return x

