import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    """
    2-layer Graph Convolutional Network (GCN) model for node classification.

    Args:
        x: [N, F],  Node features w/ N nodes and F features.
        edge_index: [2, E] Edge indices w/ E edges.

    Returns:
        logits: [N, C], Logits for each node w/ C classes.
    """
    def __init__(self, in_dim: int, hid: int=16, out_dim: int=7, dropout: float=0.5):
        super().__init__()
        # Hint: two GCNConv layers + Dropout in between.
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x
    
