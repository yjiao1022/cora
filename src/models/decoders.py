"""
Edge decoders for link prediction:
- Dot product: <z_u, z_v>
- MLP over [z_u || z_v || |z_u - z_v|]
"""

from __future__ import annotations
import torch
import torch.nn as nn

def decode_dot(z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Args:
        z : [N, D] node embeddings
        edge_index : [2, M] (u,v) pairs to score

    Returns:
        logits : [M] raw scores
    """
    u, v = edge_index
    return (z[u] * z[v]).sum(dim=1)

class MLPDecoder(nn.Module):
    """
    MLP on concatenated pair features [z_u, z_v, |z_u - z_v|] → logit.
    """
    def __init__(self, dim: int, hid: int = 128):
        super().__init__()
    
        self.mlp = nn.Sequential(
            nn.Linear(3*dim, hid),
            nn.ReLU(),
            nn.Linear(hid, 1),
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        u, v = edge_index
        zu, zv = z[u], z[v]
        feat = torch.cat([zu, zv, (zu-zv).abs()], dim=1)
        return self.mlp(feat).squeeze(-1)