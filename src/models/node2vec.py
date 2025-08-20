"""
Thin wrapper to build a PyG Node2Vec model from edge_index.
"""
from __future__ import annotations
from typing import Any

import torch

from torch_geometric.nn import Node2Vec


# Node2Vec training
def build_node2vec(edge_index: torch.Tensor, **hparams: Any):
    """
    Create a torch_geometric.nn.models.Node2Vec instance.

    Args:
        edge_index : LongTensor [2, E]
            Graph edges (COO).
        **hparams :
            Node2Vec hyperparams (embedding_dim, walk_length, context_size, walks_per_node,
            num_negative_samples, p, q, sparse).

    Returns:
        model : torch.nn.Module
            A configured Node2Vec model.    
    """
    model = Node2Vec(edge_index=edge_index, **hparams)
    return model

