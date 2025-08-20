import math
from typing import List, Tuple, Optional

import torch
import matplotlib.pyplot as plt

def get_attention(model, x, edge_index, layer=0):
    """
    Extract raw attention weights from a specific GAT layer in the model.

    Args:
        model (nn.Module): The trained GAT model.
        x (Tensor): Node feature matrix of shape [N, F].
        edge_index (LongTensor): Graph connectivity in COO format with shape [2, E].
        layer (int, optional): Index of the GAT layer to inspect. 
            - For a 2-layer GAT, use `0` for the first layer and `1` for the second.

    Returns:
        ei_used : torch.LongTensor
            Edge indices [2, E_used] as used internally by the probed layer
            (can include added self-loops, etc.).
        alpha_mean : torch.Tensor
            Attention weights [E_used], averaged over heads if multi-head.


    Notes:
        - You must call the GAT layer with `return_attention_weights=True`.
        - Example:
            _, (ei, alpha) = model.g1(x, edge_index, return_attention_weights=True)
        - These α values represent how much a source node contributes to its destination node’s update.
    """
    model.eval()
    with torch.no_grad():
        if layer == 0:
            _, (ei_used, alpha) = model.g1(x, edge_index, return_attention_weights=True)
        elif layer == 1:
            h = model.act(model.g1(x, edge_index))
            h = model.dropout(h)
            _, (ei_used, alpha) = model.g2(h, edge_index, return_attention_weights=True)

        else:
            raise ValueError("Layer must be 0 or 1 for a 2-layer GAT.")
        
        if alpha.dim() == 1:
            alpha_mean = alpha
        else: 
            if alpha.shape[0] == ei_used.size(1):   # [E, H]
                alpha_mean = alpha.mean(dim=1)
            elif alpha.shape[1] == ei_used.size(1):   # [H, E]
                alpha_mean = alpha.mean(dim=0)
            else:
                raise RuntimeError("Unexposed alpha shape vs. edges.")
    return (ei_used, alpha_mean)


def top_k_neighbors(ei: torch.Tensor, alpha: torch.Tensor, node_id: int, k: int = 5) -> List[Tuple[int, float]]:
    """
    Get top-k incoming neighbors of `node_id` by attention weight.

    Args:
        ei : torch.LongTensor
            Edge indices [2, E]; ei[0] = src, ei[1] = dst.
        alpha : torch.Tensor
            Attention weights [E] (already averaged across heads).
        node_id : int
            Destination node whose influential neighbors we want.
        k : int, default=5
            Number of neighbors to return.

    Returns:
        neighbors : List[(int, float)]
            List of (src_node, weight) sorted by weight desc, length ≤ k.
    """
    dst = ei[1]
    mask = (dst == node_id)
    if mask.sum() == 0:
        return []
    src_nodes = ei[0][mask]
    weights = alpha[mask]

    # Sort by weight (desc)
    order = torch.argsort(weights, descending=True)
    src_top = src_nodes[order][:k].tolist()
    w_top = weights[order][:k].tolist()
    return list(zip(src_top, [float(w) for w in w_top]))


def plot_attention_subgraph(
    G: Optional[object],
    center: int,
    nbrs: List[Tuple[int, float]],
    save_path: Optional[str] = None,
):
    """
    Plot a simple "star" subgraph: center node and its top-α neighbors.

    Args:
        G : networkx.Graph or None
            If provided, used for labels or future extensions (not required).
        center : int
            The center node id.
        nbrs : List[(neighbor_id, weight)]
            Top neighbors with attention weights.
        save_path : str or None
            If provided, save the figure; otherwise show it inline.

    Behavior
        - Places the center at (0, 0), neighbors on a circle.
        - Edge width scales with attention weight.
        - Annotates each edge with weight (rounded to 2 decimals).
    """
    # simple geometry
    r = 1.0
    cx, cy = 0.0, 0.0
    n = max(1, len(nbrs))
    angles = [2 * math.pi * i / n for i in range(n)]

    plt.figure(figsize=(4, 4))
    # center
    plt.scatter([cx], [cy], s=200, label=f"center {center}")
    plt.text(cx, cy + 0.07, str(center), ha="center", va="bottom", fontsize=9)

    # neighbors
    for (i, (nb, w)) in enumerate(nbrs):
        x = cx + r * math.cos(angles[i])
        y = cy + r * math.sin(angles[i])
        plt.scatter([x], [y], s=100)
        plt.text(x, y - 0.07, str(nb), ha="center", va="top", fontsize=8)

        # edge line (width scaled by weight)
        lw = max(0.5, 4.0 * float(w))
        plt.plot([cx, x], [cy, y], linewidth=lw)

        # annotate weight mid-edge
        midx, midy = (cx + x) / 2, (cy + y) / 2
        plt.text(midx, midy, f"{w:.2f}", fontsize=8, ha="center", va="center")

    plt.axis("off")
    plt.title(f"Attention to node {center}")
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

