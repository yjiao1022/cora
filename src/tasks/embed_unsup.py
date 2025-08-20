"""
Unsupervised node embedding pipeline (Node2Vec or DGI) + linear probe on Cora.

Deliverables produced in project root (or adjust paths via CLI):
- results/embeds/embeddings.pt        # tensor [N, D]
- results/embeds/probe_accuracy.txt    # single float accuracy on test
- results/plots/tsne.png               # 2D t-SNE plot colored by labels
- results/plots/umap.png               # 2D UMAP plot (optional, requires umap-learn)

Usage (examples)
---------------
# Node2Vec with defaults
python -m src.tasks.embed_unsup --method node2vec --epochs 100

# Node2Vec with custom hyperparams
python -m src.tasks.embed_unsup --method node2vec --embedding_dim 128 \
    --walk_length 20 --context_size 10 --walks_per_node 10 --neg_samples 1 --lr 0.01

# DGI (Deep Graph Infomax) with a tiny GCN encoder
python -m src.tasks.embed_unsup --method dgi --epochs 300 --lr 0.001 --hid 64 --dropout 0.5
"""

from __future__ import annotations
import os, argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from sklearn.manifold import TSNE

from src.models.node2vec import build_node2vec

try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

def _device_auto() -> torch.device:
    """
    Automatically selects the device to run the model on.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def train_node2vec(
    edge_index: torch.Tensor,
    *,
    embedding_dim: int = 128,
    walk_length: int = 20,
    context_size: int = 10,
    walks_per_node: int = 10,
    num_negative_samples: int = 1,
    p: float = 1.0,
    q: float = 1.0,
    epochs: int = 100,
    lr: float = 0.01,
    sparse: bool = False,
) -> torch.Tensor:
    """
    Train Node2Vec (random‑walk skip‑gram) to learn node embeddings.

    Args:
        edge_index : LongTensor [2, E]
            COO edges of the graph.
        embedding_dim : int
            Size of the learned embedding per node.
        walk_length, context_size, walks_per_node, num_negative_samples, p, q :
            Standard Node2Vec hyperparameters.
        epochs : int
            Training epochs for the skip‑gram objective.
        lr : float
            Learning rate (SparseAdam recommended if sparse=True).
        sparse : bool
            Use sparse gradients for the embedding matrix (memory‑efficient).

    Returns:
        emb : Tensor [N, embedding_dim]
            The learned embeddings (detached on CPU).

    TODO(you):
        - Instantiate torch_geometric.nn.models.Node2Vec with the hyperparameters above.
        - Choose optimizer: SparseAdam if sparse=True, else Adam.
        - Training loop: for epoch in 1..epochs:
              loss = model.loss()
              backward + step
        - Return model.embedding.weight.detach().cpu()
    """
    device = torch.device("cpu") # <- MPS backend does not support sparse gradient yet

    model = build_node2vec(
        edge_index, embedding_dim=embedding_dim, walk_length=walk_length, context_size=context_size,
        walks_per_node=walks_per_node, num_negative_samples=num_negative_samples, p=p, q=q, sparse=sparse
    ).to(device)

    # Loader samples random walks for training
    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)

    opt = torch.optim.SparseAdam if sparse else torch.optim.Adam
    opt = opt(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            opt.zero_grad(set_to_none=True)
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if epoch % 10 ==0:
            print(f"Epoch {epoch:03d}, Loss: {total_loss:.4f}")

    emb = model.embedding.weight.detach().cpu()
    return emb


def linear_probe(emb: torch.Tensor, labels: torch.Tensor, train_mask: torch.Tensor,
                 val_mask: torch.Tensor, test_mask: torch.Tensor,
                 *, lr=0.1, wd=0.0, epochs=2000, seed=423) -> Tuple[float, float]:
    """
    Train a single linear layer on frozen embeddings to predict the class labels.
    """
    torch.manual_seed(seed)
    D = emb.size()[1]
    C = int(labels.max().item()) + 1
    clf = nn.Linear(D, C)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state = -1.0, {k: v.detach().clone() for k,v in clf.state_dict().items()}
    for _ in range(1, epochs + 1):
        clf.train()
        opt.zero_grad(set_to_none=True)
        logits = clf(emb)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        opt.step()
        with torch.no_grad():
            logits = clf(emb)
            val = (logits[val_mask].argmax(1) == labels[val_mask]).float().mean().item()
        if val > best_val: 
            best_val = val
            best_state = {k: v.detach().clone() for k,v in clf.state_dict().items()}

    clf.load_state_dict(best_state)
    clf.eval()
    with torch.no_grad():
        logits = clf(emb)
        test = (logits[test_mask].argmax(1) == labels[test_mask]).float().mean().item()
    return best_val, test

def _scatter_2d(Z2, y, title, save_path=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,4))
    plt.scatter(Z2[:,0], Z2[:,1], c=y, s=10)
    plt.title(title); plt.axis("off")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)        
        plt.tight_layout(); plt.savefig(save_path, dpi=160); plt.close()
    else:   # show inline in notebook
        plt.tight_layout()
        plt.show()

def plot_tsne(emb: torch.Tensor, labels: torch.Tensor, save_path: str=None) -> None:
    """
    t‑SNE → 2D scatter.
    """
    Z2 = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca').fit_transform(emb.numpy())
    _scatter_2d(Z2, labels.cpu().numpy(), "t-SNE", save_path)



def plot_umap(emb: torch.Tensor, labels: torch.Tensor, save_path: str=None) -> None:
    """
    UMAP → 2D scatter.
    """
    Z2 = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1).fit_transform(emb.numpy())
    _scatter_2d(Z2, labels.cpu().numpy(), "UMAP", save_path)


def main():
    parser = argparse.ArgumentParser(description="Unsupervised embeddings + linear probe (Cora)")
    parser.add_argument("--method", choices=["node2vec", "dgi"], default="node2vec")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    # Node2Vec
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--walk_length", type=int, default=20)
    parser.add_argument("--context_size", type=int, default=10)
    parser.add_argument("--walks_per_node", type=int, default=10)
    parser.add_argument("--neg_samples", type=int, default=1)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--n2v_lr", type=float, default=0.01)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    ds = Planetoid(root="data/Cora", name="Cora", transform=NormalizeFeatures()); data = ds[0]

    if args.method == "node2vec":
        emb = train_node2vec(
            data.edge_index,
            embedding_dim=args.embedding_dim, walk_length=args.walk_length,
            context_size=args.context_size, walks_per_node=args.walks_per_node,
            num_negative_samples=args.neg_samples, p=args.p, q=args.q,
            epochs=args.epochs, lr=args.n2v_lr, sparse=True
        )

    os.makedirs("results/embeds", exist_ok=True)
    torch.save(emb, "results/embeds/embeddings.pt")
    print(f"Saved embeddings to results/embeds/embeddings.pt  | shape={tuple(emb.shape)}")

    val_acc, test_acc = linear_probe(
        emb, data.y, data.train_mask, data.val_mask, data.test_mask, lr=0.1, wd=0.0, epochs=2000, seed=args.seed
    )
    with open("results/embeds/probe_accuracy.txt", "w") as f:
        f.write(f"{test_acc:.4f}\n")
    print(f"Linear probe test accuracy: {test_acc:.4f}")

    plot_tsne(emb, data.y.cpu(), "results/plots/tsne.png")
    plot_umap(emb, data.y.cpu(), "results/plots/umap.png")

if __name__ == "__main__":
    main()