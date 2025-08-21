"""
Link Prediction on Cora (GraphSAGE encoder + dot/MLP decoder).

Flow
----
1) RandomLinkSplit → train/val/test with negatives.
2) Train encoder+decoder using BCEWithLogits on train edges.
3) Evaluate ROC-AUC & Average Precision on val/test.

Run
---
# Dot-product decoder (strong + simple baseline)
python -m src.tasks.link_pred --encoder sage --decoder dot --epochs 200 --hid 64 --dropout 0.5 --lr 0.01

# MLP decoder (slightly stronger, more params)
python -m src.tasks.link_pred --encoder sage --decoder mlp --epochs 200

# Reuse your unsupervised embeddings (Node2Vec/DGI) with an MLP decoder
python -m src.tasks.link_pred --encoder precomputed --decoder mlp --epochs 200 --lr 0.01
"""

from __future__ import annotations
import os, argparse
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures, RandomLinkSplit

from src.models.sage import GraphSAGE
from src.models.decoders import decode_dot, MLPDecoder


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


def split_edges(data, val_ratio=0.05, test_ratio=0.10, neg_sampling_ratio=1.0):
    """
    Create link‑prediction splits with negatives using RandomLinkSplit.

    Returns:
        train_data, val_data, test_data : Data with fields:
        .edge_index           (train graph edges)
        .edge_label_index     edges to score (pos+neg)
        .edge_label           1 for pos, 0 for neg
    """
    splitter = RandomLinkSplit(
        num_val=val_ratio, num_test=test_ratio,
        is_undirected=True, split_labels=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=neg_sampling_ratio,
    )
    return splitter(data)


def _edge_targets(split):
    """
    Return (edge_label_index, edge_label) for any split object.
    Works with both styles that RandomLinkSplit might return.
    """
    if hasattr(split, "edge_label_index") and hasattr(split, "edge_label"):
        return split.edge_label_index, split.edge_label

    # Older / alternate API: pos/neg are separate on train split
    if hasattr(split, "pos_edge_label_index") and hasattr(split, "neg_edge_label_index"):
        pos = split.pos_edge_label_index
        neg = split.neg_edge_label_index
        edge_label_index = torch.cat([pos, neg], dim=1)
        edge_label = torch.cat(
            [torch.ones(pos.size(1)), torch.zeros(neg.size(1))],
            dim=0,
        ).to(pos.device)
        return edge_label_index, edge_label

    raise AttributeError("Split lacks edge_label_index/edge_label fields.")


def get_embeddings(encoder: str, data, *, hid=64, dropout=0.5, epochs=200, lr=0.01):
    """
    Compute node embeddings z \in R^{N x D}.

    encoder='sage'       → train GraphSAGE on `data.edge_index`
    encoder='precomputed'→ load from results/embeds/embeddings.pt

    Returns:
        model_or_none, z, dim
    """
    device = _device_auto()

    if encoder == 'precomputed':
        z = torch.load("results/embeds/embeddings.pt").detach().cpu()
        return None, z, z.size(1)
    x, ei = data.x.to(device), data.edge_index.to(device)
    model = GraphSAGE(in_dim=x.size(-1), hid=hid, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        z = model(x, ei)
        (z.norm(dim=1).mean() * 0.0).backward()
        opt.step()
    
    with torch.no_grad():
        z = model(x, ei).detach().cpu()
    return model, z, hid


def score_edges(z: torch.Tensor, edge_label_index: torch.Tensor, decoder, kind: str) -> torch.Tensor:
    """
    Compute logits for edges using a chosen decoder.

    Args:
        z : [N, D]
        edge_label_index : [2, M]
        decoder : callable or nn.Module
        kind : 'dot' or 'mlp'

    Returns:
        logits : [M]
    """
    if kind == "dot":
        return decode_dot(z, edge_label_index)
    elif kind == "mlp":
        return decoder(z, edge_label_index)
    else:
        raise ValueError("Unknown decoder kind.")


def eval_split(z: torch.Tensor, split, decoder, kind: str) -> Tuple[float, float]:
    """
    Evaluate AUC/AP on a split.
    """
    eli, y = _edge_targets(split)   
    logits = score_edges(z, eli, decoder, kind)
    y_true = y.cpu().numpy()
    y_score = logits.detach().cpu().numpy()
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    return (auc, ap)


def train_linkpred(
    encoder: str = "sage",
    decoder_kind: str = "dot",
    hid: int = 64,
    dropout: float = 0.5,
    epochs: int = 200,
    lr: float = 0.01,
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    seed: int = 432,
) -> Tuple[float, float]:
    """
    Full link‑prediction training & evaluation.

    Returns:
        test_auc, test_ap
    """
    # Reproducibility
    torch.manual_seed(seed)

    # Data
    ds = Planetoid(
        root="data/Cora", name='Cora', transform=NormalizeFeatures()
    )
    data = ds[0]
    train_data, val_data, test_data = split_edges(data, val_ratio, test_ratio, 1.0)

    device = _device_auto()
    model, z, dim = get_embeddings(encoder, train_data, hid=hid, dropout=dropout, epochs=epochs, lr=lr)
    decoder = MLPDecoder(dim).to(device) if decoder_kind == "mlp" else None

    if encoder == "sage":
        x, ei = train_data.x.to(device), train_data.edge_index.to(device)
        model = model.to(device)
        params = list(model.parameters()) + (list(decoder.parameters()) if decoder is not None else [])
        opt = torch.optim.Adam(params, lr=lr)
        crit = torch.nn.BCEWithLogitsLoss()
        for _ in range(1, epochs+1):
            model.train()
            if decoder is not None:
                decoder.train()
            opt.zero_grad(set_to_none=True)
            z = model(x, ei)
            eli, labels = _edge_targets(train_data)
            eli = eli.to(device)
            labels = labels.to(device).float()
            logits = score_edges(z, eli, decoder, decoder_kind)
            loss = crit(logits, labels)
            loss.backward()
            opt.step()
        with torch.no_grad():
            z = model(x, ei).detach().cpu()

    if decoder is not None:
        decoder = decoder.to("cpu")    
    val_auc, val_ap = eval_split(z, val_data, decoder, decoder_kind)
    test_auc, test_ap = eval_split(z, test_data, decoder, decoder_kind)
    print(f"[LinkPred] val AUC={val_auc:.3f} AP={val_ap:.3f} | test AUC={test_auc:.3f} AP={test_ap:.3f}")
    return test_auc, test_ap


def main():
    parser = argparse.ArgumentParser(description="Link Prediction on Cora")
    parser.add_argument("--encoder", choices=["sage", "precomputed"], default="sage")
    parser.add_argument("--decoder", choices=["dot", "mlp"], default="dot")
    parser.add_argument("--hid", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    test_auc, test_ap = train_linkpred(
        encoder=args.encoder, decoder_kind=args.decoder,
        hid=args.hid, dropout=args.dropout, epochs=args.epochs, lr=args.lr
    )
    os.makedirs("results/linkpred", exist_ok=True)
    with open("results/linkpred/metrics.txt", "w") as f:
        f.write(f"AUC={test_auc:.4f}, AP={test_ap:.4f}\n")
    print(f"[LinkPred] test AUC={test_auc:.4f} | test AP={test_ap:.4f}")


if __name__ == "__main__":
    main()
