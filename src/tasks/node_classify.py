import os
from typing import Any, Dict, Tuple

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from src.models.gcn import GCN
from src.models.gat import GAT

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
    
def _accuracy(logits, labels) -> float:
    """Compute top‑1 accuracy given class logits and integer labels."""
    return (logits.argmax(dim=1) == labels).sum().item() / len(labels)

def train_model(
    model_cls: type, 
    hparams: Dict[str, Any], 
    *,
    epochs: int=200, lr: float=0.01, wd: float=5e-4,
    seed: int=422, dataset_root: str='data/cora'
):
    """
    Train a GNN on Cora using a generic training loop.

    This function is model‑agnostic: pass any class `model_cls` whose constructor
    accepts `in_dim`, `out_dim`, and arbitrary keyword hyperparameters (**hparams),
    and whose `forward(x, edge_index)` returns logits of shape [N, C].

    Args:
        model_cls : type
            The model class to instantiate (e.g., `GCN`, `GAT`, ...).
            Must have signature like `model_cls(in_dim=..., out_dim=..., **hparams)`.
        hparams: Dict[str, Any]
            Extra hyperparameters forwarded to `model_cls` (e.g., `{"hid": 16, "dropout": 0.5}`).
        epochs: Number of training epochs.
        lr: Learning rate for the optimizer.
        wd: Weight decay for the optimizer.
        seed: Random seed for reproducibility.
        dataset_root: Path to the dataset.

    Returns:
        best_val : float
            Best validation accuracy observed during training.
        final_test : float
            Test accuracy after restoring best‑val checkpoint.
        model : torch.nn.Module
            The trained model (weights set to best‑val state).
        data : torch_geometric.data.Data
            The Cora graph object (features, labels, masks, edge_index) moved to device.
        ds : torch_geometric.data.InMemoryDataset
            The loaded Planetoid dataset object (for metadata like `num_classes`).

    Notes:
        - Uses `NormalizeFeatures()` (row‑norm per node) which is the standard setting for Cora.
        - Tracks the best validation accuracy and restores the corresponding weights before
        computing the final test accuracy.
        - Designed to work on Apple Silicon (MPS), CUDA, or CPU transparently.
    """
    # Reproducibility
    torch.manual_seed(seed)

    # Data
    ds = Planetoid(
        root=dataset_root, name='Cora', transform=NormalizeFeatures()
    )
    data = ds[0]

    device = _device_auto()
    data = data.to(device)
    # (MPS tip) keep features in float32
    data.x = data.x.to(torch.float32)

    # Model
    model = model_cls(in_dim=data.num_node_features, out_dim=ds.num_classes, **hparams).to(device)

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Masks
    tr, va, te = data.train_mask, data.val_mask, data.test_mask

    # Training loop with best‑val tracking
    best_val = -1 
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    
    for epoch in range(1, epochs + 1):

        # Train
        model.train()
        opt.zero_grad(set_to_none=True)

        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[tr], data.y[tr])
        loss.backward()
        opt.step()

        # Eval (no grad)
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_acc = _accuracy(logits[va], data.y[va])
            test_acc = _accuracy(logits[te], data.y[te])

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | loss {loss.item():.4f} | val {val_acc:.3f} | test {test_acc:.3f}")

    # Restore best state and final test
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        final_test = _accuracy(logits[te], data.y[te])
    print(f"Best val: {best_val:.3f} | Final test: {final_test:.3f}")
    return best_val, final_test, model, data, ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN models on Cora")
    parser.add_argument(
        "--model", type=str, default="gcn",
        choices=["gcn", "gat"],
        help="Which model to train."
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=None, help="Override dropout (0.5) if set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hid", type=int, default=None, help="Number of hidden dimensions.")

    # GAT-specific
    parser.add_argument("--heads", type=int, default=None, help="Number of attention heads (GAT only).")

    args = parser.parse_args()

    # Pick model + default hyperparams
    if args.model == "gcn":
        hparams = {"hid": 16, "dropout": args.dropout or 0.5}
        model_cls = GCN
    elif args.model == "gat":
        hparams = {"hid": 8, "heads": 8, "dropout": args.dropout or 0.6}
        model_cls = GAT

    train_model(
        model_cls,
        hparams,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.wd,
        seed=args.seed,
    )

# python -m src.tasks.node_classify --model gcn
# python -m src.tasks.node_classify --model gat --hid=8 --heads 8 --epochs 400 --lr 0.005 --dropout 0.8 --wd 5e-4 --seed 4020
