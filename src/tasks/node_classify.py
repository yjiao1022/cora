import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from src.models.gcn import GCN

def device_auto():
    """
    Automatically selects the device to run the model on.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).sum().item() / len(labels)

def train_gcn(
    hidden=16, lr=0.01, wd=5e-4, dropout=0.5, 
    epochs=200, seed=422, dataset_root='data/cora'
):
    """
    Trains a GCN model on the Cora dataset.

    Args:
        hidden: Number of hidden units in the GCN.
        lr: Learning rate for the optimizer.
        wd: Weight decay for the optimizer.
        dropout: Dropout rate for the GCN layers.
        epochs: Number of training epochs.
        seed: Random seed for reproducibility.
        dataset_root: Path to the dataset.

    Returns:
        model: The trained GCN model.
    """
    torch.manual_seed(seed)

    ds = Planetoid(
        root=dataset_root, name='Cora', transform=NormalizeFeatures()
    )
    data = ds[0]

    device = device_auto()
    data = data.to(device)

    model = GCN(
        in_dim=data.num_node_features, hid=hidden, 
        out_dim=ds.num_classes, dropout=dropout
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    best_val, best_state = 0, None
    for epoch in range(1, epochs + 1):

        # Train
        model.train()
        opt.zero_grad(set_to_none=True)

        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask])

        loss.backward()
        opt.step()

        # Eval (no grad)
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_acc = accuracy(logits[val_mask], data.y[val_mask])
            test_acc = accuracy(logits[test_mask], data.y[test_mask])

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | loss {loss.item():.4f} | val {val_acc:.3f} | test {test_acc:.3f}")

    # Restore best state and final test
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        final_test_acc = accuracy(logits[test_mask], data.y[test_mask])
    print(f"Best val: {best_val:.3f} | Final test: {final_test_acc:.3f}")
    return final_test_acc

if __name__ == "__main__":
    train_gcn()


