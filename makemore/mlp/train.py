import torch
import torch.nn.functional as F
from mlp.data import build_vocab, load_data, make_splits
from mlp.logging_utils import log_experiment
from mlp.mlp import forward, init_params, save_params


def train(params, Xtr, Ytr, steps=200_000, batch=32, seed=42):
    g = torch.Generator().manual_seed(seed)
    for step in range(steps + 1):
        ix = torch.randint(0, Xtr.shape[0], (batch,), generator=g)
        xb, yb = Xtr[ix], Ytr[ix]

        logits = forward(params, xb)
        loss = F.cross_entropy(logits, yb)

        for p in params:
            p.grad = None
        loss.backward()

        lr = 0.1 if step < steps * 0.5 else 0.01
        for p in params:
            p.data -= lr * p.grad

        if step % 20_000 == 0:
            print(f"step {step:6d}   loss {loss.item():.4f}")

    return loss.item()


def main():
    words = load_data()
    stoi, itos = build_vocab(words)
    V = len(stoi)
    (Xtr, Ytr), (Xdev, Ydev), _ = make_splits(stoi)

    # Initialize the config with all hyperparams
    config = {
        "block": 3,
        "emb": 4,
        "hidden": 300,
        "lr": 0.05,
        "steps": 200_000,
    }

    print(f"Config: {config}")

    params = init_params(
        V,
        hidden=config["hidden"],
        emb_dim=config["emb"],
    )

    print("Training...")
    train(params, Xtr, Ytr, steps=config["steps"])

    # Claculate the losses
    with torch.no_grad():
        train_loss = F.cross_entropy(forward(params, Xtr), Ytr).item()
        val_loss = F.cross_entropy(forward(params, Xdev), Ydev).item()

    print(f"Train: {train_loss:.4f}   Val: {val_loss:.4f}")

    # Save the model and log the experiment
    save_params(params)
    log_experiment(config, train_loss, val_loss)


if __name__ == "__main__":
    main()
