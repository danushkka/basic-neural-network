import torch
import torch.nn.functional as F

from src.makemore.data import build_vocab, load_data, make_splits
from src.makemore.model import forward, init_params
from src.makemore.utils import save_checkpoint


def main():
    # Initialize the config with all hyperparams
    config = {
        "block": 4,
        "emb": 10,
        "hidden": 200,
        "lr": "0.1/0.01",
        "steps": 200_000,
        "batch_norm": True,
    }

    words = load_data()
    stoi, itos = build_vocab(words)
    V = len(stoi)
    (Xtr, Ytr), (Xdev, Ydev), _ = make_splits(stoi, block_size=config["block"])

    print(f"Config: {config}")

    params = init_params(
        V, hidden=config["hidden"], emb_dim=config["emb"], block_size=config["block"]
    )

    print("Training...")

    g = torch.Generator().manual_seed(42)
    bn_mean = torch.zeros((1, config["hidden"]))
    bn_std = torch.ones((1, config["hidden"]))

    for step in range(config["steps"] + 1):
        ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
        xb, yb = Xtr[ix], Ytr[ix]

        logits, bn_mean, bn_std = forward(
            params,
            xb,
            is_training=True,
            bn_running_mean=bn_mean,
            bn_running_std=bn_std,
        )
        loss = F.cross_entropy(logits, yb)

        for p in params:
            p.grad = None
        loss.backward()

        lr = 0.1 if step < config["steps"] * 0.5 else 0.01
        for p in params:
            p.data -= lr * p.grad

        if step % (config["steps"] // 10) == 0:
            print(f"step {step:6d}   loss {loss.item():.4f}")

    # Save everything
    save_checkpoint(params, bn_mean, bn_std)


if __name__ == "__main__":
    main()
