import torch
import torch.nn.functional as F
from mlp.data import build_vocab, load_data, make_splits
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
    stoi, _itos = build_vocab(words)
    V = len(stoi)
    (Xtr, Ytr), (Xdev, _Ydev), _ = make_splits(stoi)

    print(f"Vocab: {V}, Train: {len(Xtr)}, Val: {len(Xdev)}")

    params = init_params(V)
    print(f"Params: {sum(p.nelement() for p in params)}")

    print("\nTraining...")
    final_loss = train(params, Xtr, Ytr)
    print(f"Train batch loss: {final_loss:.4f}")

    save_params(params)


if __name__ == "__main__":
    main()
