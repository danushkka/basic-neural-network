import os
import random

import matplotlib

matplotlib.use("Agg")  # headless backend — no GUI needed
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def main():
    # ----- INITIALIZE ----- #

    words = load_data()
    stoi, itos = build_vocab(words)
    V = len(stoi)

    random.seed(42)
    random.shuffle(words)
    n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
    Xtr, Ytr = build_dataset(words[:n1], stoi)  # first 80%
    Xdev, Ydev = build_dataset(words[n1:n2], stoi)  # 10% (between 80% and 90%)
    Xtest, _Ytest = build_dataset(words[n2:], stoi)  # last 10%

    print(f"Vocab size: {V}")
    print(f"Training examples: {len(Xtr)}")
    print(f"Validation examples: {len(Xdev)}")
    print(f"Test examples: {len(Xtest)}")

    gen = torch.Generator().manual_seed(42)

    C = torch.randn((V, 2), generator=gen, requires_grad=True)
    W1 = torch.randn((6, 300), generator=gen, requires_grad=True)
    b1 = torch.randn(300, generator=gen, requires_grad=True)
    W2 = torch.randn((300, V), generator=gen, requires_grad=True)
    b2 = torch.randn(V, generator=gen, requires_grad=True)

    params = [C, W1, b1, W2, b2]

    print(f"Total params: {sum(p.nelement() for p in params)}")

    # ----- TRAIN ----- #

    print("\nTraining...")
    steps = 96_000
    for step in range(steps + 1):
        # make a batch of random 32 chs
        ix = torch.randint(0, Xtr.shape[0], (32,), generator=gen)
        xb, yb = Xtr[ix], Ytr[ix]

        # Forward pass
        emb = C[xb]  # (32, 3, 2)
        emb_flat = emb.view(emb.shape[0], -1)  # (32, 6)
        h = torch.tanh(emb_flat @ W1 + b1)  # (32, 100)

        logits = h @ W2 + b2  # (32, V)
        loss = F.cross_entropy(
            logits, yb
        )  # Internally: log-softmax + target logit + mean + negate

        # Backward pass
        for p in params:
            p.grad = None
        loss.backward()

        # Update
        lr = 0.05 if step < steps * 0.5 else 0.01
        for p in params:
            p.data -= lr * p.grad

        if step % (steps / 10) == 0:
            print(f"step {step:5d}   loss {loss.item():.4f}")

    print("\nTraining finished")
    print(f"Training loss: {loss:.4f}")

    print("\nValidation")
    emb = C[Xdev]
    emb_flat = emb.view(emb.shape[0], -1)
    h = torch.tanh(emb_flat @ W1 + b1)

    logits = h @ W2 + b2
    loss_dev = F.cross_entropy(logits, Ydev)

    print(f"Validation loss: {loss_dev:.4f}")

    print(f"\nDifference: {(loss_dev - loss):.4f}")

    # Depict
    depict(C, itos)


def load_data(path="makemore/data/names.txt"):
    with open(path) as f:
        return f.read().splitlines()


def build_vocab(words):
    chars = sorted(set("".join(words)))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}

    return stoi, itos


def build_dataset(words, stoi, block_size=3):
    X, Y = [], []  # X == context, Y == expected prediction
    for w in words:
        context = [0] * block_size  # set initial context as "..."
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # update the context with the next ch

    return torch.tensor(X), torch.tensor(Y)


def depict(C, itos):
    plt.figure(figsize=(8, 8))
    plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
    for i in range(C.shape[0]):
        plt.text(
            C[i, 0].item(),
            C[i, 1].item(),
            itos[i],
            ha="center",
            va="center",
            color="white",
        )
    plt.grid(True, which="minor")  # fix: Karpathy's "minor" needs a dict or this
    plt.title("Learned character embeddings")

    # ---- Save to makemore/graphs/ ----
    os.makedirs("makemore/graphs", exist_ok=True)
    out = "makemore/graphs/embeddings.svg"
    plt.savefig(out, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
