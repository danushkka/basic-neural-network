import itertools

import torch
import torch.nn.functional as F


def main():
    # ----- INITIALIZE ----- #
    words = load_data()
    stoi, itos = build_vocab(words)
    V = len(stoi)

    xs, ys = build_dataset(words, stoi)

    # ----- TRAIN ----- #

    gen = torch.Generator().manual_seed(42)
    W = torch.randn((V, V), generator=gen, requires_grad=True)

    lr = 50.0
    for step in range(200):
        xenc = F.one_hot(xs, num_classes=V).float()
        logits = xenc @ W

        counts = logits.exp()
        probs = counts / counts.sum(dim=1, keepdim=True)

        loss = -probs[torch.arange(len(ys)), ys].log().mean()

        W.grad = None
        loss.backward()

        W.data -= lr * W.grad

    # ----- GENERATE ----- #

    # g = torch.Generator().manual_seed(228_1488)
    print("\nGenerated names")

    for _ in range(10):
        out = []
        ix = 0
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=V).float()
            logits = xenc @ W
            counts = logits.exp()
            p = counts / counts.sum(dim=1, keepdim=True)
            ix = torch.multinomial(p, num_samples=1, generator=gen).item()
            if ix == 0:
                break

            out.append(itos[ix])

        print(" ", "".join(out))

    print(f"\nLoss: {loss}")


def load_data(path="makemore/data/names.txt"):
    with open(path) as f:
        return f.read().splitlines()


def build_vocab(words):
    chars = sorted(set("".join(words)))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0  # "." is of 0-th index
    itos = {i: s for s, i in stoi.items()}

    return stoi, itos


def build_dataset(words, stoi):
    xs, ys = [], []
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in itertools.pairwise(chs):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])

    return torch.tensor(xs), torch.tensor(ys)


if __name__ == "__main__":
    main()
