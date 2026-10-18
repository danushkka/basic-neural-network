import random

import torch


def load_data(path="../makemore/data/names.txt"):
    with open(path) as f:
        return f.read().splitlines()


def build_vocab(words):
    chars = sorted(set("".join(words)))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}

    return stoi, itos


def build_dataset(words, stoi, block_size=4):
    X, Y = [], []  # X == context, Y == expected prediction
    for w in words:
        context = [0] * block_size  # set initial context as "..."
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # update the context with the next ch

    return torch.tensor(X), torch.tensor(Y)


def make_splits(stoi, seed=42):
    words = load_data()
    random.seed(42)
    random.shuffle(words)
    n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
    Xtr, Ytr = build_dataset(words[:n1], stoi)  # first 80%
    Xdev, Ydev = build_dataset(words[n1:n2], stoi)  # 10% (between 80% and 90%)
    Xtest, Ytest = build_dataset(words[n2:], stoi)  # last 10%

    return (Xtr, Ytr), (Xdev, Ydev), (Xtest, Ytest)
