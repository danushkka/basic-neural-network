import torch
import torch.nn.functional as F
from mlp.data import build_vocab, load_data, make_splits
from mlp.mlp import forward, load_params
from mlp.mlp_viz import depict


def evaluate(params, X, Y):
    with torch.no_grad():
        logits = forward(params, X)
        return F.cross_entropy(logits, Y).item()


def main():
    words = load_data()
    stoi, itos = build_vocab(words)
    params = load_params()

    (Xtr, Ytr), (Xdev, Ydev), (Xtest, Ytest) = make_splits(stoi)

    print(f"Train:    {evaluate(params, Xtr, Ytr):.4f}")
    print(f"Validate: {evaluate(params, Xdev, Ydev):.4f}")
    print(f"Test:     {evaluate(params, Xtest, Ytest):.4f}")

    # Plot embeddings
    depict(params[0], itos)


if __name__ == "__main__":
    main()
