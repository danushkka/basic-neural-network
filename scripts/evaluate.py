import torch
import torch.nn.functional as F

from src.makemore.data import build_vocab, load_data, make_splits
from src.makemore.model import forward
from src.makemore.utils import load_checkpoint


def evaluate(params, X, Y, bn_running_mean, bn_running_std):
    with torch.no_grad():
        logits, _, _ = forward(
            params,
            X,
            is_training=False,
            bn_running_mean=bn_running_mean,
            bn_running_std=bn_running_std,
        )
        return F.cross_entropy(logits, Y).item()


def main():
    words = load_data()
    stoi, itos = build_vocab(words)
    (Xtr, Ytr), (Xdev, Ydev), (Xtest, Ytest) = make_splits(stoi, 4)

    params, bn_mean, bn_std = load_checkpoint()
    with torch.no_grad():
        logits_tr, _, _ = forward(
            params,
            Xtr,
            is_training=False,
            bn_running_mean=bn_mean,
            bn_running_std=bn_std,
        )
        train_loss = F.cross_entropy(logits_tr, Ytr).item()

        logits_dev, _, _ = forward(
            params,
            Xdev,
            is_training=False,
            bn_running_mean=bn_mean,
            bn_running_std=bn_std,
        )
        val_loss = F.cross_entropy(logits_dev, Ydev).item()

    print(f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")


if __name__ == "__main__":
    main()
