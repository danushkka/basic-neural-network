import os

import torch


def save_checkpoint(
    params, bn_running_mean, bn_running_std, path="src/makemore/checkpoints/model.pt"
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "params": [p.detach() for p in params],
            "bn_running_mean": bn_running_mean,
            "bn_running_std": bn_running_std,
        },
        path,
    )
    print(f"Saved checkpoint → {path}")


def load_checkpoint(path="src/makemore/checkpoints/model.pt"):
    ckpt = torch.load(path)
    params = ckpt["params"]
    for p in params:
        p.requires_grad_(True)
    return params, ckpt["bn_running_mean"], ckpt["bn_running_std"]
