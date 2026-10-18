import torch


def init_params(V, hidden=300, emb_dim=8, seed=42):
    g = torch.Generator().manual_seed(seed)
    C = torch.randn((V, emb_dim), generator=g, requires_grad=True)
    W1 = torch.randn((4 * emb_dim, hidden), generator=g, requires_grad=True)
    b1 = torch.randn(hidden, generator=g, requires_grad=True)
    W2 = torch.randn((hidden, V), generator=g, requires_grad=True)
    b2 = torch.randn(V, generator=g, requires_grad=True)
    return [C, W1, b1, W2, b2]


def forward(params, xb):
    C, W1, b1, W2, b2 = params
    emb = C[xb]  # (B, 3, 2)
    emb_flat = emb.view(emb.shape[0], -1)  # (B, 6)
    h = torch.tanh(emb_flat @ W1 + b1)  # (B, hidden)
    return h @ W2 + b2  # (B, V)


def save_params(params, path="mlp/checkpoints/model.pt"):
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save([p.detach() for p in params], path)
    print(f"Saved model → {path}")


def load_params(path="mlp/checkpoints/model.pt"):
    params = torch.load(path)
    for p in params:
        p.requires_grad_(True)
    return params
