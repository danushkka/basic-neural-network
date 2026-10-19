import torch


def init_params(V, hidden=200, emb_dim=10, block_size=4, seed=42):
    g = torch.Generator().manual_seed(seed)
    std = (5 / 3) / ((emb_dim * block_size) ** 0.5)

    C = torch.randn((V, emb_dim), generator=g, requires_grad=True)

    W1 = (
        torch.randn((block_size * emb_dim, hidden), generator=g) * std
    ).requires_grad_(True)

    gamma = torch.ones((1, hidden), requires_grad=True)
    beta = torch.zeros((1, hidden), requires_grad=True)

    W2 = (torch.randn((hidden, V), generator=g) * 0.01).requires_grad_(True)
    b2 = (torch.randn(V, generator=g) * 0.01).requires_grad_(True)

    return [C, W1, gamma, beta, W2, b2]


def forward(
    params,
    xb,
    is_training=None,
    bn_running_mean=None,
    bn_running_std=None,
    bn_momentum=0.1,
):
    C, W1, gamma, beta, W2, b2 = params

    emb = C[xb]
    emb_flat = emb.view(emb.shape[0], -1)

    h_preact = emb_flat @ W1

    if is_training:
        bn_mei = h_preact.mean(dim=0, keepdim=True)
        bn_std = h_preact.std(dim=0, keepdim=True)

        h_norm = (h_preact - bn_mei) / (bn_std + 1e-5)

        with torch.no_grad():
            bn_running_mean = (1 - bn_momentum) * bn_running_mean + bn_momentum * bn_mei
            bn_running_std = (1 - bn_momentum) * bn_running_std + bn_momentum * bn_std
    else:
        h_norm = (h_preact - bn_running_mean) / (bn_running_std + 1e-5)

    h_norm = gamma * h_norm + beta

    h = torch.tanh(h_norm)
    logits = h @ W2 + b2

    return logits, bn_running_mean, bn_running_std
