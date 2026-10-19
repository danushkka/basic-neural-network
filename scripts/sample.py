import torch
import torch.nn.functional as F
from mlp.data import build_vocab, load_data
from mlp.mlp import forward, load_params


def sample_name(params, itos, block_size=4, temperature=1.0, generator=None):
    context = [0] * block_size
    out = []

    while True:
        xb = torch.tensor([context])  # (1, block_size)
        logits = forward(params, xb)  # (1, V)
        logits = logits / temperature  # temperature scaling
        probs = F.softmax(logits, dim=1)  # (1, V)

        ix = torch.multinomial(probs, num_samples=1, generator=generator).item()
        context = context[1:] + [ix]

        if ix == 0:  # hit end token
            break
        out.append(itos[ix])

    return "".join(out)


def main():
    words = load_data()
    _stoi, itos = build_vocab(words)
    params = load_params()

    g = torch.Generator().manual_seed(42)

    print("--- Temperature = 1.0 (default) ---")
    for _ in range(10):
        print(" ", sample_name(params, itos, generator=g))

    print("\n--- Temperature = 0.5 (conservative) ---")
    for _ in range(10):
        print(" ", sample_name(params, itos, temperature=0.5, generator=g))

    print("\n--- Temperature = 1.5 (creative) ---")
    for _ in range(10):
        print(" ", sample_name(params, itos, temperature=1.5, generator=g))


if __name__ == "__main__":
    main()
