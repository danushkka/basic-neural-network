import itertools
import math

import torch

from makemore.viz import plot_counts


def main():
    words = load_data()
    print("Total names:", len(words))
    print("First ten:", words[:10])
    print("The shortest:", min(len(name) for name in words))
    print("The longest:", max(len(name) for name in words))

    stoi, itos = build_vocab(words)
    v = len(stoi)
    print("\nVocab size:", v)
    print("Chars:", "".join(itos[i] for i in range(v)))

    N = build_bigram(words, stoi, v)
    print("\nMatrix shape:", N.shape)
    print("Total bigrams:", N.sum().item())

    P = N.float()
    P = P / P.sum(dim=1, keepdim=True).clamp(min=1)

    gen = torch.Generator().manual_seed(42)
    print("\nGenerated names:")
    for _ in range(10):
        print(" ", sample_names(P, itos, gen))

    print("\nNLL loss:", nll_loss(words, stoi, P))

    plot_counts(N, itos)


def load_data(path="makemore/data/names.txt"):
    with open(path) as f:
        words = f.read().splitlines()
    return words


def build_vocab(words):
    chars = sorted(set("".join(words)))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}

    return stoi, itos


def build_bigram(words, stoi, vocab_size):
    N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)

    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in itertools.pairwise(chs):
            i, j = stoi[ch1], stoi[ch2]
            N[i, j] += 1

    return N


def sample_names(P, itos, gen):
    out = []
    ix = 0  # starting at "."

    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, generator=gen).item()
        if ix == 0:  # hit the end of the word (".")
            break
        out.append(itos[ix])

    return "".join(out)


"""

Summarizes total loss of all probs

By default we'd expect our model to give 100% accurate predictions, but it's probs are set to 
the basic quantity prob (e.g. letter `a` comes after `s` 3 times of 100, so its prob is 0.03).
Which means that at this exact combo the model is right at only log(0.03), which we add to the 
total likelihood of all the model's basic predictions and fnally find an average loss.

"""


def nll_loss(words, stoi, P):
    log_likelihood = 0.0
    n = 0
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in itertools.pairwise(chs):
            i, j = stoi[ch1], stoi[ch2]
            prob = P[i, j].item()  # the prob of that specific char sequence
            log_likelihood += math.log(prob)
            n += 1

    return -log_likelihood / n


if __name__ == "__main__":
    main()
