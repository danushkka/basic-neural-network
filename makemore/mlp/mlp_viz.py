import os

import matplotlib

matplotlib.use("Agg")  # headless backend — no GUI needed
import matplotlib.pyplot as plt


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
    os.makedirs("graphs", exist_ok=True)
    out = "graphs/embeddings.svg"
    plt.savefig(out, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
