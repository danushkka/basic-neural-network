import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_counts(N, itos, name="bigram_counts"):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(N, cmap="Blues")
    for i in range(len(itos)):
        for j in range(len(itos)):
            chstr = itos[i] + itos[j]
            ax.text(j, i, chstr, ha="center", va="bottom", color="gray", fontsize=7)
            ax.text(
                j, i, N[i, j].item(), ha="center", va="top", color="gray", fontsize=6
            )
    ax.axis("off")
    fig.savefig(f"makemore/graphs/{name}.svg", format="svg")
    plt.close(fig)
