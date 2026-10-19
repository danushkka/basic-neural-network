from pathlib import Path  # noqa: I001
from graphviz import Digraph

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — works in WSL without a display
import matplotlib.pyplot as plt

# Resolve the graphs folder relative to THIS file
GRAPHS_DIR = Path(__file__).parent / "graphs"


def trace(root):
    """Return all nodes and edges in the graph rooted at `root`."""
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root, format="svg", rankdir="LR"):
    """Visualize a computational graph."""
    assert rankdir in ["LR", "TB"]
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={"rankdir": rankdir})

    for n in nodes:
        uid = str(id(n))
        dot.node(
            name=uid,
            label=f"{{ data {n.data:.4f} | grad {n.grad:.4f} }}",
            shape="record",
        )
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


def _render(root, name, auto_open=False):
    """Render a single graph to micrograd/graphs/<name>.svg"""
    GRAPHS_DIR.mkdir(exist_ok=True)
    out_path = GRAPHS_DIR / name
    draw_dot(root).render(filename=str(out_path), view=auto_open, cleanup=True)
    return out_path


def visualize(root, name="graph", auto_open=False):
    """Render before + after backward as two SVGs in micrograd/graphs/."""
    before = _render(root, f"{name}_before", auto_open=auto_open)
    root.backward()
    after = _render(root, f"{name}_after", auto_open=auto_open)
    print(f"Saved: {before}.svg")
    print(f"Saved: {after}.svg")


def plot_loss(history, name="loss_curve", log_y=False):
    """
    Plot a loss curve and save it into micrograd/graphs/.

    history: list of floats (loss per step)
    name:    output filename (without extension)
    log_y:   if True, use log scale on y-axis (useful when loss spans orders of magnitude)
    """
    GRAPHS_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(history)), history, linewidth=2)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)

    if log_y:
        ax.set_yscale("log")

    fig.tight_layout()
    out_path = GRAPHS_DIR / f"{name}.svg"
    fig.savefig(out_path, format="svg")
    plt.close(fig)  # free memory — important in a loop
    return out_path
