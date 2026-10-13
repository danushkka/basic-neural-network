from micrograd.nn import MLP
from micrograd.viz import plot_loss


def main():
    # Tiny toy dataset (binary-ish classification)
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    n = MLP(3, [8, 10, 12, 14, 16, 14, 12, 10, 8, 1])

    loss_history = []
    lr = 0.05
    steps = 100

    for step in range(steps):
        # ---- Forward ----
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        # ---- Zero grads ----
        for p in n.parameters():
            p.grad = 0.0

        # ---- Backward ----
        loss.backward()

        # ---- Update ----
        for p in n.parameters():
            p.data -= lr * p.grad

        loss_history.append(loss.data)

        if step % 10 == 0 or step == steps - 1:
            print(f"step {step:3d}  loss {loss.data:.4f}")

    # ---- Plot ----
    plot_loss(loss_history, name="loss_curve")
    print("\nSaved loss curve to micrograd/graphs/loss_curve.svg")

    # ---- Final predictions ----
    print("\nFinal predictions:")
    for x, y in zip(xs, ys):
        pred = n(x)
        print(f"  input {x}  target {y:+.1f}  pred {pred.data:+.4f}")


if __name__ == "__main__":
    main()
