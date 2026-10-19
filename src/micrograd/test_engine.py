from micrograd.engine import Value
from micrograd.nn import MLP
from micrograd.viz import visualize


def main():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    e = a * b
    d = e + c
    f = Value(-2.0)
    L = d * f

    print("a.grad =", a.grad)
    print("b.grad =", b.grad)
    print("c.grad =", c.grad)
    print("f.grad =", f.grad)
    print("L.grad =", L.grad)

    print("-----------------")

    n = MLP(3, [4, 4, 1])
    x = [Value(2.0), Value(4.6), Value(-3.1)]
    out = n(x)
    print("MLP output:", out)

    visualize(out, name="chain")

    # out.backward()
    print("input grads:", [xi.grad for xi in x])
    print("parameter count:", len(n.params()))
    print("first param grad:", n.params()[0].grad)


if __name__ == "__main__":
    main()
