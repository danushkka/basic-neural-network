import random

from micrograd.engine import Value


class Neuron:
    def __init__(self, nin):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # act = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        act = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        out = act.tanh()
        return out

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, nin, nout):
        # list of `nout` neurons of the input `nin`
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        # neurons of the input `x`
        outs = [n(x) for n in self.neurons]

        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # parameters for each neuron
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    # nouts = list of layer sizes (`nout`s), e.g. [4, 4, 1]
    def __init__(self, nin, nouts):
        sz = [nin] + nouts  # [3, 4, 4, 1]
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)  # reassign the value on each layer
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
