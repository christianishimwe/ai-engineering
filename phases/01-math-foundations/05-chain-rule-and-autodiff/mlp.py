from layer import Layer


class MLP:
    def __init__(self, sizes):
        self.layers = [Layer(sizes[i], sizes[i + 1])
                       for i in range(len(sizes) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
