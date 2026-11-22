class RidgeRegression:
    def __init__(self, n_features, learning_rate=0.01, alpha=1.0):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.alpha = alpha
        self.loss_history = []
        self.n_features = n_features

    def predict_single(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def fit(self, X, y, epochs=1000, print_every=200):
        # for each epoch,
        # compute the loss, the gradient, and update weights and bias
        n = len(y)
        for epoch in range(epochs):
            # predicti
            pred = self.predict(X)
            # compute the loss
            loss = sum((pi - yi) ** 2 for pi, yi in zip(pred, y)) / n + self.alpha * \
                sum(w ** 2 for w in self.weights)
            self.loss_history.append(loss)
            # find the gradients
            grad_w = list(((2 / n) * sum(x[i] * (pi - yi) for x, pi, yi in zip(
                X, pred, y))) + 2 * self.alpha * self.weights[i] for i in range(self.n_features))
            grad_b = 2 / n * sum(pi - yi for pi, yi in zip(pred, y))
            # update the weights and bias
            self.weights = list(w - self.lr * g for w,
                                g in zip(self.weights, grad_w))
            self.bias = self.bias - self.lr * grad_b
            if epoch % print_every == 0:
                print(f"Epoch number {epoch}, loss: {loss}")
