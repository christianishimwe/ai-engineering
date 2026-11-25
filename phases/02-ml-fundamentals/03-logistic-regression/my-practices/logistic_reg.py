import math


def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


class LogisticRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.loss_history = []

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return sigmoid(z)

    def predict(self, x, threshold=0.5):
        return 1 if self.predict_proba(x) >= threshold else 0

    def compute_loss(self, X, y):
        n = len(y)
        total = 0.0
        for i in range(len(X)):
            p = self.predict_proba(X[i])
            p = max(1e-15, min(1 - 1e-15, p))
            total += y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p)
        return -total / n

    def fit(self, X, y, epochs=1000, print_every=200):
        for epoch in epochs:
            # compute the loss
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)
            # now cokmpute the gradient
            n_features = len(X[0])
            n_data = len(X)
            grad_w = [0.0 for _ in range(n_features)]
            for i in range(n_features):
                # let's udpate wi
                for j in range(n_data):
                    grad_w[i] += (self.predict_proba(X[j]) - y[j]) * X[j][i]
                grad_w[i] /= n_data
            self.bias -= self.lr * \
                (1 / n_data) * sum(self.predict_proba(xi) - yi for xi, yi in zip(X, y))
            # update the grads
            for i in range(n_features):
                self.weights -= self.lr * grad_w[i]
            # update the bias
            if epoch % print_every == 0:
                # print the loss
                print(f"loss at epoch {epoch} is {loss}")
