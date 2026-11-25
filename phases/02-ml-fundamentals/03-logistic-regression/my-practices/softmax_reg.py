import math


class SoftmaxRegression:
    def __init__(self, n_features, n_classes, learning_rate=0.01):
        self.weights = [[0.0 for _ in range(n_features)]
                        for _ in range(n_classes)]
        self.bias = [0.0 for _ in range(n_classes)]
        self.n_classes = n_classes
        self.n_features = n_features
        self.lr = learning_rate
        pass

    def softmax(self, scores):
        max_score = max(scores)
        exp_scores = [math.exp(score - max_score) for score in scores]
        tot = sum(exp_scores)
        return [e / tot for e in exp_scores]

    def predict_proba(self, x):
        # predcit probability of x for each class
        # predict x for each class
        scores_x = [sum(xi * wi for xi, wi in zip(x, self.weights[k])) + self.bias[k]
                    for k in range(self.n_classes)]
        return self.softmax(scores_x)

    def predict(self, x):
        probabilities = self.predict_proba(x)
        return probabilities.index(max(probabilities))

    def accuracy(self, X, y):
        n = len(X)
        counts = 0
        for i in range(n):
            x = X[i]
            pred_label = self.predict(x)
            if pred_label == y[i]:
                counts += 1
        return counts / n

    def fit(self, X, y, epochs=1000, print_every=200):
        n_data = len(X)
        for epoch in range(epochs):
            # initialize the gradients map
            grad_features = [
                [0.0 for _ in range(self.n_features)] for _ in range(self.n_classes)]
            grad_biases = [0.0 for _ in range(self.n_classes)]
            # now for every data point
            for i in range(n_data):
                proba_x = self.predict_proba(X[i])
                for k in range(self.n_classes):
                    y_here = 1.0 if y[i] == k else 0.0
                    for j in range(self.n_features):
                        grad_features[k][j] += (proba_x[k] - y_here) * X[i][j]
                    # compute the bias term
                    grad_biases[k] += (proba_x[k] - y_here)

            for k in range(self.n_classes):
                for j in range(self.n_features):
                    grad_features[k][j] /= n_data
                grad_biases[k] /= n_data

            # now update them
            for k in range(self.n_classes):
                for j in range(self.n_features):
                    self.weights[k][j] -= self.lr * grad_features[k][j]
                self.bias[k] -= self.lr * grad_biases[k]
