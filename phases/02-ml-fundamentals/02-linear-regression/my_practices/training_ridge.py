# let's generate data first
import random
from ridge_regression import RidgeRegression
random.seed(42)


N = 100
X_multi = []
y_multi = []
for _ in range(N):
    size = random.uniform(500, 3000)
    bedrooms = random.randint(1, 5)
    age = random.randint(0, 50)
    # math to get the price
    price = 50 * size + 10000 * bedrooms - 1000 * \
        age + 50000 + random.gauss(0, 20000)  # and add some noise too
    X_multi.append([size, bedrooms, age])
    y_multi.append(price)

# standardizeing the features


def standardize(X):
    n_features = len(X[0])
    means = [sum(x[i] / len(X) for x in X) for i in range(n_features)]
    N = len(X)
    stds = []
    for j in range(n_features):
        variance = sum((X[i][j] - means[j]) ** 2 for i in range(N)) / N
        stds.append(variance ** 0.5)
    X_scaled = [[(x[j] - means[j]) / stds[j] if stds[j] >
                 0 else 0 for j in range(n_features)] for x in X]
    return X_scaled, means, stds


# let's standardize y
y_mean = sum(y_multi) / len(y_multi)
y_std = (sum([(y - y_mean) ** 2 for y in y_multi]) / len(y_multi)) ** 0.5
y_scaled = [(y - y_mean) / y_std for y in y_multi]

# let's stadnardize our features (X)
X_scaled, means, stds = standardize(X_multi)
# print(X_scaled, means, stds)
# now let's train
n_features = len(X_multi[0])
model = RidgeRegression(n_features=n_features, learning_rate=0.01, alpha=0.1)
model.fit(X_scaled, y_scaled)
