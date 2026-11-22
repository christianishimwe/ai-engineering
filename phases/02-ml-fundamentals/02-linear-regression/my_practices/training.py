# let's gernerate data
import random
import math
from linear_regression import LinearRegression

random.seed(42)

TRUE_W = 3.0
TRUE_B = 7.0
N_SAMPLES = 100
X = [random.uniform(0, 10) for _ in range(N_SAMPLES)]
y = [TRUE_W * x + TRUE_B + random.gauss(0, 2.0)
     for x in X]  # adding a little noise
# normal linear regression, now train and predict
print(f"Generated data: y = {TRUE_W} * x + {TRUE_B} + noise")
print(
    f"First 5 samples: {[(round(X[i], 2), round(y[i], 2)) for i in range(5)]}")
print("====== Training Linear Regression Model(gradient descent)")
model = LinearRegression(learning_rate=0.005)
model.fit(X, y, epochs=1000, print_every=200)
print(f"\n Learned y = {model.w:.4f}x + {model.b:.4f}")
print(f"True: y = {TRUE_W}x + {TRUE_B}")
print(f"R-squared: {model.r_squared(X, y):.4f}")
