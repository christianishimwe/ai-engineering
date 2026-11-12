# let's build numerical derivative
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)


def f(x):
    return x ** 2


for x in [-2, -1, 0, 1, 2]:
    numerical = numerical_derivative(f, x)
    analytical = 2 * x
    # print(f"x={x:2d} f'(x) numerical={numerical:.6f} analytical={analytical:.1f}")

# computing partial derivatices and gradients


def numerical_gradient(f, point, h=1e-7):
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)
        point_plus[i] += h
        point_minus[i] -= h
        gradient.append((f(point_plus) - f(point_minus)) / (2 * h))
    return gradient


def f_multi(point):
    x, y = point
    return x**2 + 3*x*y + y**2


grad = numerical_gradient(f_multi, [1.0, 2.0])
print(f"numerical gradient at (1, 2): {grad}")
print(f"analytical gradient at (1, 2): [{2**1 + 3*2}, {3 + 2 * 2}]")

# gradient descent to find the minimum of f(x) = x ^ 2
lr = 0.1  # learning rate
x = 5
for step in range(20):
    # 20 epochs, pick this random point(only one point in our case)
    # calculate the gradient at this point
    # update the point by moving int the opposite direction of the gradient
    grad = 2 * x
    x = x - lr * grad
    # print(f"step {step+1}: grad={grad}, x={x:.3f}")


# gradient descent on a 2d function
def f_2d(point):
    x, y = point
    return x**2 + y**2


lr = 0.1
point = [3, 4]
for step in range(20):
    # first calculate the gradient on this point
    grad = numerical_gradient(f_2d, point)
    # now update the point using the learning rate
    new_point = [p - lr * q for p, q in zip(point, grad)]
    point = new_point
    print(f"step {step+1}: grad={grad}, point={point}")
