# now let's use the closed form to compute the linear regression and see how better this is
class LinearRegressionNormal:
    def __init__(self):
        self.w = 0.0
        self.b = 0.0

    def predict(self, X):
        return [self.w * x + self.b for x in X]

    def fit(self, X, y):

        pass

    def r_squared(self, X, y):
        pass
