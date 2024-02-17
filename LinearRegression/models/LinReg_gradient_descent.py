import numpy as np

class LinearRegressionGD:
    def __init__(self, l=0.001, iter=1000, penalty=1):
        self.l = l
        self.n_i = iter
        self.penalty = penalty
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_s, n_f = X.shape
        self.m = n_s
        self.n = n_f

        self.bias = 0
        self.weights = np.zeros(n_f)

        for i in range(self.n_i):
            y_pred = self.predict(X)

            dw = (1 / n_s) * (np.dot(X.T, (y_pred - y)))
            db = (1 / n_s) * np.sum(y_pred - y)

            self.weights -= self.l * dw
            self.bias -= self.l * db

    def predict(self, X):
        return self.bias + np.dot(X, self.weights)
