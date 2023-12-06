import numpy as np

class LinearRegressionNE:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        ones = np.ones(X.shape[0])
        X_1 = np.column_stack([ones, X])

        XTX = X_1.T.dot(X_1)
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X_1.T).dot(y)

        # (XT*X)^-1*XT*y
        self.weights = w

    def predict(self, X):
        ones = np.ones(X.shape[0])
        X_1 = np.column_stack([ones, X])
        return np.dot(X_1, self.weights)