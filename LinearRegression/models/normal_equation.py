import numpy as np

class LinearRegressionNE:
    def __init__(self, regularization=0.1):
        self.weights = None
        self.r = regularization

    def fit(self, X, y):
        ones = np.ones(X.shape[0])
        X_1 = np.column_stack([ones, X])

        XTX = X_1.T.dot(X_1)

        # to avoid singular matrix problem - add small number to diagonal
        XTX = XTX + self.r * np.eye(XTX.shape[0])

        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X_1.T).dot(y)

        self.weights = w

    def predict(self, X):
        ones = np.ones(X.shape[0])
        X_1 = np.column_stack([ones, X])
        return np.dot(X_1, self.weights)