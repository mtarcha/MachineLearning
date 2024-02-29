import numpy as np


class LogisticRegression:
    """
    My implementation from scratch of LR
    """

    def __init__(self, lr=0.001, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m_s, n_f = X.shape
        self.weights = np.zeros(n_f)
        self.bias = 0

        # gradient descent
        for i in range(self.iterations):
            y_pred = self.predict_probability(X)
            dw = np.dot(X.T, (y_pred - y))
            db = np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_pred = self.predict_probability(X)
        y_pred_cls = [1 if i >= 0.5 else 0 for i in y_pred]
        return y_pred_cls
    
    
    def predict_probability(self, X):
        lin_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(lin_model)
        return y_pred


    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

