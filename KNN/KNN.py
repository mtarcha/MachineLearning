import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X.to_numpy()
        self.y_train = y.to_numpy()

    def predict(self, X):
        X_test = X.to_numpy()
        result = [self.__predict(x) for x in X_test]
        return np.array(result)
    
    def score(self, X, y):
        y_test = y.to_numpy()
        lenght = len(y_test)
        prediction = self.predict(X)
        accuracy = np.sum(prediction == y_test) / lenght

        return accuracy

    def __predict(self, x):
        distances = [np.linalg.norm(x-x_train) for x_train in self.X_train]
        k_indeces = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indeces]
        win_label = Counter(k_labels).most_common(1)
        return win_label[0][0]

    