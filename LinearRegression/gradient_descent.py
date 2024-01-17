import numpy as np

class GradientDescent:
    def __init__(self, l, iter, regularization):
        self.l = l
        self.iterations = iter
        self.regularization = regularization
        self.w = None

    def fit(self, X, y):
        x_1 = np.insert(X, 0, 1, axis=1)
        n_s, n_f = x_1.shape
        self.m = n_s

        self.w = np.zeros(n_f)
        for i in range(self.iterations):
            y_pred = self.__predict(x_1)

            #if i % 100 == 0:
                #cost = self.__cost_function(y, y_pred)
                #print("cost function for the iteration {}----->{} :)".format(i, cost))
            
            dw = (1 / n_s) * (np.dot(x_1.T, (y_pred - y)) + self.regularization.derivation(self.w))
            self.w -= self.l * dw

    def predict(self, X):
        x_1 = np.insert(X, 0, 1, axis=1)
        return  np.dot(x_1, self.w)
    
    def __predict(self, X):
        return np.dot(X, self.w)
    
    def __cost_function(self, y, y_pred):
        return (1 / (2*self.m)) * np.sum(np.square(y_pred - y)) + self.regularization(self.w)
