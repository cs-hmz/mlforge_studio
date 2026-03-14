import numpy as np
from mlforge_studio.base_model import BaseModel

class MultipleLinearRegressionGD(BaseModel):
    def __init__(self, learning_rate=0.0001, epochs=3000):
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coef_ = None
        self.X = None
        self.y = None
        self.n = 0
        self.p_minus_one = 0 # number of features

    def gradient_descent(self):
        predictions = np.dot(self.X, self.coef_)
        errors = predictions - self.y
        gradient = (2/self.n) * np.dot(self.X.T, errors)

        return self.coef_ - self.learning_rate * gradient


    def fit(self, X, y):
        self.n, self.p_minus_one = X.shape
        self.X = np.ones((self.n, self.p_minus_one + 1))
        self.X[:, 1:] = np.array(X)
        self.y = np.array(y)
        self.coef_ = np.zeros(self.p_minus_one + 1)

        for _ in range(self.epochs):
            self.coef_ = self.gradient_descent()

        return self
    
    def predict(self, X):

        n_samples = X.shape[0]

        X_with_bias = np.ones((n_samples, self.p_minus_one + 1))
        X_with_bias[:, 1:] = X

        return np.dot(X_with_bias, self.coef_)