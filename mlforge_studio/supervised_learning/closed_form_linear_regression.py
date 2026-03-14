import numpy as np
from mlforge_studio.base_model import BaseModel

class LinearRegressionClosedForm(BaseModel):
    def __init__(self):
        super().__init__()
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        X = np.c_[np.ones((X.shape[0], 1)), X]

        A = np.linalg.inv(X.T @ X) @ X.T @ y
        self.intercept_ = A[0]
        self.coef_ = A[1:]
        return self
    
    def predict(self, X):
        X = np.array(X)

        return X @ self.coef_ + self.intercept_