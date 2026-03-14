import numpy as np
from mlforge_studio.base_model import BaseModel

class SimpleLinearRegressionGD(BaseModel):
    def __init__(self, learning_rate=0.0001, epochs=3000):
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.b_now = 0.0
        self.w_now = 0.0

    def gradient_descent(self, x, y):
        w_gradient = 0
        b_gradient = 0
        n = len(x)
        for i in range(n):
            w_gradient += -(2/n) * (y[i] - (self.w_now * x[i] + self.b_now)) * x[i]
            b_gradient += -(2/n) * (y[i] - (self.w_now * x[i] + self.b_now))
        w = self.w_now - self.learning_rate * w_gradient
        b = self.b_now - self.learning_rate * b_gradient
        return w, b
    
    def fit(self, X, y):
        x = np.array(X)
        y = np.array(y)
        
        for _ in range(self.epochs):
            self.w_now, self.b_now = self.gradient_descent(x, y)
        return self
    
    def predict(self, value):
        return self.w_now * value + self.b_now