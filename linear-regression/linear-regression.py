from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class LinearRegressor:
    def __init__(self) -> None:
        self.learning_rate = 0.001
        self.weights = np.array([0 for i in range(0, 100)])
        self.b = 0
        self.losses = np.array([], dtype=np.float64)
    
    def fit(self, X_train:List[List[float]], y:List[float]) -> None:
        self.X = X
        self.y = y
        
        for epoch in range(4):
            for i in range(X_train.shape[0]):
                y = self.predict(X_train[i])
                loss = (y - self.y[i]) ** 2
                
                np.append(self.losses, loss)

                der_lw, der_lb = self.calculate_grads(X_train[i], y, self.y[i])

                self.weights = self.weights - self.learning_rate * der_lw
                self.b = self.b - self.learning_rate * der_lb
            
            if epoch % 1 == 0:
                print(loss)
                
    def calculate_grads(self, x:List[float], y:float, true_label:float) -> Tuple[float, float]:
        der_lw = 2 * x * (y - true_label)
        der_lb = 2 * (y - true_label)
        
        return der_lw, der_lb
    
    def predict(self, x:List[float]) -> float:
        return np.dot(x, self.weights) + self.b
    