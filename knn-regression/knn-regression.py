import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from typing import List

class KnnRegressor:
    def __init__(self, k:int=5, n_features:int=100, noise:int=50) -> None:
        self.k = k
        self.n_features = n_features
        self.noise = noise
    
    def load_data(self) -> None:
        X, y = make_regression(n_features=self.n_features, noise=self.noise)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)
        
    def predict(self, x:List) -> float:
        train_array = np.array(list(zip(self.X_train, self.y_train)), dtype=object)
        predict_values = np.array([])
        
        if x.shape[1] != self.X_train.shape[1]:
            raise ValueError("value x is not suitable")
            
        for item in x:
            norm_list = [(np.linalg.norm(item - train_array[i][0]), train_array[i][1]) for i in range(train_array.shape[0])]
            norm_list = sorted(norm_list, key=lambda x: x[0])
            
            predict_values = np.append(predict_values, np.mean(norm_list[:self.k]))
        
        return predict_values