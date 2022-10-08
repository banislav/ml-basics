from typing import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

class KnnRegressor:
    def __init__(self, k_neighbors:int=5) -> None:
        self.k_neighbors = k_neighbors
    
    def load_data(self, X:List[List[float]], y:List[float]) -> None:
        self.X, self.y = X, y
        
    def predict(self, x:List[float]) -> List[float]:
        predict_values = np.array([])
        
        if x.shape[1] != self.X.shape[1]:
            raise ValueError("value x is not suitable")
            
        for item in x:
            norm_list = np.array(np.linalg.norm(self.X - item, axis=1))
            neighbors_indexes = norm_list.argsort()[:self.k_neighbors]
            
            predict_values = np.append(predict_values, np.mean(self.y[neighbors_indexes]))
        
        return predict_values
    
def predict_test(X_train, X_test, y_train):
    #Make instance of my knn regressor
    regressor = KnnRegressor(k_neighbors=5)
    regressor.load_data(X=X_train, y=y_train)
    
    #Make instance of sklearn knn regressor
    neighbors = KNeighborsRegressor(n_neighbors=5)
    neighbors.fit(X=X_train, y=y_train)
    
    return regressor.predict(X_test) == neighbors.predict(X_test)

if __name__ == '__main__':
    X, y = make_regression(n_features=100, noise=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    print(predict_test(X_train, X_test, y_train))