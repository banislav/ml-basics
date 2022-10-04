import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression


class KnnRegressor:
    def __init__(self, k:int=5):
        self.k = k
    
    def load_data(self) -> None:
        self.dataset = make_regression(n_features=1, noise=50)
    
    def predict(self, x:float) -> float:
        test_x, test_y = self.dataset
        test_array = list(zip(test_x, test_y))
        norm_list = [np.linalg.norm(test_array[i]) for i in range(len(test_array))]
        norm_list = sorted([float(x) for x in norm_list])
        
        predict_value = np.mean(norm_list[:5])
        return predict_value
    
    
regressor = KnnRegressor(k=5)
regressor.load_data()
regressor.predict(3)