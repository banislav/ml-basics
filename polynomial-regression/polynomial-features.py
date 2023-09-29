from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class PolynomialRegrFeatures:
    def __init__(self, degree:int=1, include_bias:bool=True) -> None:
        self.degree = degree
        self.include_bias = include_bias
       
    def fit(self, X:List[List[float]]) -> None:
        self.X = X
        self.bias = 0 if self.include_bias else None
        
    def transform(self) -> np.array:
        X_transformed = np.array([[], []], dtype=float)
        
        for x in self.X:
            X_transformed = np.append(X_transformed, [x**i for i in range(self.degree+1)])
            
        return X_transformed.reshape(-1, self.degree+1)
    
    def fit_transform(self, X:List[List[float]]) -> np.array:
        self.fit(X)
        return self.transform()