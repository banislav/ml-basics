from typing import List
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class KnnClassifier:
    def __init__(self, k_neighbors:int=5) -> None:
        self.k_neighbors = k_neighbors
        
    def fit(self, X, y) -> None:
        self.X, self.y = X, y
    
    def predict(self, X:List[float]) -> int:
        norm_list = np.array([])
        categories_list = np.array([], dtype=int)
        
        if X.shape[1] != self.X.shape[1]:
            raise ValueError("Unexpected argument shape")
        
        for x in X:
            norm_list = np.linalg.norm(self.X - x, axis=1)
            neighbors_index = norm_list.argsort()[:self.k_neighbors]
            neighbors_list = self.y[neighbors_index]
            
            values, counts = np.unique(neighbors_list, return_counts=True)
            category = values[counts.argsort()[-1]]
            categories_list = np.append(categories_list, category)
            
        return categories_list
 

if __name__ == '__main__':
    X, y = make_classification()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    my_classifier = KnnClassifier()
    my_classifier.fit(X_train, y_train)
    my_result = my_classifier.predict(X_test)
    
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    result = classifier.predict(X_test)
    
    print(my_result == result)