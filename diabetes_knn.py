# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 03:25:28 2019

@author: Arnob
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
dataset = pd.read_csv('pima-indians-diabetes.csv')
X= dataset.iloc[:,0:8]
y= dataset.iloc[:,8]
sc = StandardScaler()
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train, y_train)
    print(k)
    print("Test Set accuracy: ",knn_classifier.score(X_test, y_test)*100)
    print("Train Set accuracy: ",knn_classifier.score(X_train, y_train)*100)
