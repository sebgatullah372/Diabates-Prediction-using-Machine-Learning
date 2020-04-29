# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 04:30:19 2019

@author: Arnob
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('pima-indians-diabetes.csv')
X= dataset.iloc[:,0:8]
y= dataset.iloc[:,8]
sc = StandardScaler()
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
random_forest_classifier = RandomForestClassifier(n_estimators = 25)
random_forest_classifier.fit(X_train,y_train)
predictions = random_forest_classifier.predict(X_test)
y_train_pred = random_forest_classifier.predict(X_train)
print("Test set: ", random_forest_classifier.score(X_test, y_test)*100)
print("Train set: ", random_forest_classifier.score(X_train, y_train)*100)
y_test_val=[]
for row,value in y_test.items():
    y_test_val.append(value)
for i in range(20):
	print('%s => %d (expected %d)' % (X_test[i].tolist(), predictions[i], y_test_val[i]))
acc_test_set=accuracy_score(y_test, predictions)
print("Test set accuracy: ", acc_test_set*100)

acc_train_set=accuracy_score(y_train, y_train_pred)
print("Train Set accuracy: ",acc_train_set*100)