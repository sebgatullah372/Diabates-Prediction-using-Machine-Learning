import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
dataset = pd.read_csv('pima-indians-diabetes.csv')
# creating input features and target variables
X= dataset.iloc[:,0:8]
y= dataset.iloc[:,8]

#standardizing the input feature
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=8))
#Second  Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))

#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

#Fitting the data to the training dataset
classifier.fit(X_train,y_train, batch_size=10, epochs=100)

_, eval_model=classifier.evaluate(X_train, y_train)


print(eval_model*100)
y_test_val=[]
for row,value in y_test.items():
    y_test_val.append(value)
predictions = classifier.predict_classes(X_test)
y_train_pred = classifier.predict_classes(X_train)
# summarize the first 5 cases
for i in range(20):
	print('%s => %d (expected %d)' % (X_test[i].tolist(), predictions[i], y_test_val[i]))
acc_test_set=accuracy_score(y_test, predictions)
print("Test Set accuracy: ",acc_test_set*100)
acc_train_set=accuracy_score(y_train, y_train_pred)
print("Train Set accuracy: ",acc_train_set*100)

