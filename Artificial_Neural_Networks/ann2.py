# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:36:29 2020

@author: mehedee
"""

# we are infront of a classificaiton problem
# we are going to 


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 



dataset  = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,  13].values


# categorical variable 

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer


labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])


ct = ColumnTransformer([("Geography", OneHotEncoder(), [1,2])], remainder = 'passthrough')
X = ct.fit_transform(X).toarray()


onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(x).toarray()





from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25, random_state =True)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
x_train = sc.fit_transform(X_train)
x_test = sc.transform(X_test)



y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confustion_matrix(y_test,y_pred)


# part2 now build ann
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim=5, kernel_initializer="uniform", activation ="relu",input_dim=10))
classifier.add(Dense(output_dim=5, kernel_initializer="uniform", activation ="relu"))
classifier.add(Dense(output_dim=1, kernel_initializer="uniform", activation ="sigmoid"))



y_pred = classifier.predict(X_test)
