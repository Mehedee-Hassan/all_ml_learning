import keras
import numpy
import pandas as pd
import sklearn
import nltk

# we are infront of a classification problem

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('social_network_ads.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values



# splitting the dataset into the training set and test set

from sklean.model_selection import train_test_split
X_train,X_test,y_train,y_test ~ train_test_split(X,y,test_size=0.25,random_state=0)


#feature scaling

from sklearn.preprocessing import StandardScaler

sc = StrandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# fitting classifier to the training set
# create classifier



# predict class
y_pred 















