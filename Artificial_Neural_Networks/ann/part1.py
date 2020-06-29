
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
path = "C:\\Users\\mehedee\\Documents\\Python Scripts\\tutorial\\Artificial_Neural_Networks\\data\\"
dataset = pd.read_csv(path+"datasets_1276_2288_Churn_Modelling.csv")

X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values



from sklearn.model_selection import train_test_split


X_train , X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25 )


# Encoding categorical data


from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder =OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform()
