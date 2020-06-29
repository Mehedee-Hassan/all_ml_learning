
# from now on 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('C:\\Users\\mehedee\\Documents\\Python Scripts\\tutorial\\Artificial_Neural_Networks\\ML_DS\\Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values


"""


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=0 )

# feature scaling
# most of the algorithm we are going to use have feature scaling
# so we dont need to use feature scaling 

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

"""

# create your regressor



# fitting the regression model to the dataset
from sklearn.preprocessing import PolynomialFeatures


y_pred = regressor.predict(6.5)

plt.scatter(X,y,color='red')
plt.plot(X  ,regressor.predict(X), color='orange')
plt.title('Truth or bluff (Regression model)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()






# for higher resolution and smoother graph 
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X,y,color='red')
plt.plot(X_grid  ,regressor.predict(X_grid), color='orange')
plt.title('Truth or bluff (Regression model)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()






"""from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=0 )
"""

"""
from sklearn.linear_model import LinearRegression

lin_reg= LinearRegression()
lin_reg.fit(X,y)

# fitting polynomial regression to the dataset

from sklearn.preprocessing import PolynomialFeatures

poly_reg =PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X) 


lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)



plt.scatter(X,y , color= 'red')
plt.plot(X, lin_reg.predict(X),color='blue')
plt.title("Truth or bluff (Linear Regression)")
plt.xlabel('positon level')
plt.ylabel('Salary')
plt.show()



plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(X),color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#predicting new result with linear regression


"""