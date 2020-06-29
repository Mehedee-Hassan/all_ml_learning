import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

path ="C:\\Users\\mehedee\\Documents\\Python Scripts\\tutorial\\Artificial_Neural_Networks\\kaggle\\housedata\\"

%config InlineBackend.figure_format = 'retina'
%matplotlib inline




train = pd.read_csv(path+"house_train.csv")
test = pd.read_csv(path+"house_test.csv")

train.head()



all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']
                      ))


all_data


#- data procession





matplotlib.rcParams['figure.figsize'] = (12.0,6.0)

prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)" :np.log1p(train["SalePrice"])})
prices.hist()



train["SalePrice"] = np.log1p(train["SalePrice"])

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x : skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])



all_data = pd.get_dummies(all_data)

all_data = all_data.fillna(all_data.mean())


X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice 

train.head()










from sklearn.linear_model import Ridge,RidgeCV,ElasticNet,LassoCV,LassoLarsCV
from sklearn.model_selection import cross_val_score



def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model,  X_train,y,scoring="neg_mean_square_error ",cv= 5))
    reurn (rmse)
    
model_ridge=Ridge()

alphas = [ 0.05,0.1,0.3,1,3,5,10,15,30,50,75]

cv_ridge = [rmse_cv(Ridge(lapha = alpha)).mean() for aplpha in alphas]


cv_ridge = pd.Series(cv_ridge,index = alphas)
cv_ridge.plot(title="Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")







































