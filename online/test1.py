import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd



x = [8.0,1,2.5,4,28.0]
x_with_nan = [8.0,1,2.5,math.nan ,4,28.0]
x

x_with_nan

math.isnan(np.nan),np.isnan(math.nan)
math.isnan(x_with_nan[3]),np.isnan(x_with_nan[3])


math.isnan(np.nan), np.isnan(math.nan)
math.isnan(x_with_nan[3]),np.isnan(x_with_nan[3])


y,y_with_nan = np.array(x) , np.array(x_with_nan)
z,z_with_nan = pd.Series(x) , pd.Series(x_with_nan)
y

y_with_nan

z 

z_with_nan


mean_ = sum(x) / len(x)
mean_


mean_ = statistics.mean(x)
mean_

mean_ = y.mean()
mean_



np.mean(x_with_nan)




x = [8.0,1,2.4,4,28.0]
w = [0.1,0.2,0.3,0.25,0.15]

wmean = sum (w[i] * x[i] for i in range(len(x)))/sum(w)
wmean


y,z,w = np.array(x),pd.Series(x),np.array(w)


wmean = np.average(y, weights=w)

