# project1 - Prediting stok price

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


from sklearn.metrics import r2_score,median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error ,mean_squared_error,mean_squared_log_error


from scipy.optimize import minimize
import statsmodels.tsa.api as smt
import statsmodels.api as sm

from tqdm import tqdm_notebook
from itertools import product



def mean_absolute_percentage_error(y_true,y_pred):
    return np.mean(np.abs((y_true-y_pred)/y_true))*100

import warnings 
warnings.filterwarnings('ignore')

data_file = 'stock_prices_sample.csv'
data_path = 'C:\\Users\\mehedee\\Documents\\Python Scripts\\tutorial\\online\\TS\\stock-prediction-master\\data\\'



data = pd.read_csv(data_path+data_file, index_col=['DATE'], parse_dates=['DATE'])
print(data.head())


data = data[data.TICKER != 'GEF']
data = data[data.TYPE != 'Intraday']


drop_cols = ['SPLIT_RATIO', 'EX_DIVIDEND', 'ADJ_FACTOR', 'ADJ_VOLUME', 'ADJ_CLOSE', 'ADJ_LOW', 'ADJ_HIGH', 'ADJ_OPEN', 'VOLUME', 'FREQUENCY', 'TYPE', 'FIGI']
data.drop(drop_cols,axis=1,inplace=True)
data.head()










