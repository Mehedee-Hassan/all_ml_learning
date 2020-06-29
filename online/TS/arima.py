#SARIMA

import pandas as pd
import numpy as np

#%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

data_path = 'C:\\Users\\mehedee\\Documents\\data\\course\\UDEMY_TSA_FINAL\\Data\\'
df1 = pd.read_csv(data_path+'airline_passengers.csv',index_col='Month',parse_dates=True)
# index_col='Date',parse_dates=True
df1.index.freq ='MS'

df2 = pd.read_csv(data_path+'DailyTotalFemaleBirths.csv',index_col='Date',parse_dates=True)
df2.index.freq = 'D'

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdariam import auto_arima

df3 = pd.read_csv()

from statsmodels.tsa.stattools import adfuller

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
        
        

from statsmodels.tsa.statespace.tools import diff





