# -*- coding: utf-8 -*-
"""
Created on Tue May 12 09:09:47 2020

@author: mehedee
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('C:\\Users\\mehedee\\Documents\\Python Scripts\\tutorial\\Artificial_Neural_Networks\\ML_DS\\Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values
