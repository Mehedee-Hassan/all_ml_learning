# -*- coding: utf-8 -*-
"""
Created on Mon May 18 18:44:51 2020

@author: mehedee
"""

import numpy as np
import pandas as pd

np.random.seed(12345)

import matplotlib.pyplot as plt

plt.rc('figure', figsize=(10,6))

previous_max = pd.options.display.max_rows

pd.options.display.max_rows = 20


np.set_printoptions(precision=4, suppress=True)

np.ones((10,5)).shape
np.ones((3,4,5),dtype=np.float64).strides


ints = np.ones(10,dtype= p.unit16)
float = np.ones(10,dtype=np.float32)

# numpy array explanation

ints = np.ones(10,dtype=np.uint16)
floats = np.ones(10,dtype=np.float32)