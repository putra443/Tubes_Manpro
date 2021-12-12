# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 23:06:15 2021

@author: yogaa
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split 
    

df = pd.read_csv('weatherAUS.csv',usecols=('Rainfall','Humidity3pm','RainToday'))
X = df.drop(['RainToday'], 1)
Y = df['RainToday']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=404)

from sklearn.tree import DecisionTreeRegressor


DT_model = DecisionTreeRegressor(max_depth=5).fit(X_train,Y_train)
DT_predict = DT_model.predict(X_test)
print(DT_predict)

from sklearn.neighbors import KNeighborsRegressor

KNN_model = KNeighborsRegressor(n_neighbors=2).fit(X_train,Y_train)
KNN_predict = KNN_model.predict(X_test) #Predictions on Testing data
print(KNN_predict)
