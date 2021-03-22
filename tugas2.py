# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 20:06:13 2021

@author: ASUS ROG
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import chardet

#import dataset
#strip value to float

dataset = pd.read_csv('tugas1_datasetnetflix.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
print(X)
print(Y)    
#taking care mising data(nan)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,1:8])
X[:,1:8] = imputer.transform(X[:,1:8])
print(X)
#encoding dependet
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)
print(Y)
#Spliting 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.4,random_state = 1)
print(X_train)
print(Y_train)
print(X_test)
print(Y_test)                       
#feature scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,1:8] = sc.fit_transform(X_train[:,1:8])
X_test[:,1:8] = sc.transform(X_test[:,1:8])
print(X_train)
print(X_test)
