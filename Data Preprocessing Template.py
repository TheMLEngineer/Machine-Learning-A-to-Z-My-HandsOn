# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 23:14:05 2018

@author: KarthikM
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#missing data get

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN",strategy = "mean",axis = 0) #imputer is object
imputer = imputer.fit(X[:, 1:3])  
# column 1,2 is taken 3 is put coz python gets 1 and 2 when put like 1,3
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding categorical Data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

'''
One hot encoder used to create dummy variables , there's also another method , get dummies
in pandas

'''
#creating obj
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features= [0])

#categorical_features 0 ku reason it has to take only 0th column

#Obj is created so we have to fit it into matrix
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
#we only use label encoder coz its dependent var vector. ML model knows it's categorical
#so it knows if we put 0 or 1 (s r no) It really mwans s r no so it don't need one hot encoder
y = labelencoder_y.fit_transform(y)

#Spliting data set into training and test set

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 0)

#feature scaling

from sklearn.preprocessing import StandardScalar
#create obj
sc_X = StandardScalar()
X_train = sc_X.fit_transform(X_train)
#fit and transform on training data set
X_test = sc_X.transform(X_test)
#Only transform on test data set









     