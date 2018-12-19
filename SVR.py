# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 17:44:02 2018

@author: KarthikM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('SVR Data Set.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2:3]


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


#Creating SVR

from sklearn.svm import SVR
#obj
regressor = SVR(kernel = 'rbf')
#Fitting SVR obj to dataset
regressor.fit(X,y)




y_pred = sc_y.iinverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))



plt.scatter(X,y,color='red')
plt.plot(regressor.predict(X),color = 'blue')
plt.title('Regression model')
plt.xlabel('X axis')
plt.ylabel('Y axis')