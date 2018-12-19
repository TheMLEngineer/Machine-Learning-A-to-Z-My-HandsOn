# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 17:44:02 2018

@author: KarthikM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('SVR Data Set')
X = dataset.iloc[: 1,2].values
y = dataset.iloc[:,2]

y_pred = regressor.predict(6.5)



plt.scatter(X,y,color='red')
plt.plot(regressor.predict(X),color = 'blue')
plt.title('Regression model')
plt.xlabel('X axis')
plt.ylabel('Y axis')
