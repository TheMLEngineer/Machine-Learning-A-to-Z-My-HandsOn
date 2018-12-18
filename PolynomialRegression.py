# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 23:44:05 2018

@author: KarthikM
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values   # change x to matrix we do 1:2 
y = dataset.iloc[:, 2].values


#Data set is less so we need to train model by all data we have.

"""
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


from sklearn.linear_model import LinearRegression
#obj
lin_reg = LinearRegression()
lin_reg.fit(X,y)

plt.scatter(X,y)

from sklearn.preprocessing import PolynomialFeatures
#obj
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)
# Visualising the results Linear regression

plt.scatter(X,y,color='red')  #plot real data
plt.plot(X,lin_reg.predict(X),color='blue')

plt.title('Truth or Bluff (Linear regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising The polinomial results
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
#The created X-grid is vector so to turn it into matrix we us reshape
plt.scatter(X,y,color='red')  #plot real data
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
'''
#Don't use Xpoly coz X_poly was already defined for an 
existing features of matrix X ...
But the above line is applicable to all matrix of features X

'''
plt.title('Truth or Bluff (Polinomial regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predict using linear regression

lin_reg.predict(6.5) #level is 6.5

#prediction by Polynomian regression 

lin_reg2.predict(poly_reg.fit_transform(6.5))





















