# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 22:13:40 2018

@author: KarthikM
"""
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

feature scaling not needed 4 this coz
 it's simple linear regression, also depends on this data set

"""

#Fitting Simple linear regression into training sets

from sklearn.linear_model import LinearRegression
#regressor is the 1st ML model created in this course , Yes We Did It

regressor = LinearRegression()
regressor.fit(X_train,y_train)


#Predict the test set results

y_pred = regressor.predict(X_test)
#Visualising the training set results

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show() #end of graph 


#Visualising the test set results

plt.scatter(X_test,y_test,color='red')

#regressor is already trained so no need to change X_train
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show() #end of graph 


















