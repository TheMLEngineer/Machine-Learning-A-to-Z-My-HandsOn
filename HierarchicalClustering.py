# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 12:33:30 2019

@author: KarthikM
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Import the dataset 
dataset = pd.read_csv('HierarchicalClustering_Data.csv')

X = dataset.iloc[:,[3,4]].values

#Using dendogram

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Eucledian Distances')
plt.show()


#Fitting hirearchical clustering
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5,affinity='euclidean',linkage='ward')
#N cluster we fiound by dendrogram
y_hc = hc.fit_predict(X)


#visualizing the clusters

plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='red',label='Careful')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='blue',label='Standard')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='green',label='Target')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='cyan',label='Careless')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='magenta',label='Sensible')
plt.title('Clusters of clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending score (1 - 100)')
plt.legend()
plt.show() 
