# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:11:23 2019

@author: kmuthu2
"""

#Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the datasets 

dataset = pd.read_csv('KmeansClustering_Data.csv')
X = dataset.iloc[:,[3,4]].values

#Using elbow method to find optimal no of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    #Creating obj
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    #Maxiter and ninit value is default one
    #Now we fit it into our data
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')


#Apllying kmeans clustering
kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(X)

#visualizing the clusters

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Careful')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='Standard')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Target')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='Careless')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='Sensible')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending score (1 - 100)')

