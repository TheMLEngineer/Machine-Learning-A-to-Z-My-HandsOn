# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 00:19:46 2019

@author: KarthikM
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
#IMPORTING DATASET

dataset = pd.read_csv('Thompson_Sampling_Data.csv')

#Implementing Thompson sampling without libraries
#Step 1
d=10
N = 10000   
ads_selected = []

number_of_rewards_1 = [0] * d
number_of_rewards_0 = [0] * d

Total_reward = 0
#Step 2
for n in range(0,N):
    ad = 0  
    max_random = 0
    for i in range(0,d): # Used to select the right ad
        random_data = random.betavariate(number_of_rewards_1[i] + 1 , number_of_rewards_0[i] + 1)
        if random_data > max_random:
            max_random = random_data
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
    else:
        number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1

    Total_reward = Total_reward + reward

#Visualizing the results
plt.hist(ads_selected)
plt.title('Histogram of Ad selection')
plt.xlabel('Ads')
plt.ylabel('No of time each Ad was selected')
plt.show()


