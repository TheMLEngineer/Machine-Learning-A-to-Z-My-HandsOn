# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 17:20:40 2019

@author: kmuthu2
"""

#IMPORTING THE LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
#IMPORTING DATASET

dataset = pd.read_csv('UpperConfidenceBound_data.csv')

#Implementing UCB without libraries
#Step 1
d=10
N = 10000   
ads_selected = []
number_of_selections = [0] * d
sums_of_rewards = [0] * d
Total_reward = 0
#Step 2
for n in range(0,N):
    ad = 0  
    max_upper_bound = 0
    for i in range(0,d): # Used to select the right ad
        if number_of_selections[i] > 0:
            average_reward = sums_of_rewards[i] / number_of_selections[i]
            delta_i =math.sqrt(3/2 * math.log(n + 1) / number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400 # this line is used if the user dosen't click any ad then it goes to the next if loop and again the for loop above will be started
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1 
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    Total_reward = Total_reward + reward



