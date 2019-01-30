# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 18:05:55 2019

@author: kmuthu2
"""

# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('NLP_data.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#Making corpus(means a list of texts)... Why we name as corpus coz commonly used in NLP
corpus = []
for i in range(0,1000):
    #Making new cleaned version of the review 

    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i]) 
    #We're adding space in between coz if dont possibility of joining 2 words is there and the combined word make no sense

    #Changing all to lower case
    review = review.lower()
    #spliting words in the review
    review = review.split()
    #removing unrelated words from the list created using split
    #we use set fun so it'll be fast

    #Stemming process (Keeping only present tense of the word)
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #Join the different words in the review (Back from list into string)
    review = ' '.join(review)
    corpus.append(review)

#create bag of model for purpose of tokenization
from sklearn.feature_extraction.text import CountVectorizer
#obj creation
#max features parameter is used to filter names and other unrelated words
cv = CountVectorizer(max_features = 1500)
#Creating sparse matrix
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


accuracy = (55+91)/200 # Based on confusion matrix


