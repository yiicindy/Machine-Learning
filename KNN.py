# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 17:54:37 2017

@author: Cindy Wang
"""

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
import random
style.use('fivethirtyeight')


df = pd.read_csv('loan.csv')
df= df.drop(['ID'],1)

#get X and Y
x=np.array(df.drop(['RESPONSE'],1))
y=np.array(df['RESPONSE'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size=0.2)

#using knn in sklearn
knn=neighbors.KNeighborsClassifier(n_neighbors=25)
knn.fit(x_train,y_train)

accuracy=knn.score(x_test, y_test)
#print(accuracy)

####using the knn model created by myself
#1. Design my own knn function
def knneighbor(dataset, newdata, k_num=5):
    if len(dataset) >= k_num:
        warnings.warn('K should be larger than the number of label!') #fundamental criteria
    distance=[]
    for groups in dataset:
        for features in dataset[groups]:
            Euclidean_distance=np.linalg.norm(np.array(features)-np.array(newdata))
            distance.append([Euclidean_distance,groups])
    np.sort(distance[0])
    vote = [i[1] for i in sorted(distance)[:k_num]]
    majority_vote = Counter(vote).most_common(1)[0][0]
    confidence=Counter(vote).most_common(1)[0][1]/ float(k_num)
    #print(majority_vote, confidence)
    return majority_vote, confidence
    
#2. prepare the train data and the test data
full_data=df.astype(float).values.tolist()  #remember to convert to the list first 
test_size=0.25
random.shuffle(full_data)
train_set= {0:[], 1:[]}  # we want to use the dictionary in the knneighbor function
train_data = full_data[:-int(test_size*len(full_data))]
test_set= {0:[], 1:[]}
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

#3. calculate the accuracy of my own knneighbor function 
right=0
total=0
for group in test_set:
    for newdata in test_set[group]:
        result, confidence = knneighbor(train_set, newdata, k_num=25)
        if result== group:
            right += 1
        total+=1
accuracy_knneighbor = float(right)/total
#print(accuracy_knneighbor)

#4. compare two results:
print('the accuracy of knn in scikit-learn is', accuracy, ', and the accuracy of knn written by myself is', accuracy_knneighbor)