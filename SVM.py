# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 13:22:48 2017

@author: Cindy Wang
"""

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
import random
style.use('ggplot')

#write our own svm algorithm
class Support_Vector_Machine:
    def __init__(self, visualization = True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    
    def fit(self, data):
        self.data = data
        # { ||w||: [w, b]}
        opt_dict = { }
        
        transforms = [[1,1], [-1,1], [1,-1], [-1,-1]]
        
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None
        
        # support vectors yi(xi.w+b) = 1
        
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001
                      ]
        # extremely expensive
        b_range_multiple = 5
        # we don't need to take as small of steps with b as we do w
        b_multiple = 5
        latest_optimum = self.max_feature_value *10
        
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        for yi in self.data:
                            for xi in self.data[yi]:
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                    
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                        
                if w[0] < 0:
                    optimized = True
                    print 'Optimized a step.'
                else: 
                    w = w-step
            
            norms = sorted([n for n in opt_dict])
            # ||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_choice = opt_choice[0][0] + step*2
        

    def predict(self, features):
        # sign(x.w +b)
        classification = np.sign(np.dot(np.array(features),self.w) + self.b)
        if classification !=0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
            
        return classification

# only for visualizing all the result
    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]
        
        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1 and nsv = -1 and decision boundary = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]
        
        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        #(w.x+b) =1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1,psv2], 'k')

        #(w.x+b) = -1
        # negetive support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1,nsv2], 'k')
 
        #(w.x+b) = 0
        # decision boundary
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1,db2], 'y--')

        plt.show()
        
        
data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
                          
              1:np.array([[5,1],
                          [6,-1],
                          [7,3],])}

svm1 = Support_Vector_Machine()
svm1.fit(data=data_dict)

predict_me = [[0,10],
              [1,3], 
              [6,-5],
              [5,5],]

for p in predict_me:
    svm1.predict(p)
    
svm1.visualize()










#df = pd.read_csv('loan.csv')
#df= df.drop(['ID'],1)
#
##get X and Y
#x=np.array(df.drop(['RESPONSE'],1))
#y=np.array(df['RESPONSE'])
#
#x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size=0.2)
#
#clf = svm.SVC()
#clf.fit(x_train, y_train)
#
#accuracy=clf.score(x_test, y_test)
#print(accuracy)



