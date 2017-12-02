# -*- coding: utf-8 -*-
"""
Created on Fri Dec 01 18:39:47 2017

@author: Administrator
"""

import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

df = pd.read_csv('loan.csv')
df= df.drop(['ID'],1)

#get X and Y
x=np.array(df.drop(['RESPONSE'],1))
y=np.array(df['RESPONSE'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size=0.2)

##############################################
# Check for missing data
np.any(np.isnan(df), axis=0) # axis=0 means checking in column
#In financial data, we often use median to substitute NaN data
# We use Imputer class to process the missing data
imp=Imputer(strategy = 'median', axis=0)
imp.fit(df)
imputed_data = imp.transform(df) # use the median got from fit() to substitute the missing data in the data set
###############################################

# Train the model
rf=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                          max_depth=None, max_features='sqrt', max_leaf_nodes=None,
                          min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0,
                          n_estimators=20, n_jobs=1, oob_score=False, random_state=None, #oob_score means whether to use out-of-bag
                          verbose=0, warm_start=False)
rf.fit(x_train, y_train)

#Score the model
pred = rf.predict(x_test)
accuracy = rf.score(x_test, y_test)
confidence = rf.predict_proba(x_test) #class probabilities for x_test
importance_features = rf.feature_importances_
print 'The accuracy is:', accuracy
print 'The probabilities for each class are:', confidence
print 'the importance of each feature is:', importance_features