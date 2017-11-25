# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:38:18 2017

@author: Cindy Wang
"""

import pandas as pd
import quandl as qd
import math, datetime, time
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn. linear_model import LinearRegression 
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('ggplot')


#prepare the data needed in the regression
df = qd.get('WIKI/GOOGL')
df1=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df1['HL_PCT']=(df1['Adj. High']-df1['Adj. Low']) / df1['Adj. Low'] *100
df1['OC_PCT']=(df1['Adj. Close']-df1['Adj. Open']) / df1['Adj. Open'] * 100
df1=df1[['Adj. Close','HL_PCT','OC_PCT','Adj. Volume']]

df1.fillna(-99999, inplace=True)

shift_num = int(math.ceil(0.001*len(df1)))
df1['label']=df1['Adj. Close'].shift(-shift_num)

#regression
x = np.array(df1.drop(['label'],1))
x= preprocessing.scale(x)
x_last=x[-shift_num:]
x=x[:-shift_num]

df1.dropna(inplace=True)
y=np.array(df1['label'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.2)

rg=LinearRegression()
rg.fit(x_train, y_train)

#pickle build: save the trained model to save time every time using the model to predict
with open ('linearregression.pickle','wb') as f:
    pickle.dump(rg, f)

#pickle use: next time if we want to use this trained model, we can just use the following two lines in any new file
pickle_in = open('linearregression.pickle','rb')
rg = pickle.load(pickle_in)


accuracy= rg.score(x_test, y_test) #R^2
y_predict=rg.predict(x_last) #prediction

#plot the known and predicted close price
df1['forecast'] = np.nan
last_date = df1.iloc[-1].name
last_unix = time.mktime(last_date.timetuple())
one_day = 86400
next_unix = last_unix + one_day

for i in y_predict:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df1.loc[next_date] = [np.nan for _ in range(len(df1.columns)-1)] + [i]

df1['Adj. Close'].plot()
df1['forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Generate random dataset
def random_dataset(num, var, step=2, cor=False):
    val=1
    ys=[]
    for i in range(num):
        y=val+ random.randrange(-var, var)
        ys.append(y)
        if cor and cor == 'pos':
            val+=step
        elif cor and cor=='neg':
            val-= step
    xs = [i for i in range(len(ys))]
    return np.array(xs,dtype=np.float64), np.array(ys, dtype=np.float64)
    
xs, ys = random_dataset(40, 4, 2, cor='pos')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    